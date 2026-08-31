"""The three `triage llm` verbs, and the error boundary they answer for.

The argument parser lives in `triage/cli.py` with every other verb, so that
`triage --help` is one document and so that building the parser costs no import
of this package. The handlers live here, and are imported the moment one of
these verbs runs and not before: a core install that never asks for a model
never imports a model client.

**Why this module has its own error boundary.** An `LlmUnavailableError` is a
fact about the machine (Ollama is not running, a model is not pulled) and its
message already names the command that fixes it, so it is exit 2 with that
message rather than a traceback. The command line's own boundary does not list
it and should not: `triage/cli.py` would have to import this package to name the
exception, which is exactly the import this split exists to avoid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from triage.cli import EXIT_OK, EXIT_USAGE

if TYPE_CHECKING:
    from triage.llm.ollama_client import OllamaClient


def _client(args: argparse.Namespace) -> OllamaClient:
    """One configured client, from the flags every llm verb shares."""
    from triage.llm.ollama_client import OllamaClient

    return OllamaClient(
        host=args.host,
        chat_model=args.model,
        num_ctx=args.num_ctx,
        seed=args.llm_seed,
    )


def _resolved(client: OllamaClient, args: argparse.Namespace) -> str:
    """The chat model to use: the one asked for, or the fallback that is here.

    Resolved once per command and printed, because a run that quietly used the
    7B fallback when the reader believed it was using the 12B would make every
    number it produced unreproducible for anybody reading the flag.
    """
    chosen = client.resolve_chat_model(args.model)
    if chosen != args.model:
        print(
            f"note: {args.model} is not pulled on {args.host}; using {chosen} instead",
            file=sys.stderr,
        )
    return chosen


def run_annotate(args: argparse.Namespace) -> int:
    """Classify a split with the local model, and write both artifact shapes."""
    from triage.autofill.evaluate import summarise, write_evaluation
    from triage.autofill.generator import load_split
    from triage.core.store import Store
    from triage.llm.annotator import ENGINE_LLM, Annotator, AnnotatorConfig, run_ablation

    client = _client(args)
    chat_model = _resolved(client, args)
    records = load_split(args.data, args.split)
    examples = load_split(args.data, "train")
    settings = AnnotatorConfig(
        k=args.k,
        chat_model=chat_model,
        embed_model=args.embed_model,
        limit=args.limit,
        engine=ENGINE_LLM,
    )

    with Store(args.database) as store:
        if args.ablation:
            result = run_ablation(
                client,
                store,
                records=records,
                examples=examples,
                out_dir=args.out,
                split=args.split,
                locale=args.locale,
                bootstrap=args.bootstrap,
                seed=args.seed,
                config=settings,
            )
            print(result.zero_shot.describe())
            print(result.retrieval.describe())
            print(f"\nablation: {result.headline()}")
            print(f"baseline {result.baseline}; runs under {result.root / 'runs'}")
            print(
                f"measured through the comparison layer in "
                f"{result.root / 'ablation.db'} in {result.seconds:.1f} s"
            )
            return EXIT_OK

        rows = [
            record for record in records if args.locale == "all" or record.locale == args.locale
        ]
        pool = [
            record for record in examples if args.locale == "all" or record.locale == args.locale
        ]
        run = Annotator(client, store, settings, examples=pool).annotate(rows)
        print(run.describe())
        paths = write_evaluation(
            args.out,
            records=rows[: len(run.annotations)],
            model=None,
            split=args.split,
            locale=args.locale,
            bootstrap=args.bootstrap,
            seed=args.seed,
            extra_engines={ENGINE_LLM: run.scored()},
        )

    report = summarise(paths)
    for name, engine in sorted(report["engines"].items()):
        print(
            f"{name:>14}: accuracy {engine['accuracy']:.4f}, macro F1 {engine['macro_f1']:.4f} "
            f"over {engine['n_rows']} rows of {args.locale} {args.split}"
        )
    print(f"outcomes: {paths.outcomes} ({len(paths.runs)} run directories under {paths.root})")
    return EXIT_OK


def run_summarize(args: argparse.Namespace) -> int:
    """Write the grounded plain language summary of one analysis to stdout."""
    from triage.cli import analyse_database
    from triage.core.store import Store
    from triage.llm.summarizer import summarise_report

    client = _client(args)
    chat_model = _resolved(client, args)
    analysis = analyse_database(args, sensitivity=False)

    with Store(args.database) as store:
        summary = summarise_report(
            client,
            store,
            analysis.report,
            refusals=[refusal.describe() for refusal in analysis.refusals],
            model=chat_model,
        )

    print(f"\n{summary.heading}\n")
    print(summary.text or "(the grounding pass removed every sentence)")
    print(f"\n{summary.caption()}")
    if summary.dropped_sentences:
        print("\ndropped for containing unsupported numbers:", file=sys.stderr)
        for sentence in summary.dropped_sentences:
            print(f"  - {sentence}", file=sys.stderr)
    if args.output:
        Path(args.output).write_text(
            f"{summary.heading}\n\n{summary.text}\n\n{summary.caption()}\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"written to {args.output}")
    return EXIT_OK


def run_ask(args: argparse.Namespace) -> int:
    """Answer one scoped question, or refuse it naming what can be answered."""
    from triage.core.store import Store
    from triage.llm.ask import ask

    database = Path(args.database)
    if not database.is_file():
        print(
            f"error: no such database: {database}. `triage llm ask` reads a database "
            f"`triage ingest` has already written",
            file=sys.stderr,
        )
        return EXIT_USAGE

    client = _client(args)
    chat_model = _resolved(client, args)
    with Store(database) as store:
        result = ask(
            client,
            store,
            args.question,
            root=args.root,
            baseline=args.baseline,
            k=args.k,
            model=chat_model,
            embed_model=args.embed_model,
        )

    print(f"\n{result.text}\n")
    if result.refused:
        return EXIT_USAGE
    print(result.describe())
    return EXIT_OK


HANDLERS = {"annotate": run_annotate, "summarize": run_summarize, "ask": run_ask}


def run(args: argparse.Namespace) -> int:
    """Dispatch on the llm verb, answering for this layer's own failures.

    An `LlmError` names the command that would fix it (`ollama serve`,
    `ollama pull ...`), so it is a usage error with that message rather than a
    traceback. A `ValueError` from this package is a fact about the request, the
    same convention `triage autofill` follows.
    """
    from triage.llm.ollama_client import LlmError

    try:
        return HANDLERS[getattr(args, "llm_verb", "ask")](args)
    except (LlmError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
