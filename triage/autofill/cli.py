"""The four autofill verbs, behind `triage autofill`.

The argument parser lives in `triage/cli.py` with every other verb, so `triage
--help` is one document; the handlers live here so that nothing in the command
line module imports this package until somebody asks for it. That split is the
same one `triage report` uses for the reporting stack, for the same reason: the
cost of a feature should be paid by the people who use it.

**On the `llm` policy.** It scores the same split with the local annotator
(6.3) and puts the result in the same table as the other two engines. It needs
a running Ollama and it says so when there is not one, rather than falling back
to the model, which would report LLM numbers that no LLM produced. The response
cache lives in `--database`, so a second run of the same command reproduces the
same numbers without asking the model anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from triage.cli import EXIT_OK, EXIT_USAGE

if TYPE_CHECKING:
    # Types only, so the annotations below are precise while the modules that
    # define them are still imported at the moment a verb runs and not before.
    from triage.autofill.evaluate import ScoredEngine
    from triage.autofill.generator import FieldRecord

#: The policy names 5.5 lists.
POLICIES = ("model", "heuristic", "llm")


def run_generate(args: argparse.Namespace) -> int:
    """Write the corpus, its HTML pages and the metadata that records the knobs."""
    from triage.autofill.generator import GeneratorConfig, write_corpus

    config = GeneratorConfig(n_fields=args.n_fields, locales=tuple(args.locale))
    paths = write_corpus(args.out, config, seed=args.seed)
    counts = {
        split: sum(1 for _ in path.open(encoding="utf-8")) for split, path in paths.splits.items()
    }
    print(
        f"corpus: {args.n_fields} fields in {', '.join(config.locales)} under {paths.root} "
        f"(train {counts['train']}, val {counts['val']}, test {counts['test']}; "
        f"{len(paths.pages)} pages; seed {args.seed})"
    )
    return EXIT_OK


def run_train(args: argparse.Namespace) -> int:
    """Train one model and leave one run directory `triage ingest` can read."""
    from triage.autofill.features import featurise
    from triage.autofill.generator import load_split
    from triage.autofill.model import TrainConfig, train, write_run_dir
    from triage.autofill.sweep import locale_mix

    train_records = load_split(args.data, "train")
    val_records = load_split(args.data, "val")
    config = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        l2=args.l2,
        seed=args.seed,
        epochs=args.epochs,
    )
    result = train(featurise(train_records), featurise(val_records), config)
    directory = write_run_dir(
        Path(args.out) / f"{config.variant}_seed{config.seed}",
        result,
        config,
        locale_mix=locale_mix(train_records),
        extra_config={"n_train": result.n_train, "n_val": result.n_val},
    )
    final = result.history[-1]
    print(
        f"trained: {directory} in {result.seconds:.1f} s "
        f"(val accuracy {final['val/accuracy']:.4f}, val macro F1 {final['val/macro_f1']:.4f})"
    )
    return EXIT_OK


def run_sweep(args: argparse.Namespace) -> int:
    """Train the whole grid, then say what to type next."""
    from triage.autofill.sweep import sweep

    result = sweep(
        args.data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_seeds=args.seeds,
        show_progress=not args.quiet,
    )
    print(
        f"sweep: {len(result.runs)} runs in {result.seconds:.1f} s under {result.root}; "
        f"best {result.best_variant} seed {result.best_seed} at val macro F1 "
        f"{result.best_macro_f1:.4f}, weights in {result.weights.name}"
    )
    print(f"next: triage ingest {result.root} --database triage.db")
    return EXIT_OK


def run_evaluate(args: argparse.Namespace) -> int:
    """Score the chosen engine against the heuristic and write both artifacts."""
    from triage.autofill.evaluate import summarise, write_evaluation
    from triage.autofill.generator import load_split
    from triage.autofill.model import LogisticModel

    if args.policy == "model" and not args.weights:
        print(
            "error: --policy model needs --weights pointing at a .npz written by "
            "`triage autofill train` or `triage autofill sweep`",
            file=sys.stderr,
        )
        return EXIT_USAGE

    from triage.autofill.policy import RewardModel

    records = load_split(args.data, args.split)
    model = LogisticModel.load(args.weights) if args.policy == "model" else None
    # 5.3 learns the decision thresholds on the validation split. When that IS
    # the split being scored there is nothing to load and the comparison is
    # labelled in sample; when it is not, the thresholds come from val and the
    # reported numbers are the ones that cost something.
    fit_records = None if args.split == "val" else load_split(args.data, "val")
    reward = RewardModel.parse(args.penalties) if args.penalties else RewardModel()
    extra_engines = None
    if args.policy == "llm":
        records, extra_engines = _annotate_with_the_llm(args, records)
    paths = write_evaluation(
        args.out,
        records=records,
        model=model,
        split=args.split,
        locale=args.locale,
        bootstrap=args.bootstrap,
        seed=args.seed,
        fit_records=fit_records,
        reward=reward,
        decision_policy=args.decision_policy,
        extra_engines=extra_engines,
    )
    report = summarise(paths)
    for name, engine in sorted(report["engines"].items()):
        print(
            f"{name:>6}: accuracy {engine['accuracy']:.4f}, macro F1 {engine['macro_f1']:.4f}, "
            f"ECE {engine['ece']:.4f} over {engine['n_rows']} rows of {args.locale} {args.split}"
        )
    if paths.calibration is not None:
        import json

        metrics = json.loads(paths.calibration.read_text(encoding="utf-8"))["metrics"]
        print(
            f"calibration: ECE {metrics['val/ece_pre']:.4f} before temperature scaling, "
            f"{metrics['val/ece_post']:.4f} after, at temperature "
            f"{metrics['val/temperature']:.3f}"
        )
    if paths.policy is not None:
        _print_policies(paths.policy)
    print(
        f"outcomes: {paths.outcomes} ({len(paths.runs)} run directories under "
        f"{paths.root / 'runs'})"
    )
    return EXIT_OK


def _annotate_with_the_llm(
    args: argparse.Namespace, records: list[FieldRecord]
) -> tuple[list[FieldRecord], dict[str, ScoredEngine]]:
    """Score the split with the local annotator, and hand back what it scored.

    The rows come back as well as the predictions, because `--limit` shortens
    the pass and the evaluation has to be over exactly the rows that were
    classified. Scoring 200 rows and reporting them against a split of 800 would
    be a number about nothing, and the shape of that mistake is a length
    mismatch that `evaluate_engines` refuses anyway.
    """
    from triage.autofill.generator import load_split
    from triage.core.store import Store
    from triage.llm.annotator import ENGINE_LLM, Annotator, AnnotatorConfig
    from triage.llm.ollama_client import OllamaClient

    client = OllamaClient(host=args.llm_host, chat_model=args.llm_model)
    chat_model = client.resolve_chat_model(args.llm_model)
    if chat_model != args.llm_model:
        print(f"note: using {chat_model}; {args.llm_model} is not pulled", file=sys.stderr)

    rows = [record for record in records if args.locale == "all" or record.locale == args.locale]
    examples = [
        record
        for record in load_split(args.data, "train")
        if args.locale == "all" or record.locale == args.locale
    ]
    settings = AnnotatorConfig(
        k=args.llm_k, chat_model=chat_model, limit=args.llm_limit, engine=ENGINE_LLM
    )
    with Store(args.database) as store:
        run = Annotator(client, store, settings, examples=examples).annotate(rows)
    print(run.describe())
    return rows[: len(run.annotations)], {ENGINE_LLM: run.scored()}


def _print_policies(path: Path) -> None:
    """The contextual bandit comparison, read back from what was written.

    Read back rather than kept in memory, so that what the terminal says is what
    the artifact says and the two cannot drift.
    """
    from triage.autofill.policy import load_policy

    comparison = load_policy(path)
    fitted = "in sample" if comparison.in_sample else f"fitted on {comparison.fit_split}"
    print(f"decision policy (contextual bandit, {fitted}): {comparison.reward.describe()}")
    print(f"{'policy':>19}  {'fill rate':>9}  {'reward':>8}  {'correction cost':>15}")
    for outcome in comparison.outcomes:
        marker = " *" if outcome.name == comparison.chosen else "  "
        print(
            f"{outcome.name:>19}  {outcome.fill_rate:>9.4f}  {outcome.expected_reward:>8.4f}  "
            f"{outcome.correction_cost:>15.4f}{marker}"
        )
    thompson = comparison.thompson
    print(
        f"{'thompson':>19}  {thompson.n_filled / thompson.n_rows:>9.4f}  "
        f"{thompson.expected_reward:>8.4f}  {thompson.correction_cost:>15.4f}"
    )
    print(
        f"regret of the Thompson run against {thompson.best_fixed}: "
        f"{thompson.cumulative_regret:.2f} over {thompson.n_rows} decisions "
        f"({thompson.regret_per_decision:.4f} each); policy exported to {path}"
    )


HANDLERS = {
    "generate": run_generate,
    "train": run_train,
    "sweep": run_sweep,
    "evaluate": run_evaluate,
}


def run(args: argparse.Namespace) -> int:
    """Dispatch on the autofill verb, answering for its own domain errors.

    A `ValueError` from this package is a fact about the request (a locale with
    no rows, a bootstrap of one), which is what exit 2 means. The command line's
    own error boundary does not list it, and should not: a `ValueError` from the
    statistics layer IS a bug and keeps its traceback.

    An `LlmError` reaches here only from `--policy llm`, and it already names
    the command that fixes it, so it is the same kind of answer: exit 2 with the
    message. It is imported inside the handler so that the other three policies
    never import the model client at all.
    """
    from triage.llm.ollama_client import LlmError

    try:
        return HANDLERS[args.autofill_verb](args)
    except (LlmError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
