"""The four autofill verbs, behind `triage autofill`.

The argument parser lives in `triage/cli.py` with every other verb, so `triage
--help` is one document; the handlers live here so that nothing in the command
line module imports this package until somebody asks for it. That split is the
same one `triage report` uses for the reporting stack, for the same reason: the
cost of a feature should be paid by the people who use it.

**On the `llm` policy.** It is a declared choice that refuses, rather than an
omitted one. A policy silently missing from the list reads as a decision not to
have it; a policy that answers "not yet implemented until part 11" tells a
reader that it is coming and stops them from wiring a pipeline around its
absence. What it must never do is quietly fall back to the model, which would
report LLM numbers that no LLM produced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triage.cli import EXIT_OK, EXIT_USAGE

#: The policy names 5.5 lists, and the one that is not here yet.
POLICIES = ("model", "heuristic", "llm")
NOT_YET = "llm"


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

    if args.policy == NOT_YET:
        print(
            f"error: the {NOT_YET!r} policy is not yet implemented until part 11; "
            f"the engines available today are 'model' and 'heuristic'",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.policy == "model" and not args.weights:
        print(
            "error: --policy model needs --weights pointing at a .npz written by "
            "`triage autofill train` or `triage autofill sweep`",
            file=sys.stderr,
        )
        return EXIT_USAGE

    records = load_split(args.data, args.split)
    model = LogisticModel.load(args.weights) if args.policy == "model" else None
    paths = write_evaluation(
        args.out,
        records=records,
        model=model,
        split=args.split,
        locale=args.locale,
        bootstrap=args.bootstrap,
        seed=args.seed,
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
    print(
        f"outcomes: {paths.outcomes} ({len(paths.runs)} run directories under "
        f"{paths.root / 'runs'})"
    )
    return EXIT_OK


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
    """
    try:
        return HANDLERS[args.autofill_verb](args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
