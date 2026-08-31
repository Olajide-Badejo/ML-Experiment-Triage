"""Four verbs: `ingest`, `compare`, `report`, `demo`.

    triage ingest experiments/results/demo_sweep --database triage.db
    triage compare --database triage.db --baseline lr0.0010_bs32
    triage report  --database triage.db --baseline lr0.0010_bs32 --output report.html
    triage demo

`ingest` is the only verb that touches log files. `compare` and `report` read
the database, so they are fast enough to rerun freely, and both are pure
functions of the database plus the recorded permutation seed. `demo` runs all
three over a synthetic sweep whose ground truth is known, in a temporary
directory, so a fresh `pip install` can show what the tool does.

**Exit codes mean one thing each.** Exit 1 used to mean both "some runs failed
to parse" and "this tool crashed", which is exactly the distinction a CI gate
needs to make; and `compare` exited 0 having performed zero comparisons, so a
gate could pass because nothing was tested. The table is in `EXIT_CODES` and
printed by `--help`.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sqlite3
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triage import __version__
from triage.analysis.comparison import (
    ComparisonConfig,
    ComparisonError,
    ComparisonRefusal,
    ComparisonResult,
    compare_all,
)
from triage.analysis.regression import RegressionConfig, TriageReport, classify, rank
from triage.analysis.sensitivity import SensitivityResult, analyse
from triage.calibration import SUMMARY
from triage.core.experiment import Experiment, SeriesError
from triage.core.store import Store, StoreError
from triage.ingest import ingest
from triage.parsers import ParseError, parsers_for

DEFAULT_DATABASE = "triage.db"

#: Accepted values of `--log-level`, lowest detail last so `--help` reads in
#: the order a reader would raise it.
LOG_LEVELS = ("debug", "info", "warning", "error")

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_USAGE = 2
EXIT_RUNS_FAILED = 3
EXIT_NO_COMPARISONS = 4

#: What each exit code means, printed in `--help` because a caller writing a CI
#: gate should not have to read this file to find out. One meaning per code:
#: exit 1 covered two of them and a job could not tell a crash from a result.
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "success",
    EXIT_CRASH: "a bug in triage: an unexpected exception, with its traceback",
    EXIT_USAGE: "a usage error or a failure this tool foresaw, reported as `error: ...`",
    EXIT_RUNS_FAILED: "the work was done, but at least one run failed to parse",
    EXIT_NO_COMPARISONS: "the run finished having performed zero comparisons",
}

#: The exceptions the CLI answers for. Anything outside this list is a bug in
#: this package rather than a fact about the input, and it keeps its traceback
#: and exit 1 for exactly that reason. `KeyError` is here despite being a
#: builtin because the store and the experiment model both raise it as a
#: not found signal; `_message` below unwraps its repr quoting.
HANDLED_ERRORS = (
    ComparisonError,
    SeriesError,
    ParseError,
    StoreError,
    OSError,
    sqlite3.Error,
    KeyError,
)


def configure_logging(level: str) -> None:
    """Send every `triage.*` log record to stderr at `level`.

    The package used to import `logging` nowhere at all, so the only way to
    learn why a directory had not been read as a run was to add prints and
    rebuild. Records go to stderr because they are diagnostics: stdout carries
    the verdict table and the report path, and a reader piping those somewhere
    must not get progress and warnings mixed into the data.

    Configuration is scoped to the `triage` logger rather than the root, and
    `propagate` is turned off, so importing this package as a library does not
    reconfigure the host application's logging.
    """
    logger = logging.getLogger("triage")
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)


# Measured by the calibration suite in tests/statistics and reproduced by
# `make test`. Carried into every report footer so a reader never has to take
# the error rates on trust. The numbers live in triage/calibration.py, which is
# the single source the README, the PDFs, the figures and this all read.
CALIBRATION_NOTE = SUMMARY


def bounded_float(
    low: float | None = None,
    high: float | None = None,
    low_inclusive: bool = False,
    high_inclusive: bool = True,
) -> Callable[[str], float]:
    """An argparse `type=` that refuses a number outside a parameter's domain.

    Every numeric flag on this tool has a domain, and none of them were checked:
    `--alpha 2.0 --fdr -1 --permutations -5` were all accepted in silence, and
    what they produced was not an error but a wrong answer, which is far worse.
    `--window-fraction 5` did raise, eight frames down and as a traceback.

    Validation belongs at the parse boundary because that is the only place that
    can name the flag the user typed. The bounds are the domain of the quantity,
    not a judgement about good values: a false discovery rate of 1.0 is a
    perfectly meaningful instruction to accept everything, and is allowed.
    """

    def parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{raw!r} is not a number") from None
        if low is not None and (value < low if low_inclusive else value <= low):
            edge = "at or above" if low_inclusive else "above"
            raise argparse.ArgumentTypeError(f"must be {edge} {low:g}, got {value:g}")
        if high is not None and (value > high if high_inclusive else value >= high):
            edge = "at or below" if high_inclusive else "below"
            raise argparse.ArgumentTypeError(f"must be {edge} {high:g}, got {value:g}")
        return value

    return parse


def bounded_int(low: int, high: int | None = None) -> Callable[[str], int]:
    """The same, for a count. `low` is inclusive, because counts start at one."""

    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{raw!r} is not a whole number") from None
        if value < low:
            raise argparse.ArgumentTypeError(f"must be at or above {low}, got {value}")
        if high is not None and value > high:
            raise argparse.ArgumentTypeError(f"must be at or below {high}, got {value}")
        return value

    return parse


#: A probability: strictly inside zero, up to and including one. Zero is
#: excluded because a gate no result can clear is not a gate.
probability = bounded_float(low=0.0, high=1.0)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"path to the triage database (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "for CI logs: no progress bars and no count summary. Everything else "
            "always prints, because everything else is either a result (the verdict "
            "table, the caveats, the refusals, the report path) or the provenance "
            "of one (the gates, the families, the statistic, the permutation seed), "
            "and a number without its provenance is not something this tool emits"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=LOG_LEVELS,
        help=(
            "diagnostic detail on stderr (default: warning). `debug` explains every "
            "discovery decision, which is the way to answer why a directory was not "
            "read as a run"
        ),
    )


def add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline", required=True, help="run id or variant key to compare against"
    )
    parser.add_argument("--tags", nargs="*", default=None, help="metrics to compare (default: all)")
    parser.add_argument(
        "--window-fraction",
        type=bounded_float(low=0.0, high=1.0),
        default=ComparisonConfig.window_fraction,
        help="final window as a fraction of the run, in (0, 1]",
    )
    parser.add_argument(
        "--window-minimum",
        # Two, not one: a "window" of a single point has no spread, so every
        # statistic computed over it is that point and every interval is empty.
        type=bounded_int(low=2),
        default=ComparisonConfig.window_minimum,
        help="smallest final window in points (at least 2)",
    )
    parser.add_argument(
        "--permutations",
        type=bounded_int(low=1),
        default=ComparisonConfig.n_permutations,
        help="resamples when exhaustive enumeration is too large (at least 1)",
    )
    parser.add_argument(
        "--alpha",
        type=probability,
        default=RegressionConfig.alpha,
        help=(
            "admissibility bound: a design whose smallest attainable p value exceeds "
            "this is reported as inconclusive rather than as no change. The "
            "statistical gate is --fdr"
        ),
    )
    parser.add_argument(
        "--practical-threshold",
        type=bounded_float(low=0.0, low_inclusive=True, high=None),
        default=None,
        help=(
            "practical gate, as a relative percent (default: "
            f"{RegressionConfig.practical_threshold_pct:g})"
        ),
    )
    parser.add_argument(
        "--practical-threshold-absolute",
        type=bounded_float(low=0.0, low_inclusive=True, high=None),
        default=None,
        help=(
            "practical gate in the metric's own units instead of as a percentage, "
            "which is the right gate for a metric whose baseline can be zero: a "
            "relative gate divides by the baseline and quietly downgrades a real "
            "regression to nothing when it cannot"
        ),
    )
    parser.add_argument(
        "--fdr",
        type=probability,
        default=RegressionConfig.false_discovery_rate,
        help=(
            "Benjamini Hochberg false discovery rate, which is the statistical gate: "
            "a finding is significant when its adjusted p is at or below this "
            f"(default: {RegressionConfig.false_discovery_rate:g})"
        ),
    )
    parser.add_argument(
        "--seed",
        # numpy's `default_rng` refuses a negative seed, so a negative one here
        # would have failed inside the first permutation rather than at the flag.
        type=bounded_int(low=0),
        default=ComparisonConfig.seed,
        help="permutation seed, printed in every report (a non negative integer)",
    )
    parser.add_argument(
        "--higher-is-better",
        action="append",
        metavar="TAG",
        help=(
            "treat TAG as a metric where a larger value is an improvement, "
            "overriding the guess made from its name (repeatable)"
        ),
    )
    parser.add_argument(
        "--lower-is-better",
        action="append",
        metavar="TAG",
        help="the same override in the other direction (repeatable)",
    )


def directions_from(args: argparse.Namespace) -> dict[str, bool]:
    """The per tag direction overrides named on the command line.

    Direction is inferred from the tag name, which is a convenience that is
    occasionally wrong: a name this tool has never seen defaults to lower is
    better, and a metric can be spelled in a way no table will catch. These two
    flags are how that is corrected from outside the code.
    """
    higher = list(getattr(args, "higher_is_better", None) or [])
    lower = list(getattr(args, "lower_is_better", None) or [])
    both = sorted(set(higher) & set(lower))
    if both:
        raise ComparisonError(
            f"{', '.join(both)} given as both higher and lower is better; a metric "
            f"improves in one direction, so name it in one flag"
        )
    return dict.fromkeys(higher, True) | dict.fromkeys(lower, False)


def practical_gate_from(args: argparse.Namespace) -> tuple[float | None, float | None]:
    """The one practical threshold in force, as `(percent, absolute)`.

    `RegressionConfig` requires exactly one of the two, and will not guess which
    one a caller meant. The command line is where the guessing is legitimate:
    naming the absolute gate is an unambiguous request for it, so the percentage
    default steps aside. Naming both is not a request for anything.
    """
    percent = getattr(args, "practical_threshold", None)
    absolute = getattr(args, "practical_threshold_absolute", None)
    if percent is not None and absolute is not None:
        raise ComparisonError(
            "--practical-threshold and --practical-threshold-absolute set two "
            "different practical gates; a finding clears one gate, so name one"
        )
    if absolute is not None:
        return None, absolute
    return (RegressionConfig.practical_threshold_pct if percent is None else percent), None


def configs_from(args: argparse.Namespace) -> tuple[ComparisonConfig, RegressionConfig]:
    percent, absolute = practical_gate_from(args)
    return (
        ComparisonConfig(
            window_fraction=args.window_fraction,
            window_minimum=args.window_minimum,
            n_permutations=args.permutations,
            alpha=args.alpha,
            seed=args.seed,
            directions=directions_from(args),
        ),
        RegressionConfig(
            alpha=args.alpha,
            practical_threshold_pct=percent,
            practical_threshold_absolute=absolute,
            false_discovery_rate=args.fdr,
        ),
    )


def exit_code_table() -> str:
    """The exit codes, as `--help` prints them."""
    rows = "\n".join(f"  {code}  {meaning}" for code, meaning in sorted(EXIT_CODES.items()))
    return f"exit codes:\n{rows}\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage",
        description=__doc__,
        epilog=exit_code_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"triage {__version__}")
    subparsers = parser.add_subparsers(dest="verb", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="parse a directory of runs into the database"
    )
    ingest_parser.add_argument("path", help="directory holding one subdirectory per run")
    ingest_parser.add_argument(
        "--force", action="store_true", help="reparse even when the source is unchanged"
    )
    ingest_parser.add_argument(
        "--outcomes",
        action="store_true",
        help=(
            "read step free JSONL under this path as cross sectional outcome rows "
            "(one row per scored unit) even when the file declares no record schema. "
            "A file that DOES declare one is recognised without this flag; the flag is "
            "how you say that an undeclared step free file is evaluation output rather "
            "than a log this tool failed to understand"
        ),
    )
    add_common(ingest_parser)

    compare_parser = subparsers.add_parser("compare", help="print the ranked verdict table")
    add_common(compare_parser)
    add_analysis_options(compare_parser)

    report_parser = subparsers.add_parser("report", help="write the self contained HTML report")
    add_common(report_parser)
    add_analysis_options(report_parser)
    report_parser.add_argument(
        "--output", default="triage_report.html", help="where to write the HTML report"
    )
    report_parser.add_argument("--title", default="ML Experiment Triage", help="report heading")
    report_parser.add_argument(
        "--autofill",
        metavar="DIR",
        default=None,
        help=(
            "an output directory from `triage autofill evaluate`. Adds the reference "
            "workload's calibration section: expected and maximum calibration error "
            "before and after temperature scaling, and the reliability diagram. A "
            "directory with no calibration file adds nothing and is not an error"
        ),
    )
    report_parser.add_argument(
        "--llm-summary",
        action="store_true",
        help=(
            "add a plain language summary written by a local model, with every number "
            "in it checked against this report and every sentence carrying an "
            "unsupported one deleted. OFF by default, and that is a correctness "
            "property rather than caution: without it the page is a byte identical "
            "function of the database, which is what the reproducibility gate checks"
        ),
    )
    add_llm_options(report_parser)

    demo_parser = subparsers.add_parser(
        "demo",
        help="synthesise the sweep in a temporary directory and report on it end to end",
        description=(
            "Generate a synthetic sweep whose ground truth is known, ingest it, compare "
            "every condition against the baseline and write the HTML report. Nothing is "
            "left behind but the report, unless --keep says otherwise."
        ),
    )
    demo_parser.add_argument(
        "--output",
        default="triage_demo_report.html",
        help="where to write the demo report (default: triage_demo_report.html)",
    )
    demo_parser.add_argument(
        "--keep",
        metavar="DIR",
        default=None,
        help=(
            "keep the synthesised sweep and its database under DIR instead of a "
            "temporary directory that is deleted afterwards"
        ),
    )
    demo_parser.add_argument(
        "--quiet", action="store_true", help="no progress bars and no count summary"
    )
    demo_parser.add_argument(
        "--log-level", default="warning", choices=LOG_LEVELS, help=argparse.SUPPRESS
    )

    add_autofill(subparsers)
    add_llm(subparsers)
    return parser


#: The verb `triage ask` forwards to. Section 6.5 writes the question answering
#: verb as `triage ask "..."` and Section 5.5 lists it as `triage llm ask`; both
#: spellings are in the specification, so both work and one of them is an alias
#: rather than a second implementation.
ASK_ALIAS = "ask"


def add_llm_options(parser: argparse.ArgumentParser) -> None:
    """The flags every `triage llm` verb shares: where the model is and which."""
    from triage.llm.embeddings import DEFAULT_EMBEDDING_MODEL
    from triage.llm.ollama_client import (
        DEFAULT_CHAT_MODEL,
        DEFAULT_HOST,
        DEFAULT_NUM_CTX,
        MAX_NUM_CTX,
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=(
            f"where Ollama is listening (default: {DEFAULT_HOST}). Everything in this "
            f"layer is local; there is no hosted model to fall back to"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CHAT_MODEL,
        help=(
            f"chat model (default: {DEFAULT_CHAT_MODEL}). When it is not pulled, the "
            f"faster fallback is used and a note says so on stderr"
        ),
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            f"embedding model (default: {DEFAULT_EMBEDDING_MODEL}). Its prompt prefixes "
            f"come from the model record, so a model with no record is refused rather "
            f"than embedded without them"
        ),
    )
    parser.add_argument(
        "--num-ctx",
        type=bounded_int(low=256, high=MAX_NUM_CTX),
        default=DEFAULT_NUM_CTX,
        help=(
            f"context window (default: {DEFAULT_NUM_CTX}, maximum {MAX_NUM_CTX}). The "
            f"ceiling is the VRAM measurement in the docs, not a preference"
        ),
    )
    parser.add_argument(
        "--llm-seed",
        type=bounded_int(low=0),
        default=0,
        help="the seed sent with every request, at temperature 0 (default: 0)",
    )


def add_ask_options(parser: argparse.ArgumentParser) -> None:
    """The question answering flags, shared by `triage llm ask` and `triage ask`."""
    parser.add_argument("question", help="a question about this database, in plain English")
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"the database to read (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "the baseline to compare against, needed for any question about verdicts, "
            "refusals or sensitivity. Without it those questions are refused rather "
            "than answered against an arbitrary run"
        ),
    )
    parser.add_argument(
        "--root", default=".", help="where the prose corpus is (default: the current directory)"
    )
    parser.add_argument(
        "--k", type=bounded_int(low=1), default=4, help="prose chunks retrieved (default: 4)"
    )
    add_llm_options(parser)
    parser.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-level", default="warning", choices=LOG_LEVELS, help=argparse.SUPPRESS
    )


def add_llm(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Section 6's three verbs, plus the `triage ask` alias 6.5 spells out.

    The flag declarations import `triage.llm.ollama_client` and
    `triage.llm.embeddings` for their default values, which is deliberate and
    cheap: both are standard library plus numpy, they exist so the defaults in
    `--help` are the ones the code actually uses rather than a second copy that
    can drift, and neither reaches the network at import time.
    """
    llm_parser = subparsers.add_parser(
        "llm",
        help="local model verbs: annotate, summarize, ask (Ollama on localhost)",
        description=(
            "Retrieval augmented field annotation, a grounded plain language summary of "
            "a report, and scoped question answering. Everything runs against a local "
            "Ollama; nothing here calls a hosted model."
        ),
    )
    verbs = llm_parser.add_subparsers(dest="llm_verb", required=True)

    annotate = verbs.add_parser(
        "annotate",
        help="classify a split with the local model, using its nearest labelled examples",
    )
    annotate.add_argument("--data", required=True, metavar="DIR", help="a generated corpus")
    annotate.add_argument(
        "--out", required=True, metavar="DIR", help="where the runs and the outcomes file go"
    )
    annotate.add_argument(
        "--split", default="val", choices=["train", "val", "test"], help="which split to classify"
    )
    annotate.add_argument(
        "--locale", default="de_DE", choices=["all", "en_US", "de_DE"], help="restrict to a locale"
    )
    annotate.add_argument(
        "--k", type=bounded_int(low=1, high=32), default=8, help="few shot examples (default: 8)"
    )
    annotate.add_argument(
        "--limit",
        type=bounded_int(low=1),
        default=None,
        help="classify only the first N rows: a quick pass, recorded as such",
    )
    annotate.add_argument(
        "--bootstrap", type=bounded_int(low=2), default=5, help="bootstrap replicates (default: 5)"
    )
    annotate.add_argument("--seed", type=bounded_int(low=0), default=0, help="resampling seed")
    annotate.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "score the split twice, zero shot and retrieval augmented, and report the "
            "difference as a verdict with a p value through the comparison layer. This "
            "is how the claim that retrieval helps is checked rather than asserted"
        ),
    )
    add_common(annotate)
    add_llm_options(annotate)

    summarize = verbs.add_parser(
        "summarize",
        help="a grounded plain language summary of one analysis, numbers checked",
        description=(
            "Renders the whole report into the prompt, generates a 150 to 250 word "
            "summary, then deletes every sentence containing a number the report does "
            "not support and says how many were dropped."
        ),
    )
    add_common(summarize)
    add_analysis_options(summarize)
    summarize.add_argument(
        "--output", default=None, metavar="FILE", help="also write the summary to a file"
    )
    add_llm_options(summarize)

    ask_parser = verbs.add_parser(
        "ask",
        help="answer a scoped question about this database, or refuse it",
        description=(
            "Retrieval runs over the prose corpus; every number in the answer is "
            "computed from the database by a SQL template and checked afterwards. A "
            "question this cannot ground is refused, naming what it can answer."
        ),
    )
    add_ask_options(ask_parser)

    alias = subparsers.add_parser(
        ASK_ALIAS,
        help="the same as `triage llm ask`, which is how Section 6.5 spells it",
    )
    add_ask_options(alias)


def add_autofill(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The reference workload's four verbs, per 5.5.

    Declared here rather than in `triage/autofill/cli.py` so that `triage
    --help` is one document and so that building the parser costs no import of
    that package; the handlers are imported when one of these verbs runs.
    """
    autofill_parser = subparsers.add_parser(
        "autofill",
        help="this repository's reference workload: form field classification",
        description=(
            "Generate a synthetic corpus of browser form fields, train a numpy "
            "classifier on it, sweep the grid, and evaluate against a heuristic "
            "baseline. The sweep writes ordinary run directories, so `triage ingest` "
            "and `triage compare` read them with no autofill specific flag."
        ),
    )
    verbs = autofill_parser.add_subparsers(dest="autofill_verb", required=True)

    generate = verbs.add_parser("generate", help="write the synthetic corpus and its HTML pages")
    generate.add_argument("--out", required=True, metavar="DIR", help="where to write the corpus")
    generate.add_argument(
        "--locale",
        nargs="+",
        default=["en_US", "de_DE"],
        choices=["en_US", "de_DE"],
        help="locales to generate (default: both)",
    )
    generate.add_argument("--seed", type=bounded_int(low=0), default=0, help="corpus seed")
    generate.add_argument(
        "--n-fields",
        type=bounded_int(low=20),
        default=4000,
        help="how many labelled fields to write (default: 4000)",
    )
    add_common(generate)

    train_parser = verbs.add_parser("train", help="train one model into a run directory")
    train_parser.add_argument("--data", required=True, metavar="DIR", help="a generated corpus")
    train_parser.add_argument(
        "--out", required=True, metavar="RUNS_DIR", help="where the run directory goes"
    )
    train_parser.add_argument(
        "--lr", type=bounded_float(low=0.0), default=0.1, help="learning rate"
    )
    train_parser.add_argument(
        "--batch-size", type=bounded_int(low=1), default=64, help="minibatch size"
    )
    train_parser.add_argument(
        "--l2", type=bounded_float(low=0.0, low_inclusive=True), default=1e-4, help="L2 strength"
    )
    train_parser.add_argument("--seed", type=bounded_int(low=0), default=0, help="training seed")
    train_parser.add_argument(
        "--epochs", type=bounded_int(low=1), default=8, help="passes over the training split"
    )
    add_common(train_parser)

    sweep_parser = verbs.add_parser(
        "sweep",
        help="train the whole demo grid: lr {0.03, 0.1, 0.3} x l2 {0, 1e-4} x seeds 0 to 4",
    )
    sweep_parser.add_argument("--data", required=True, metavar="DIR", help="a generated corpus")
    sweep_parser.add_argument(
        "--out", required=True, metavar="RUNS_DIR", help="where the 30 run directories go"
    )
    sweep_parser.add_argument(
        "--epochs", type=bounded_int(low=1), default=8, help="passes over the training split"
    )
    sweep_parser.add_argument(
        "--batch-size", type=bounded_int(low=1), default=64, help="minibatch size"
    )
    sweep_parser.add_argument(
        "--seeds",
        type=bounded_int(low=1, high=5),
        default=5,
        help=(
            "how many of the five seeds to run (default: 5). Fewer is a smoke test: "
            "three or more is what the seed replicated mode needs"
        ),
    )
    add_common(sweep_parser)

    evaluate = verbs.add_parser(
        "evaluate", help="score a policy against the heuristic baseline and write both artifacts"
    )
    evaluate.add_argument("--data", required=True, metavar="DIR", help="a generated corpus")
    evaluate.add_argument(
        "--out", required=True, metavar="DIR", help="where the runs and the outcomes file go"
    )
    evaluate.add_argument(
        "--weights", metavar="W.npz", default=None, help="a model written by train or sweep"
    )
    evaluate.add_argument(
        "--policy",
        default="model",
        choices=["model", "heuristic", "llm"],
        help=(
            "which engine is under study. The heuristic is always scored beside it, "
            "because a number with nothing to compare against is not a result. "
            "`llm` runs the local annotator over the split and needs a running Ollama"
        ),
    )
    evaluate.add_argument(
        "--llm-host",
        default="http://localhost:11434",
        help="where Ollama is listening, for --policy llm (default: http://localhost:11434)",
    )
    evaluate.add_argument(
        "--llm-model",
        default="mistral-nemo:12b-instruct-2407-q4_K_M",
        help="chat model for --policy llm; the faster fallback is used when it is absent",
    )
    evaluate.add_argument(
        "--llm-k",
        type=bounded_int(low=1, high=32),
        default=8,
        help="few shot examples retrieved per field for --policy llm (default: 8)",
    )
    evaluate.add_argument(
        "--llm-limit",
        type=bounded_int(low=1),
        default=None,
        help=(
            "classify only the first N rows with --policy llm. The evaluation is then "
            "over exactly those rows and says so, rather than over the whole split"
        ),
    )
    evaluate.add_argument(
        "--bootstrap",
        type=bounded_int(low=2),
        default=5,
        help=(
            "bootstrap replicates, one run directory each, replicate k as seed k "
            "(default: 5, which is what the seed replicated mode wants)"
        ),
    )
    evaluate.add_argument(
        "--split", default="val", choices=["train", "val", "test"], help="which split to score"
    )
    evaluate.add_argument(
        "--locale", default="all", choices=["all", "en_US", "de_DE"], help="restrict to one locale"
    )
    evaluate.add_argument("--seed", type=bounded_int(low=0), default=0, help="resampling seed")
    evaluate.add_argument(
        "--decision-policy",
        default="per_type_threshold",
        choices=["always_fill", "never_fill", "global_threshold", "per_type_threshold"],
        help=(
            "which fill or skip policy is EXPORTED for the agentic demo (default: "
            "per_type_threshold). Every policy is compared and reported whatever this "
            "says; this only names the one written out as chosen. The Thompson sampling "
            "run is a simulation of online learning rather than an export target: its "
            "decisions are a function of the stream it saw, and a demo needs a policy "
            "that is the same on every run"
        ),
    )
    evaluate.add_argument(
        "--penalties",
        metavar="SPEC",
        default=None,
        help=(
            "what a wrong fill costs, by cost tier, as `payment=4,identity=4,address=2,"
            "other=1` (the defaults). Tiers left out keep their default. The unit is one "
            "correct fill, so these are ratios rather than currency"
        ),
    )
    add_common(evaluate)

    agentic = verbs.add_parser(
        "agentic",
        help="fill the generated HTML pages in a headless browser and score what landed",
        description=(
            "Load each generated form page over file://, read every input's signals "
            "out of the live DOM, classify them, apply the exported fill or skip "
            "policy, type the locale's synthetic profile into the fields it decides "
            "to fill, and score what the DOM holds afterwards against the ground "
            "truth the generator embedded. Writes one ordinary run directory per "
            "page and replicate, so `triage ingest` and `triage compare` rank the "
            "engines on the task metric. Synthetic data, local pages, nothing on "
            "the network."
        ),
    )
    agentic.add_argument(
        "--pages",
        required=True,
        metavar="DIR",
        help="the `pages` directory `triage autofill generate` wrote",
    )
    agentic.add_argument(
        "--out", required=True, metavar="DIR", help="where the runs and the summary go"
    )
    agentic.add_argument(
        "--policy",
        default="model",
        choices=["model", "heuristic", "llm"],
        help="which engine classifies the fields (default: model)",
    )
    agentic.add_argument(
        "--weights", metavar="W.npz", default=None, help="a model written by train or sweep"
    )
    agentic.add_argument(
        "--data",
        metavar="DIR",
        default=None,
        help=(
            "the corpus the pages came from, for --policy llm, which retrieves its few "
            "shot examples from the training split (default: the parent of --pages)"
        ),
    )
    agentic.add_argument(
        "--decisions",
        metavar="POLICY.json",
        default=None,
        help=(
            "the contextual bandit comparison `triage autofill evaluate` exported. Its "
            "chosen policy decides which fields are filled; without it the demo fills "
            "every field it has a value for, which is the naive product 5.3 compares "
            "against"
        ),
    )
    agentic.add_argument(
        "--no-browser",
        action="store_true",
        help=(
            "extract the fields by parsing the HTML instead of driving Chrome. The "
            "decision, fill and scoring logic is the same code either way, which is "
            "what makes this runnable in CI"
        ),
    )
    agentic.add_argument(
        "--screenshot",
        metavar="PNG",
        default=None,
        help="capture ONE page to this file (browser mode only)",
    )
    agentic.add_argument(
        "--screenshot-page",
        metavar="STEM",
        default=None,
        help="which page to capture, by file stem (default: the first)",
    )
    agentic.add_argument(
        "--bootstrap",
        type=bounded_int(low=2),
        default=5,
        help="resampled replicates per page, one run directory each (default: 5)",
    )
    agentic.add_argument("--seed", type=bounded_int(low=0), default=0, help="resampling seed")
    agentic.add_argument(
        "--penalties",
        metavar="SPEC",
        default=None,
        help="what a wrong fill costs, by cost tier, as `payment=4,address=2` (see evaluate)",
    )
    agentic.add_argument(
        "--llm-host",
        default="http://localhost:11434",
        help="where Ollama is listening, for --policy llm (default: http://localhost:11434)",
    )
    agentic.add_argument(
        "--llm-model",
        default="mistral-nemo:12b-instruct-2407-q4_K_M",
        help="chat model for --policy llm; the faster fallback is used when it is absent",
    )
    agentic.add_argument(
        "--llm-k",
        type=bounded_int(low=1, high=32),
        default=8,
        help="few shot examples retrieved per field for --policy llm (default: 8)",
    )
    add_common(agentic)


def run_autofill(args: argparse.Namespace) -> int:
    """Hand off to the reference workload, imported only when it is asked for."""
    from triage.autofill.cli import run

    return run(args)


def run_llm(args: argparse.Namespace) -> int:
    """Hand off to the local model layer, imported only when it is asked for."""
    from triage.llm.cli import run

    return run(args)


def run_ask_alias(args: argparse.Namespace) -> int:
    """`triage ask` is `triage llm ask`, spelled the way 6.5 spells it."""
    args.llm_verb = ASK_ALIAS
    return run_llm(args)


def run_ingest(args: argparse.Namespace) -> int:
    with Store(args.database) as store:
        result = ingest(
            args.path,
            store,
            parsers=parsers_for(outcomes=args.outcomes),
            force=args.force,
            show_progress=not args.quiet,
        )
        stats = store.statistics()

    print(f"ingest: {result.summary()}")
    for run_id in result.outcome_runs:
        print(
            f"  {run_id} holds cross sectional outcome rows, not a training curve: compare it "
            f"with paired_permutation rather than `triage compare`",
            file=sys.stderr,
        )
    for run_id, message in result.failed:
        print(f"  failed: {run_id}: {message}", file=sys.stderr)
        # The traceback is kept rather than printed by default: it is what
        # separates a corrupt log from a bug in this tool, and it is noise
        # until somebody is looking for exactly that.
        logging.getLogger("triage.cli").debug(
            "traceback for %s:\n%s", run_id, result.tracebacks.get(run_id, "(not recorded)")
        )
    for run_id in result.empty:
        print(f"  warning: {run_id} parsed but carries no scalar metrics", file=sys.stderr)
    for run_id, dropped in sorted(result.dropped_by_run.items()):
        print(
            f"  warning: {run_id} lost {dropped} non finite point(s) (NaN or infinite), "
            f"which are dropped rather than compared",
            file=sys.stderr,
        )
    cross_sectional = (
        f", {stats['outcome_runs']} outcome file(s) holding {stats['outcome_rows']:,} rows"
        if stats["outcome_runs"]
        else ""
    )
    print(
        f"database: {stats['runs']} runs, {stats['series']} series, {stats['points']:,} points"
        f"{cross_sectional}, {stats['database_bytes'] / 1024:.0f} KB on disk "
        f"({stats['compression_ratio']:.1f}x compression on the series)"
    )
    return EXIT_RUNS_FAILED if result.failed else EXIT_OK


@dataclass
class Analysis:
    """Everything one analysis pass produced, including what it refused to do.

    The refusals travel beside the findings because both outputs have to name
    them. A comparison this tool declined to make is the thing it most wants to
    be trusted for, and for one release it was the thing most easily missed:
    `compare_all` caught the refusal and moved on.

    The two configurations are carried here rather than rebuilt by each caller.
    `run_compare` and `run_report` each called `configs_from(args)` a second time
    to print what they had just analysed with, which is two chances to print a
    configuration that is not the one that ran.
    """

    report: TriageReport
    experiments: list[Experiment]
    results: list[ComparisonResult]
    refusals: tuple[ComparisonRefusal, ...]
    comparison_config: ComparisonConfig
    regression_config: RegressionConfig
    #: Empty when the caller asked for no sensitivity pass. `compare` does not
    #: print a sensitivity table, and it used to compute the whole one anyway
    #: and throw it away: a Spearman permutation test per parameter per tag,
    #: paid for on every invocation of the verb people run most.
    sensitivity: list[SensitivityResult] = field(default_factory=list)


def analyse_database(args: argparse.Namespace, sensitivity: bool = True) -> Analysis:
    """Compare every condition in one database against the baseline.

    `sensitivity=False` skips the hyperparameter sensitivity pass, which only
    the HTML report renders.
    """
    comparison_config, regression_config = configs_from(args)
    # Read only, and that is a correctness property rather than tidiness. A read
    # write open creates the file it was pointed at, so a typo in --database
    # became an empty database and then "this sweep has no runs"; and it sets
    # the journal mode, which is a write into the database header, so rendering
    # a report changed the bytes of the database the report is evidence about.
    with Store.open_read_only(args.database) as store:
        experiments = store.load_all()
    if not experiments:
        raise ComparisonError(f"{args.database} holds no runs; run `triage ingest` first")

    results = compare_all(experiments, args.baseline, args.tags, comparison_config)
    # `classify` judges in input order so a caller can join its own results to
    # the findings positionally; severity order, which is what a reader wants
    # to see first, is asked for here.
    findings = rank(classify(results, regression_config))
    report = TriageReport(findings=findings, config=regression_config, baseline=args.baseline)
    return Analysis(
        report=report,
        experiments=experiments,
        results=list(results),
        refusals=results.refusals,
        comparison_config=comparison_config,
        regression_config=regression_config,
        sensitivity=(analyse(experiments, args.tags, comparison_config) if sensitivity else []),
    )


def run_compare(args: argparse.Namespace) -> int:
    analysis = analyse_database(args, sensitivity=False)
    report = analysis.report
    refusals = analysis.refusals
    comparison_config = analysis.comparison_config
    regression_config = analysis.regression_config

    # `--quiet` removes the count summary and nothing else, and the boundary is
    # ground rule 3 rather than taste: everything below states how a number in
    # the table was arrived at, so suppressing any of it would print numbers
    # without their provenance, which is the one thing this tool does not do.
    # The summary is the one line that states nothing the table does not.
    if not args.quiet:
        print(f"\n{report.summary()}")
    print(f"gates: {regression_config.describe()}")
    # An adjusted p value is only readable beside the family it was adjusted
    # in, and this tool now corrects one family per comparison mode rather
    # than pooling two different claims about the world into one denominator.
    print(f"families: {report.family_note()}")
    print(f"statistic: {comparison_config.describe()}\n")

    header = f"{'candidate':<28} {'metric':<18} {'change':>9} {'p':>10} {'p adj':>10}  verdict"
    print(header)
    print("-" * len(header))
    for finding in report.findings:
        result = finding.result
        mode = " [weak]" if result.is_weak_mode else ""
        print(
            f"{result.candidate[:27]:<28} {result.tag[:17]:<18} "
            f"{result.relative_effect_pct:>+8.2f}% {result.p_value:>10.4f} "
            f"{finding.adjusted_p:>10.4f}  {finding.verdict}{mode}"
        )

    warnings = {warning for finding in report.findings for warning in finding.result.warnings}
    if warnings or refusals:
        print("\ncaveats:")
        for warning in sorted(warnings):
            print(f"  - {warning}")
        # A comparison that was refused is not a comparison that was fine. It
        # is named here, under the same heading as every other caveat, because
        # the alternative is a table that quietly has fewer rows than the sweep
        # has conditions.
        for refusal in refusals:
            print(f"  - no comparison for {refusal.describe()}")

    print(f"\npermutation seed {comparison_config.seed}; rerunning reproduces these numbers.")
    return no_comparisons_code(report)


def no_comparisons_code(report: TriageReport) -> int:
    """`EXIT_NO_COMPARISONS` when nothing was compared, otherwise success.

    A gate that passes because nothing was tested is worse than one that fails,
    and this tool exited 0 in exactly that case: three conditions too short for
    any calibrated test produced an empty table, a cheerful exit, and a green
    build. The refusals are printed above whatever this returns; the code is so
    that a CI job does not have to read them to notice.
    """
    if report.findings:
        return EXIT_OK
    print(
        "no comparisons were performed: nothing in this database could be compared "
        "against the baseline on any metric. Any refusals are listed above"
    )
    return EXIT_NO_COMPARISONS


def llm_summary_for(args: argparse.Namespace, analysis: Analysis) -> Any:
    """The generated summary block, or `None` when nobody asked for one.

    `None` is the whole point of this function existing separately. Without
    `--llm-summary` nothing here imports the LLM layer, nothing reaches
    localhost, and the rendered page is exactly the page this tool rendered
    before Section 6 existed, which is what the byte identical reproducibility
    gate is checking.

    A model that is not running is reported and skipped rather than raised. The
    report is the product; a summary is an addition to it, and failing to
    produce a whole report because an optional paragraph could not be written
    would be the wrong trade in both directions.
    """
    if not getattr(args, "llm_summary", False):
        return None

    from triage.core.store import Store as WritableStore
    from triage.llm.ollama_client import LlmError, OllamaClient
    from triage.llm.summarizer import summarise_report
    from triage.report.html_report import SummaryBlock

    try:
        client = OllamaClient(host=args.host, chat_model=args.model, num_ctx=args.num_ctx)
        chosen = client.resolve_chat_model(args.model)
        with WritableStore(args.database) as store:
            summary = summarise_report(
                client,
                store,
                analysis.report,
                refusals=[refusal.describe() for refusal in analysis.refusals],
                model=chosen,
            )
    except LlmError as error:
        print(f"note: no automated summary in this report: {error}", file=sys.stderr)
        return None
    if summary.n_dropped:
        print(
            f"note: the grounding pass deleted {summary.n_dropped} generated sentence(s) "
            f"carrying numbers this report does not support",
            file=sys.stderr,
        )
    return SummaryBlock.of(summary)


def run_report(args: argparse.Namespace) -> int:
    from triage.report.html_report import build_context, load_autofill_section, render

    analysis = analyse_database(args)

    # `getattr` because `run_demo` assembles its own namespace, and a report of
    # the classic sweep has no autofill evaluation to attach.
    autofill_dir = getattr(args, "autofill", None)
    autofill = load_autofill_section(autofill_dir) if autofill_dir else None
    if autofill_dir and autofill is None:
        print(
            f"note: {autofill_dir} holds no calibration.json, so the report carries no "
            f"autofill section. `triage autofill evaluate --policy heuristic` writes none: "
            f"a rule engine has no probabilities to calibrate",
            file=sys.stderr,
        )

    summary = llm_summary_for(args, analysis)

    context = build_context(
        experiments=analysis.experiments,
        triage=analysis.report,
        sensitivity=analysis.sensitivity,
        baseline=args.baseline,
        database=str(Path(args.database)),
        comparison_config=analysis.comparison_config,
        regression_config=analysis.regression_config,
        calibration=CALIBRATION_NOTE,
        refusals=analysis.refusals,
        title=args.title,
        autofill=autofill,
        llm_summary=summary,
    )
    output = render(context, args.output)
    size_kb = output.stat().st_size / 1024
    print(f"report: {output} ({size_kb:.0f} KB, self contained)")
    if not args.quiet:
        print(f"  {analysis.report.summary()}")
    return no_comparisons_code(analysis.report)


def run_demo(args: argparse.Namespace) -> int:
    """Synthesise the sweep, ingest it, and report on it, in one command.

    This exists so that `pip install ml-experiment-triage && triage demo` shows
    somebody what the tool does. Before it, the demo was a `make` target over a
    script in `examples/`, which is to say it was available only to somebody who
    had already cloned the repository, which is to say only to somebody who had
    already decided to trust it.

    The sweep and the database go in a temporary directory that is removed
    afterwards, because they are 31 runs of scaffolding and the report is the
    output. `--keep` puts them somewhere durable instead, which is what to reach
    for when the interesting thing is the database rather than the page.

    The analysis is not reimplemented here: the same `run_report` the `report`
    verb uses is called with the same arguments it would have been given, so the
    demo cannot quietly diverge from the tool it is demonstrating.
    """
    from triage.demo import BASELINE, CONDITIONS, KNOWN_BEST, generate

    with contextlib.ExitStack() as stack:
        if args.keep:
            workspace = Path(args.keep)
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            workspace = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="triage_demo_"))
            )
        sweep = workspace / "demo_sweep"
        database = workspace / "triage.db"

        if not args.quiet:
            print(
                f"demo: synthesising {sum(c.n_seeds for c in CONDITIONS)} runs across "
                f"{len(CONDITIONS)} conditions in three log formats under {sweep}"
            )
        generate(sweep, show_progress=not args.quiet)

        ingest_args = argparse.Namespace(
            path=str(sweep),
            database=str(database),
            force=False,
            outcomes=False,
            quiet=args.quiet,
        )
        code = run_ingest(ingest_args)
        if code != EXIT_OK:
            return code

        report_args = argparse.Namespace(
            database=str(database),
            baseline=BASELINE,
            tags=None,
            window_fraction=ComparisonConfig.window_fraction,
            window_minimum=ComparisonConfig.window_minimum,
            permutations=ComparisonConfig.n_permutations,
            alpha=RegressionConfig.alpha,
            practical_threshold=None,
            practical_threshold_absolute=None,
            fdr=RegressionConfig.false_discovery_rate,
            seed=ComparisonConfig.seed,
            higher_is_better=None,
            lower_is_better=None,
            output=args.output,
            title="ML Experiment Triage demo",
            quiet=args.quiet,
        )
        code = run_report(report_args)

    if not args.quiet:
        # The ground truth, printed after the verdicts rather than before them,
        # so a reader sees what the tool concluded and then what was true.
        print(
            f"\nthe sweep was built with {KNOWN_BEST} as the real winner and "
            f"{BASELINE} as the baseline; the report above was produced without "
            f"either of those facts"
        )
    return code


#: Verb to handler. A module level dict rather than a local one so a test can
#: substitute a handler and drive the error boundary directly.
HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "ingest": run_ingest,
    "compare": run_compare,
    "report": run_report,
    "demo": run_demo,
    "autofill": run_autofill,
    "llm": run_llm,
    ASK_ALIAS: run_ask_alias,
}


def message_of(error: BaseException) -> str:
    """The text of an exception, without a `KeyError`'s repr quoting.

    `str(KeyError("no run 'x'"))` is `repr` of the argument, so a message
    written for a reader came out wrapped in quotes: `error: "no run 'x'"`.
    Every other exception's `str` is the message itself.
    """
    if isinstance(error, KeyError) and error.args:
        return str(error.args[0])
    return str(error)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        return HANDLERS[args.verb](args)
    # Everything this tool can foresee about its input, and nothing else. An
    # exception outside this list is a bug here rather than a fact about the
    # data, so it keeps its traceback and the interpreter's own exit 1, which
    # is what `EXIT_CRASH` means and why it no longer shares a code with "some
    # runs failed to parse".
    except HANDLED_ERRORS as error:
        print(f"error: {message_of(error)}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
