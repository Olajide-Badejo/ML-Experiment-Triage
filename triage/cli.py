"""Three verbs: `ingest`, `compare`, `report`.

    triage ingest experiments/results/demo_sweep --database triage.db
    triage compare --database triage.db --baseline lr0.001_bs32
    triage report  --database triage.db --baseline lr0.001_bs32 --output report.html

`ingest` is the only verb that touches log files. `compare` and `report` read
the database, so they are fast enough to rerun freely, and both are pure
functions of the database plus the recorded permutation seed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from triage.analysis.comparison import (
    ComparisonConfig,
    ComparisonError,
    ComparisonRefusal,
    compare_all,
)
from triage.analysis.regression import RegressionConfig, TriageReport, classify
from triage.analysis.sensitivity import analyse
from triage.calibration import SUMMARY
from triage.core.store import Store
from triage.ingest import ingest

DEFAULT_DATABASE = "triage.db"

#: Accepted values of `--log-level`, lowest detail last so `--help` reads in
#: the order a reader would raise it.
LOG_LEVELS = ("debug", "info", "warning", "error")


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


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"path to the triage database (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress bars, for CI logs")
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
        type=float,
        default=ComparisonConfig.window_fraction,
        help="final window as a fraction of the run",
    )
    parser.add_argument(
        "--window-minimum",
        type=int,
        default=ComparisonConfig.window_minimum,
        help="smallest final window in points",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=ComparisonConfig.n_permutations,
        help="resamples when exhaustive enumeration is too large",
    )
    parser.add_argument(
        "--alpha", type=float, default=RegressionConfig.alpha, help="statistical gate"
    )
    parser.add_argument(
        "--practical-threshold",
        type=float,
        default=RegressionConfig.practical_threshold_pct,
        help="practical gate, as a relative percent",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=RegressionConfig.false_discovery_rate,
        help="Benjamini Hochberg false discovery rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ComparisonConfig.seed,
        help="permutation seed, printed in every report",
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


def configs_from(args: argparse.Namespace) -> tuple[ComparisonConfig, RegressionConfig]:
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
            practical_threshold_pct=args.practical_threshold,
            false_discovery_rate=args.fdr,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="verb", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="parse a directory of runs into the database"
    )
    ingest_parser.add_argument("path", help="directory holding one subdirectory per run")
    ingest_parser.add_argument(
        "--force", action="store_true", help="reparse even when the source is unchanged"
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

    return parser


def run_ingest(args: argparse.Namespace) -> int:
    with Store(args.database) as store:
        result = ingest(args.path, store, force=args.force, show_progress=not args.quiet)
        stats = store.statistics()

    print(f"ingest: {result.summary()}")
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
    print(
        f"database: {stats['runs']} runs, {stats['series']} series, {stats['points']:,} points, "
        f"{stats['database_bytes'] / 1024:.0f} KB on disk "
        f"({stats['compression_ratio']:.1f}x compression on the series)"
    )
    return 1 if result.failed else 0


def analyse_database(
    args: argparse.Namespace,
) -> tuple[TriageReport, list, list, tuple[ComparisonRefusal, ...]]:
    """The whole analysis for one database, including what it refused to do.

    The refusals travel out of here beside the findings because both outputs
    have to name them. A comparison this tool declined to make is the thing it
    most wants to be trusted for, and for one release it was the thing most
    easily missed: `compare_all` caught the refusal and moved on.
    """
    comparison_config, regression_config = configs_from(args)
    with Store(args.database) as store:
        experiments = store.load_all()
    if not experiments:
        raise ComparisonError(f"{args.database} holds no runs; run `triage ingest` first")

    results = compare_all(experiments, args.baseline, args.tags, comparison_config)
    findings = classify(results, regression_config)
    sensitivity = analyse(experiments, args.tags, comparison_config)
    report = TriageReport(findings=findings, config=regression_config, baseline=args.baseline)
    return report, sensitivity, experiments, results.refusals


def run_compare(args: argparse.Namespace) -> int:
    report, _sensitivity, _experiments, refusals = analyse_database(args)
    comparison_config, regression_config = configs_from(args)

    print(f"\n{report.summary()}")
    print(f"gates: {regression_config.describe()}")
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
    return 0


def run_report(args: argparse.Namespace) -> int:
    from triage.report.html_report import build_context, render

    report, sensitivity, experiments, refusals = analyse_database(args)
    comparison_config, regression_config = configs_from(args)

    context = build_context(
        experiments=experiments,
        triage=report,
        sensitivity=sensitivity,
        baseline=args.baseline,
        database=str(Path(args.database)),
        comparison_config=comparison_config,
        regression_config=regression_config,
        calibration=CALIBRATION_NOTE,
        refusals=refusals,
        title=args.title,
    )
    output = render(context, args.output)
    size_kb = output.stat().st_size / 1024
    print(f"report: {output} ({size_kb:.0f} KB, self contained)")
    print(f"  {report.summary()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    handlers = {"ingest": run_ingest, "compare": run_compare, "report": run_report}
    try:
        return handlers[args.verb](args)
    except (ComparisonError, FileNotFoundError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
