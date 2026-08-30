"""The whole tool end to end: synthesise, ingest, compare, report.

    python examples/demo_workflow.py
    python examples/demo_workflow.py --verify-committed

Plain form regenerates the synthetic sweep, rebuilds the demo database that the
LaTeX reports and CI compile from, and writes the HTML report.

The `--verify-committed` form is the reproducibility check that gives Section 7
of the design its teeth. It rebuilds everything from the fixed seeds and asserts
that every verdict matches the committed record in `examples/expected_verdicts.json`
to four decimal places. If a refactor moves a p value by so much as a rounding
error, this fails and names the number that moved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from examples.make_synthetic_runs import BASELINE, KNOWN_BEST, generate
from triage.analysis.comparison import ComparisonConfig, compare_all
from triage.analysis.regression import RegressionConfig, TriageReport, classify, rank
from triage.analysis.sensitivity import analyse
from triage.cli import CALIBRATION_NOTE
from triage.core.store import Store
from triage.ingest import ingest
from triage.report.html_report import build_context, render

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "experiments" / "results" / "demo_sweep"
DEMO_DATABASE = ROOT / "experiments" / "demo" / "triage.db"
HTML_OUTPUT = ROOT / "experiments" / "results" / "triage_report.html"
EXPECTED = ROOT / "examples" / "expected_verdicts.json"


def build(show_progress: bool = True) -> tuple[TriageReport, list, list, float]:
    """Run the pipeline and return the report, sensitivity, runs and elapsed time."""
    started = time.perf_counter()

    print("1. synthesising the sweep")
    generate(SWEEP_DIR, show_progress=show_progress)

    print("2. ingesting into the database")
    DEMO_DATABASE.parent.mkdir(parents=True, exist_ok=True)
    if DEMO_DATABASE.exists():
        # The sweep was just rewritten, so every fingerprint changed. Starting
        # from an empty database keeps this deterministic rather than depending
        # on what an earlier run happened to leave behind.
        DEMO_DATABASE.unlink()
    with Store(DEMO_DATABASE) as store:
        result = ingest(SWEEP_DIR, store, show_progress=show_progress)
        experiments = store.load_all()
        stats = store.statistics()
    print(f"   {result.summary()}")
    print(
        f"   {stats['runs']} runs, {stats['points']:,} points, "
        f"{stats['database_bytes'] / 1024:.0f} KB on disk, "
        f"{stats['compression_ratio']:.1f}x compression"
    )
    if result.failed:
        for run_id, message in result.failed:
            print(f"   failed: {run_id}: {message}", file=sys.stderr)
        raise SystemExit("ingest reported failures; stopping")

    print("3. comparing every condition against the baseline")
    comparison_config = ComparisonConfig()
    regression_config = RegressionConfig()
    results = compare_all(experiments, BASELINE, config=comparison_config)
    findings = rank(classify(results, regression_config))
    report = TriageReport(findings=findings, config=regression_config, baseline=BASELINE)
    sensitivity = analyse(experiments, config=comparison_config)
    print(f"   {report.summary()}")

    return report, sensitivity, experiments, time.perf_counter() - started


def write_html(report: TriageReport, sensitivity: list, experiments: list) -> Path:
    context = build_context(
        experiments=experiments,
        triage=report,
        sensitivity=sensitivity,
        baseline=BASELINE,
        database=str(DEMO_DATABASE.relative_to(ROOT)),
        comparison_config=ComparisonConfig(),
        regression_config=RegressionConfig(),
        calibration=CALIBRATION_NOTE,
        title="ML Experiment Triage: synthetic demo sweep",
    )
    return render(context, HTML_OUTPUT)


def verdict_record(report: TriageReport) -> list[dict]:
    """The committed reproducibility record: one row per finding, rounded."""
    return [
        {
            "candidate": finding.result.candidate,
            "tag": finding.result.tag,
            "mode": finding.result.mode,
            "verdict": finding.verdict,
            "relative_effect_pct": round(finding.result.relative_effect_pct, 4),
            "p_value": round(finding.result.p_value, 4),
            "adjusted_p": round(finding.adjusted_p, 4),
        }
        for finding in report.findings
    ]


def compare_records(actual: list[dict], expected: list[dict]) -> list[str]:
    differences: list[str] = []
    if len(actual) != len(expected):
        differences.append(f"{len(actual)} findings now, {len(expected)} in the record")
    for index, (now, before) in enumerate(zip(actual, expected, strict=False)):
        for key in before:
            if now.get(key) != before[key]:
                differences.append(
                    f"row {index} ({before['candidate']}, {before['tag']}): "
                    f"{key} was {before[key]}, now {now.get(key)}"
                )
    return differences


def print_table(report: TriageReport) -> None:
    header = f"{'candidate':<18} {'metric':<15} {'change':>9} {'p adj':>9}  verdict"
    print(f"\n{header}")
    print("-" * len(header))
    for finding in report.findings:
        result = finding.result
        mode = "  [weaker mode]" if result.is_weak_mode else ""
        print(
            f"{result.candidate:<18} {result.tag:<15} "
            f"{result.relative_effect_pct:>+8.2f}% {finding.adjusted_p:>9.4f}  "
            f"{finding.verdict}{mode}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-committed",
        action="store_true",
        help="rebuild from the fixed seeds and check every verdict against the committed record",
    )
    parser.add_argument(
        "--update-record",
        action="store_true",
        help="rewrite the committed verdict record; use only when a change is intended",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress bars")
    args = parser.parse_args()

    report, sensitivity, experiments, elapsed = build(show_progress=not args.quiet)
    record = verdict_record(report)

    if args.verify_committed:
        if not EXPECTED.exists():
            print(f"error: no committed record at {EXPECTED}", file=sys.stderr)
            return 2
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
        differences = compare_records(record, expected["findings"])
        if differences:
            print("\nREPRODUCIBILITY FAILURE: the demo no longer matches the record")
            for difference in differences:
                print(f"  {difference}")
            return 1
        print(f"\nreproducibility verified: all {len(record)} verdicts match the committed record")
        return 0

    if args.update_record:
        EXPECTED.write_text(
            json.dumps(
                {
                    "note": (
                        "Generated by examples/demo_workflow.py --update-record. "
                        "Every number here is reproduced by --verify-committed."
                    ),
                    "baseline": BASELINE,
                    "known_best": KNOWN_BEST,
                    "permutation_seed": ComparisonConfig().seed,
                    "findings": record,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote the verdict record to {EXPECTED.relative_to(ROOT)}")

    print("4. writing the HTML report")
    output = write_html(report, sensitivity, experiments)
    print(f"   {output.relative_to(ROOT)} ({output.stat().st_size / 1024:.0f} KB, self contained)")

    print_table(report)
    print(f"\nbest condition found: {report.best_candidate} (ground truth: {KNOWN_BEST})")
    print(f"demo workflow completed in {elapsed:.1f} s")
    measured = CALIBRATION_NOTE["Measured type I error, strong mode"]
    print(f"calibration behind these p values: {measured}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
