"""The CLI, driven end to end over the synthetic sweep.

These tests run the real verbs against a real database built from real files on
disk, and check the one thing that matters most about the whole pipeline: given
a sweep whose ground truth is known, does the tool find what is actually there.

The sweep is generated once per session into a temporary directory. Regenerating
it per test would triple the runtime for no extra coverage, since none of these
tests mutate it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from examples.make_synthetic_runs import BASELINE, CONDITIONS, KNOWN_BEST, generate
from triage.analysis.comparison import MODE_SEED_REPLICATE, MODE_WINDOW_BLOCK, ComparisonConfig
from triage.analysis.regression import RegressionConfig, TriageReport, classify
from triage.analysis.sensitivity import analyse
from triage.cli import main
from triage.core.store import Store

WORST_CONDITION = "lr0.0100_bs32"
NEGLIGIBLE_CONDITION = "lr0.0010_bs64"
SINGLE_SEED_CONDITION = "lr0.0030_bs128"


@pytest.fixture(scope="session")
def sweep(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("sweep") / "demo_sweep"
    generate(root, show_progress=False)
    return root


@pytest.fixture(scope="session")
def database(sweep: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("db") / "triage.db"
    assert main(["ingest", str(sweep), "--database", str(path), "--quiet"]) == 0
    return path


@pytest.fixture(scope="session")
def report(database: Path) -> TriageReport:
    with Store(database) as store:
        experiments = store.load_all()
    from triage.analysis.comparison import compare_all

    results = compare_all(experiments, BASELINE, config=ComparisonConfig())
    findings = classify(results, RegressionConfig())
    return TriageReport(findings=findings, config=RegressionConfig(), baseline=BASELINE)


# ------------------------------------------------------------------- ingest


def test_ingest_reads_every_run_in_every_format(database: Path) -> None:
    expected_runs = sum(condition.n_seeds for condition in CONDITIONS)
    with Store(database) as store:
        assert len(store) == expected_runs
        formats = {run.source_format for run in store.load_all()}
        assert formats == {"tensorboard", "csv", "jsonl"}
        assert store.tags() == ["val/accuracy", "val/loss"]


def test_reingesting_an_unchanged_sweep_does_no_work(sweep: Path, database: Path) -> None:
    with Store(database) as store:
        before = {run_id: store.source_hash(run_id) for run_id in store.run_ids()}
    assert main(["ingest", str(sweep), "--database", str(database), "--quiet"]) == 0
    with Store(database) as store:
        after = {run_id: store.source_hash(run_id) for run_id in store.run_ids()}
    assert before == after


# ------------------------------------------------------- the headline finding


def test_the_known_best_condition_is_the_top_ranked_improvement(report: TriageReport) -> None:
    """The whole point: given a sweep with a real winner, name the winner.

    The verdict table leads with regressions, because a regression is the thing
    to act on first. So the assertion is that the known best condition heads the
    improvements and is what `best_candidate` reports, which is the ranking the
    reader is pointed at.
    """
    improvements = report.improvements
    assert improvements, "the sweep contains a real improvement and it must be found"
    assert improvements[0].candidate == KNOWN_BEST
    assert report.best_candidate == KNOWN_BEST


def test_the_worst_condition_leads_the_table(report: TriageReport) -> None:
    assert report.findings[0].candidate == WORST_CONDITION
    assert report.findings[0].is_regression
    assert report.regressions[0].candidate == WORST_CONDITION


def test_regressions_are_ranked_by_falling_severity(report: TriageReport) -> None:
    severities = [finding.severity for finding in report.regressions]
    assert severities == sorted(severities, reverse=True)


def test_a_negligible_difference_is_not_called_a_finding(report: TriageReport) -> None:
    """The condition whose true effect is far below the practical gate stays quiet."""
    verdicts = {
        finding.tag: finding.verdict
        for finding in report.findings
        if finding.candidate == NEGLIGIBLE_CONDITION
    }
    assert verdicts, f"{NEGLIGIBLE_CONDITION} should still be compared"
    for verdict in verdicts.values():
        assert "regression" not in verdict
        assert "improvement" not in verdict


def test_no_condition_is_flagged_against_itself(report: TriageReport) -> None:
    assert all(finding.candidate != BASELINE for finding in report.findings)


# ------------------------------------------------------------ mode labelling


def test_the_single_seed_condition_uses_the_weaker_mode_and_says_so(
    report: TriageReport,
) -> None:
    weak = [finding for finding in report.findings if finding.result.is_weak_mode]
    assert weak, "the single seed condition must appear in the weaker mode"
    for finding in weak:
        assert SINGLE_SEED_CONDITION in finding.candidate
        assert finding.result.mode == MODE_WINDOW_BLOCK
        assert "WEAKER CLAIM" in finding.result.mode_label
        assert any("seed variance" in warning for warning in finding.result.warnings)


def test_every_replicated_condition_uses_the_strong_mode(report: TriageReport) -> None:
    strong = [finding for finding in report.findings if not finding.result.is_weak_mode]
    assert len(strong) >= 8
    for finding in strong:
        assert finding.result.mode == MODE_SEED_REPLICATE
        assert finding.result.n_baseline == 5


def test_every_p_value_is_labelled_with_its_test_and_mode(report: TriageReport) -> None:
    """Ground rule 3, checked over the whole real output rather than one result."""
    for finding in report.findings:
        label = finding.result.p_value_label()
        assert "permutation test" in label
        assert ("strong claim" in label) or ("WEAKER CLAIM" in label)


# ---------------------------------------------------------------- the verbs


def test_compare_prints_a_ranked_table(database: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "--database", str(database), "--baseline", BASELINE, "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "regression" in printed
    assert WORST_CONDITION in printed
    assert "permutation seed" in printed
    assert "[weak]" in printed, "the weaker mode must be marked in the terminal output too"


def test_report_writes_a_self_contained_html_file(database: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.html"
    assert (
        main(
            [
                "report",
                "--database",
                str(database),
                "--baseline",
                BASELINE,
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 0
    )
    html = output.read_text(encoding="utf-8")

    assert output.stat().st_size > 500_000, "the Plotly runtime should be inlined"

    # Self contained means nothing is fetched when the page loads. Searching the
    # whole file for a URL would be the wrong test and fails on a correct
    # report: the inlined Plotly bundle carries map tile attributions and a
    # default topojson host as string literals, and this report draws neither
    # maps nor choropleths. What matters is loading tags, so the check strips
    # the inlined scripts first and then looks at the markup that remains.
    assert "<script src=" not in html, "no external script tags"
    assert "<link" not in html, "no external stylesheets"
    markup = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    external = re.findall(r'(?:src|href)\s*=\s*["\']https?://', markup)
    assert not external, f"the report markup references {len(external)} external resources"
    assert KNOWN_BEST in html
    assert "WEAKER CLAIM" in html or "weaker" in html
    assert "Permutation seed" in html
    assert str(ComparisonConfig().seed) in html
    assert "Spearman" in html


def test_report_is_byte_identical_when_rerun(database: Path, tmp_path: Path) -> None:
    """Reproducibility is a deliverable, so two runs must agree on every number."""
    outputs = []
    for name in ("first.html", "second.html"):
        path = tmp_path / name
        main(
            [
                "report",
                "--database",
                str(database),
                "--baseline",
                BASELINE,
                "--output",
                str(path),
                "--quiet",
            ]
        )
        text = path.read_text(encoding="utf-8")
        # The generation timestamp is the one thing that legitimately differs.
        outputs.append([line for line in text.splitlines() if "Generated" not in line])
    assert outputs[0] == outputs[1]


def test_an_unknown_baseline_exits_with_an_error(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", "--database", str(database), "--baseline", "nope", "--quiet"]) == 2
    assert "matches no run or variant" in capsys.readouterr().err


# ------------------------------------------------------------- sensitivity


def test_sensitivity_finds_the_swept_parameters_with_their_counts(database: Path) -> None:
    with Store(database) as store:
        experiments = store.load_all()
    results = analyse(experiments, config=ComparisonConfig())

    parameters = {result.parameter for result in results}
    assert "learning_rate" in parameters
    assert "batch_size" in parameters
    assert "weight_decay" not in parameters, "a constant is not a swept parameter"

    for result in results:
        assert result.n_runs == len(experiments)
        assert result.n_distinct_values >= 3
        assert "n = " in result.p_value_label()


def test_rank_correlation_is_blind_to_the_non_monotone_learning_rate(database: Path) -> None:
    """The documented limitation of Spearman, checked against a sweep built to have it.

    The sweep's learning rate has an optimum in the middle of its range: 0.0003
    is worse than 0.001, 0.003 is the best, 0.01 is far the worst. That is a U
    shape, and the rank correlation of a U shape is near zero however large the
    effect. Batch size was swept monotonically over the same runs, so it shows
    the stronger correlation despite driving a smaller effect.

    This is the tool being right about what it can and cannot see, and the
    assertion exists so that nobody later "fixes" the weak learning rate number.
    """
    with Store(database) as store:
        experiments = store.load_all()
    loss_results = {
        result.parameter: result
        for result in analyse(experiments, config=ComparisonConfig())
        if result.tag == "val/loss"
    }

    learning_rate = loss_results["learning_rate"]
    batch_size = loss_results["batch_size"]
    assert abs(learning_rate.correlation) < abs(batch_size.correlation)
    assert abs(batch_size.correlation) > 0.25, "the monotone parameter should show up"

    # And the effect the correlation cannot see is genuinely there and correctly
    # sized. The best and worst conditions differ only in learning rate, and
    # their designed loss effects are -0.060 and +0.130, so the tool should
    # recover a gap of 0.190 between them.
    designed_gap = 0.130 - (-0.060)
    by_variant: dict[str, list[float]] = {}
    for run in experiments:
        by_variant.setdefault(run.config["variant"], []).append(
            float(run.series("val/loss").final_window().mean())
        )
    measured_gap = np.mean(by_variant["lr0.0100_bs32"]) - np.mean(by_variant["lr0.0030_bs32"])
    assert measured_gap == pytest.approx(designed_gap, abs=0.03)


# ----------------------------------------------------- reproducibility record


def test_the_committed_verdict_record_still_matches(report: TriageReport) -> None:
    """The record in examples/ is the contract that the demo has not drifted."""
    from examples.demo_workflow import EXPECTED, compare_records, verdict_record

    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    differences = compare_records(verdict_record(report), expected["findings"])
    assert not differences, "the demo verdicts have drifted from the committed record"
    assert expected["baseline"] == BASELINE
    assert expected["known_best"] == KNOWN_BEST
