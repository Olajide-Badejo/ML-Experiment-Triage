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
from triage.analysis.regression import RegressionConfig, TriageReport, classify, rank
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
    # `classify` hands back findings in input order (E4a); severity order is what
    # `rank` is for, and calling it here is exactly what the CLI does.
    findings = rank(classify(results, RegressionConfig()))
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

    # E4a moved severity order out of `classify` and into `rank`. The terminal
    # table is the reader's first look, so it has to keep calling it.
    rows = [line for line in printed.splitlines() if line.split()[1:2] and "val/" in line]
    assert rows[0].startswith(WORST_CONDITION), "the worst condition must still lead the table"


def _verdict_for(printed: str, candidate: str, tag: str) -> str:
    """The verdict cell of the one table row for this candidate and metric."""
    rows = [
        line for line in printed.splitlines() if line.startswith(candidate) and f" {tag} " in line
    ]
    assert len(rows) == 1, f"expected one row for {candidate} on {tag}, got {rows}"
    return rows[0]


def test_a_direction_flag_flips_the_verdict_on_that_metric(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D17: the documented direction override is reachable from the command line.

    The worst condition raises val/loss, which is a regression by default. Told
    that a larger val/loss is better, the same numbers must read as the
    improvement they then are, and only on the tag that was named.
    """
    argv = ["compare", "--database", str(database), "--baseline", BASELINE, "--quiet"]
    assert main(argv) == 0
    default_row = _verdict_for(capsys.readouterr().out, WORST_CONDITION, "val/loss")
    assert "regression" in default_row

    assert main([*argv, "--higher-is-better", "val/loss"]) == 0
    flipped = capsys.readouterr().out
    assert "improvement" in _verdict_for(flipped, WORST_CONDITION, "val/loss")
    # val/accuracy was not named, so its own inference still stands.
    assert "improvement" not in _verdict_for(flipped, WORST_CONDITION, "val/accuracy")


def _verdicts_from(printed: str) -> dict[tuple[str, str], str]:
    """Every table row as `(candidate, metric) -> verdict`, for diffing runs."""
    verdicts = {}
    for line in printed.splitlines():
        columns = line.split()
        if len(columns) >= 6 and columns[1].startswith("val/"):
            verdicts[(columns[0], columns[1])] = " ".join(columns[5:])
    return verdicts


def test_the_fdr_flag_changes_at_least_one_verdict(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D2: `--fdr` was a no-op, and three decades of values printed one table.

    The flag now sets the gate it names, so a rate loose enough to admit a large
    adjusted p has to move something. Nothing here asserts which row moves: the
    defect was that no row could.
    """
    argv = ["compare", "--database", str(database), "--baseline", BASELINE, "--quiet"]
    assert main([*argv, "--fdr", "0.05"]) == 0
    default = _verdicts_from(capsys.readouterr().out)
    assert default, "the demo sweep must produce a verdict table to compare against"

    assert main([*argv, "--fdr", "0.90"]) == 0
    loosened = _verdicts_from(capsys.readouterr().out)
    assert set(default) == set(loosened), "the same comparisons, judged differently"
    assert default != loosened

    assert main([*argv, "--fdr", "0.001"]) == 0
    tightened = _verdicts_from(capsys.readouterr().out)
    assert tightened != default


def test_the_absolute_practical_gate_is_reachable_and_states_itself(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """E2/D18: the gate in the metric's own units, from the command line.

    The demo's val/loss moves by well under one unit, so an absolute gate of one
    unit silences every practical verdict while the percentage gate does not.
    """
    argv = ["compare", "--database", str(database), "--baseline", BASELINE, "--quiet"]
    assert main([*argv, "--practical-threshold-absolute", "1.0"]) == 0
    printed = capsys.readouterr().out
    assert "an absolute effect of at least 1 in the units of the metric" in printed
    verdicts = set(_verdicts_from(printed).values())
    assert "regression" not in verdicts
    assert "improvement" not in verdicts


def test_naming_both_practical_gates_is_a_usage_error(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "compare",
            "--database",
            str(database),
            "--baseline",
            BASELINE,
            "--quiet",
            "--practical-threshold",
            "2.0",
            "--practical-threshold-absolute",
            "1.0",
        ]
    )
    assert code == 2
    assert "a finding clears one gate, so name one" in capsys.readouterr().err


def test_a_tag_named_in_both_direction_flags_is_a_usage_error(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "compare",
            "--database",
            str(database),
            "--baseline",
            BASELINE,
            "--quiet",
            "--higher-is-better",
            "val/loss",
            "--lower-is-better",
            "val/loss",
        ]
    )
    assert code == 2
    assert "both higher and lower is better" in capsys.readouterr().err


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


def test_report_is_byte_identical_when_rerun(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproducibility is a deliverable, so two runs must agree on every byte.

    This gate used to mask the timestamp and compare the rest, which is a way of
    not testing the thing: the mask dropped lines holding a capital G
    "Generated" while the template writes it in lower case in both places, so
    nothing was ever dropped and the test passed only while the two builds
    landed in the same minute. With `SOURCE_DATE_EPOCH` pinning the one input
    that is not the database (D27), there is nothing left to mask and the files
    are compared as bytes.
    """
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
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
        outputs.append(path)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert "2023-11-14 22:13 UTC" in outputs[0].read_text(encoding="utf-8")


def test_an_unknown_baseline_exits_with_an_error(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", "--database", str(database), "--baseline", "nope", "--quiet"]) == 2
    assert "matches no run or variant" in capsys.readouterr().err


def test_compare_names_a_mistyped_database_instead_of_creating_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D14: the read paths open read only, so a typo is a typo, not a new file.

    A read write open created the missing file, found nothing in it, and
    reported an empty sweep, which reads as "your ingest did nothing" rather
    than "that path is wrong".
    """
    missing = tmp_path / "nope" / "typo.db"
    assert main(["compare", "--database", str(missing), "--baseline", BASELINE, "--quiet"]) == 2
    assert "no such database" in capsys.readouterr().err
    assert not missing.parent.exists()


def test_reporting_on_a_database_does_not_change_it(database: Path, tmp_path: Path) -> None:
    """Evidence must not modify its own source: `report` is a pure read."""
    import hashlib

    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert (
        main(
            [
                "report",
                "--database",
                str(database),
                "--baseline",
                BASELINE,
                "--output",
                str(tmp_path / "untouched.html"),
                "--quiet",
            ]
        )
        == 0
    )
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


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
        assert result.n_variants == len(CONDITIONS)
        assert result.n_distinct_values >= 3
        # The label carries both counts: the runs behind the correlation and the
        # conditions, which is the unit the correlation is actually over.
        assert f"n variants = {len(CONDITIONS)}" in result.p_value_label()
        assert f"n runs = {len(experiments)}" in result.p_value_label()


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
