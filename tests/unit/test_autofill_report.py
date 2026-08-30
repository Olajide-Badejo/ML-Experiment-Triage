"""The autofill section of the HTML report, and its reliability diagram.

The section is optional and absent by default, and the first test here is that
its arrival changed nothing about a report that does not carry one: the report
is required to be a byte identical function of the database, and a new block in
the template is exactly the kind of change that quietly is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage.analysis.comparison import ComparisonConfig, compare_all
from triage.analysis.regression import RegressionConfig, TriageReport, classify, rank
from triage.autofill.evaluate import write_evaluation
from triage.autofill.features import featurise
from triage.autofill.generator import GeneratorConfig, generate_fields
from triage.autofill.model import TrainConfig, train
from triage.core.experiment import Experiment, MetricSeries
from triage.report.html_report import (
    AutofillSection,
    build_context,
    load_autofill_section,
    render,
)


def _runs() -> list[Experiment]:
    runs = []
    for variant, offset in (("base", 0.0), ("candidate", -0.05)):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            steps = np.arange(60)
            runs.append(
                Experiment(
                    run_id=f"{variant}_{seed}",
                    source_path=f"memory://{variant}/{seed}",
                    source_format="jsonl",
                    config={"variant": variant, "seed": seed},
                    metrics={
                        "val/loss": MetricSeries(
                            tag="val/loss",
                            steps=steps,
                            values=1.0 - 0.005 * steps + offset + rng.normal(0, 0.01, 60),
                        )
                    },
                )
            )
    return runs


def _context(autofill: AutofillSection | None) -> object:
    runs = _runs()
    config = ComparisonConfig()
    results = compare_all(runs, "base", None, config)
    triage = TriageReport(
        findings=rank(classify(results, RegressionConfig())),
        config=RegressionConfig(),
        baseline="base",
    )
    return build_context(
        experiments=runs,
        triage=triage,
        sensitivity=[],
        baseline="base",
        database="memory.db",
        comparison_config=config,
        regression_config=RegressionConfig(),
        generated_at="2026-01-01 00:00 UTC",
        autofill=autofill,
    )


@pytest.fixture(scope="module")
def section(tmp_path_factory: pytest.TempPathFactory) -> AutofillSection:
    root = tmp_path_factory.mktemp("eval")
    records = generate_fields(GeneratorConfig(n_fields=2000), seed=0)
    model = train(
        featurise([r for r in records if r.split == "train"]),
        featurise([r for r in records if r.split == "val"]),
        TrainConfig(epochs=4, seed=0),
    ).model
    write_evaluation(
        root,
        records=[r for r in records if r.split == "val"],
        model=model,
        split="val",
        locale="all",
        bootstrap=2,
        seed=0,
    )
    loaded = load_autofill_section(root)
    assert loaded is not None
    return loaded


def test_a_report_without_the_section_is_unchanged(tmp_path: Path) -> None:
    """The report is a pure function of the database, and still is."""
    first = render(_context(None), tmp_path / "a.html")
    second = render(_context(None), tmp_path / "b.html")
    assert first.read_bytes() == second.read_bytes()
    assert "Autofill calibration" not in first.read_text(encoding="utf-8")


def test_the_section_loads_from_what_the_evaluation_wrote(section: AutofillSection) -> None:
    assert section.ece_post < section.ece_pre
    assert section.temperature > 0
    assert section.n_bins == 15
    assert len(section.bins_pre) == len(section.bins_post) == 15
    assert {row["engine"] for row in section.engines} == {"ngram", "rules"}


def test_loading_a_directory_with_no_evaluation_returns_nothing(tmp_path: Path) -> None:
    """A missing calibration file is an absence, not an error."""
    assert load_autofill_section(tmp_path) is None


def test_the_rendered_report_carries_both_calibration_numbers(
    section: AutofillSection, tmp_path: Path
) -> None:
    """Acceptance criterion 3: the report shows the ECE before and after."""
    output = render(_context(section), tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")
    assert "Autofill calibration" in html
    assert f"{section.ece_pre:.4f}" in html
    assert f"{section.ece_post:.4f}" in html
    assert f"{section.temperature:.3f}" in html


def test_the_reliability_diagram_is_drawn(section: AutofillSection, tmp_path: Path) -> None:
    html = render(_context(section), tmp_path / "report.html").read_text(encoding="utf-8")
    assert 'id="figure-reliability"' in html
    assert "perfect calibration" in html


def test_the_per_engine_table_is_in_the_report(section: AutofillSection, tmp_path: Path) -> None:
    html = render(_context(section), tmp_path / "report.html").read_text(encoding="utf-8")
    for row in section.engines:
        assert f"{row['macro_f1']:.4f}" in html


def test_the_section_renders_the_same_bytes_twice(section: AutofillSection, tmp_path: Path) -> None:
    first = render(_context(section), tmp_path / "one.html")
    second = render(_context(section), tmp_path / "two.html")
    assert first.read_bytes() == second.read_bytes()


def test_the_section_survives_a_round_trip_through_json(section: AutofillSection) -> None:
    restored = AutofillSection.from_dict(json.loads(json.dumps(section.to_dict())))
    assert restored == section


def test_the_report_verb_attaches_the_section_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole chain: generate, train, evaluate, ingest, report --autofill."""
    from triage.cli import EXIT_OK, main

    data = tmp_path / "data"
    assert main(["autofill", "generate", "--out", str(data), "--n-fields", "1200"]) == EXIT_OK
    runs = tmp_path / "runs"
    assert (
        main(["autofill", "train", "--data", str(data), "--out", str(runs), "--epochs", "3"])
        == EXIT_OK
    )
    weights = next(runs.glob("*/weights.npz"))
    evaluation = tmp_path / "eval"
    assert (
        main(
            [
                "autofill",
                "evaluate",
                "--data",
                str(data),
                "--weights",
                str(weights),
                "--bootstrap",
                "3",
                "--out",
                str(evaluation),
            ]
        )
        == EXIT_OK
    )
    database = tmp_path / "t.db"
    assert main(["ingest", str(runs), "--database", str(database), "--quiet"]) == EXIT_OK

    output = tmp_path / "report.html"
    code = main(
        [
            "report",
            "--database",
            str(database),
            "--baseline",
            "lr0.1_l20.0001",
            "--output",
            str(output),
            "--autofill",
            str(evaluation),
            "--quiet",
        ]
    )
    assert code in {EXIT_OK, 4}, capsys.readouterr()
    assert "Autofill calibration" in output.read_text(encoding="utf-8")


def test_pointing_the_report_at_a_directory_with_no_calibration_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from triage.cli import main

    data = tmp_path / "data"
    main(["autofill", "generate", "--out", str(data), "--n-fields", "600"])
    runs = tmp_path / "runs"
    main(["autofill", "train", "--data", str(data), "--out", str(runs), "--epochs", "2"])
    database = tmp_path / "t.db"
    main(["ingest", str(runs), "--database", str(database), "--quiet"])
    main(
        [
            "report",
            "--database",
            str(database),
            "--baseline",
            "lr0.1_l20.0001",
            "--output",
            str(tmp_path / "r.html"),
            "--autofill",
            str(tmp_path / "nothing-here"),
            "--quiet",
        ]
    )
    assert "no calibration.json" in capsys.readouterr().err
