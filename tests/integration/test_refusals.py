"""A comparison that could not be made has to say so, in every output.

The flagship property of this tool is that it refuses rather than guesses. That
promise was being kept in the statistics layer and broken in the reporting
layer: `compare_all` caught every refusal and moved on, so a database whose runs
were all too short produced a report saying nothing at all had gone wrong.

These tests drive the two real outputs, the terminal table and the HTML file,
and check that a refused condition is NAMED in both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from triage.cli import EXIT_NO_COMPARISONS, main
from triage.core.store import Store
from triage.synthetic import CurveSpec, generate_condition

BASELINE = "base"
REFUSED = "too_short"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """A sweep with one condition the window block mode cannot calibrate."""
    rng = np.random.default_rng(880)
    runs = generate_condition(BASELINE, CurveSpec(), 3, rng) + generate_condition(
        REFUSED, CurveSpec(n_steps=120), 1, rng
    )
    path = tmp_path / "triage.db"
    with Store(path) as store:
        for run in runs:
            store.upsert(run, source_hash=run.run_id)
    return path


def test_the_terminal_output_names_the_condition_it_could_not_compare(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Exit 4, not 0: the only candidate condition here was refused, so this run
    # performed zero comparisons. D22 gave that its own code, because a CI gate
    # that goes green having tested nothing is worse than one that fails.
    code = main(["compare", "--database", str(database), "--baseline", BASELINE, "--quiet"])
    assert code == EXIT_NO_COMPARISONS
    printed = capsys.readouterr().out
    assert "caveats:" in printed
    assert REFUSED in printed
    assert "no comparison" in printed
    assert "cannot be calibrated" in printed


def test_the_report_names_the_condition_it_could_not_compare(
    database: Path, tmp_path: Path
) -> None:
    output = tmp_path / "report.html"
    code = main(
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
    # The report is still written, and it is the report that names the refusal;
    # the code says the run compared nothing (D22).
    assert code == EXIT_NO_COMPARISONS
    html = output.read_text(encoding="utf-8")
    assert "Comparisons not made" in html
    assert REFUSED in html
    assert "cannot be calibrated" in html
