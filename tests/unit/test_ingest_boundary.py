"""The ingest boundary: run identity, discovery, and the error boundary.

Everything here pins a defect verified against v1.0.0, and every one of them
was a silent one. A sweep that ingested "1 added, 1 updated" and stored a
single row; a sweep whose two real runs were skipped in favour of a stray file
at the root; a poisoned line that cost forty runs rather than one. The theme is
that the boundary must lose a run loudly or not at all.
"""

from __future__ import annotations

from pathlib import Path

from triage.core.store import Store
from triage.ingest import ingest


def write_run(directory: Path, rows: int = 30, value: float = 1.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["step,loss"] + [f"{i},{value / (i + 1):.6f}" for i in range(rows)]
    directory.joinpath("metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def write_poisoned_run(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("metrics.csv").write_text(
        "step,loss\n0,1.0\n1,NaN\n2,inf\n3,0.25\n", encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------------- D1


def test_the_ingest_summary_reports_dropped_non_finite_points(
    tmp_path: Path, temp_database: Path
) -> None:
    """Dropping a point silently is the defect; the count belongs in the summary."""
    root = tmp_path / "sweep"
    write_poisoned_run(root / "poisoned")
    write_run(root / "clean")

    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)

    assert result.dropped_non_finite == 2
    assert "2 non finite points dropped" in result.summary()


def test_a_clean_sweep_says_nothing_about_dropped_points(
    tmp_path: Path, temp_database: Path
) -> None:
    root = tmp_path / "sweep"
    write_run(root / "clean")
    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)
    assert result.dropped_non_finite == 0
    assert "non finite" not in result.summary()
