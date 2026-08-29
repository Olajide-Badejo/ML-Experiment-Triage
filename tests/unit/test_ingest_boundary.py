"""The ingest boundary: run identity, discovery, and the error boundary.

Everything here pins a defect verified against v1.0.0, and every one of them
was a silent one. A sweep that ingested "1 added, 1 updated" and stored a
single row; a sweep whose two real runs were skipped in favour of a stray file
at the root; a poisoned line that cost forty runs rather than one. The theme is
that the boundary must lose a run loudly or not at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from triage.core.store import Store
from triage.ingest import ingest
from triage.parsers import DEFAULT_PARSERS, discover_runs


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


# --------------------------------------------------------------------- D5


def discovered(root: Path) -> list[Path]:
    """The run directories `discover_runs` claims under `root`, sorted."""
    return sorted(path for _parser, path in discover_runs(root, DEFAULT_PARSERS))


def test_a_stray_file_at_the_sweep_root_does_not_swallow_the_sweep(tmp_path: Path) -> None:
    """The verified defect: a sweep of two runs discovered as ONE run, exit 0.

    `walk()` tested the parsers against a directory before descending, so an
    `index.csv` written at the sweep root claimed the root itself and both real
    runs were never looked at.
    """
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    write_run(root / "run_b")
    root.joinpath("index.csv").write_text("step,note\n0,1.0\n", encoding="utf-8")

    assert discovered(root) == [root / "run_a", root / "run_b"]


def test_the_stray_file_case_ingests_both_runs(tmp_path: Path, temp_database: Path) -> None:
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    write_run(root / "run_b")
    root.joinpath("index.csv").write_text("step,note\n0,1.0\n", encoding="utf-8")

    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)
        assert len(store) == 2
    assert len(result.added) == 2


def test_a_run_directory_with_unparseable_subdirectories_is_still_one_run(
    tmp_path: Path,
) -> None:
    """Leaf claiming must not fragment a run that holds checkpoints or plots."""
    run = write_run(tmp_path / "sweep" / "run_a")
    run.joinpath("checkpoints").mkdir()
    run.joinpath("checkpoints", "epoch0.pt").write_bytes(b"not a log")

    assert discovered(tmp_path / "sweep") == [run]


def test_a_nested_tensorboard_layout_yields_the_leaves_not_the_parent(
    tmp_path: Path,
) -> None:
    """`runX/train` and `runX/val` are two runs; `runX` itself is neither."""
    root = tmp_path / "sweep"
    write_run(root / "runX" / "train")
    write_run(root / "runX" / "val")
    root.joinpath("runX", "metrics.csv").write_text("step,loss\n0,1.0\n", encoding="utf-8")

    assert discovered(root) == [root / "runX" / "train", root / "runX" / "val"]


def test_claim_and_descend_decisions_are_logged_at_debug_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    root.joinpath("index.csv").write_text("step,note\n0,1.0\n", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="triage.parsers"):
        discover_runs(root, DEFAULT_PARSERS)

    messages = [record.getMessage() for record in caplog.records]
    assert any("claim" in message and "run_a" in message for message in messages)
    assert any("descend" in message for message in messages)
