"""Store round trip and the ingest skip logic.

The claim the whole design rests on is that parsing once into SQLite loses
nothing. "Loses nothing" has to mean bit for bit, not close, or a comparison
run against the database would not be the same comparison as one run against
the logs. These tests hold the store to that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from triage.core import Experiment, MetricSeries, Store
from triage.ingest import ingest
from triage.parsers import DEFAULT_PARSERS, CsvParser


def make_experiment(run_id: str = "run_a", n: int = 250) -> Experiment:
    rng = np.random.default_rng(7)
    steps = np.arange(n, dtype=np.int64)
    return Experiment(
        run_id=run_id,
        source_path=f"/logs/{run_id}",
        source_format="csv",
        config={"learning_rate": 0.003, "batch_size": 64, "seed": 3, "optimizer": "sgd"},
        metrics={
            "train/loss": MetricSeries(
                tag="train/loss",
                steps=steps,
                values=rng.normal(1.0, 0.3, n).astype(np.float32),
                wall_times=1.7e9 + steps.astype(np.float64),
            ),
            "val/accuracy": MetricSeries(
                tag="val/accuracy",
                steps=steps,
                values=rng.uniform(0, 1, n).astype(np.float32),
            ),
        },
        metadata={"note": "synthetic"},
    )


def test_round_trip_is_bit_for_bit(temp_database: Path) -> None:
    original = make_experiment()
    with Store(temp_database) as store:
        store.upsert(original, "hash-1")
        loaded = store.load("run_a")

    assert loaded.run_id == original.run_id
    assert loaded.source_format == original.source_format
    assert loaded.config == original.config
    assert loaded.metadata == original.metadata
    assert loaded.tags == original.tags

    for tag in original.tags:
        before, after = original.series(tag), loaded.series(tag)
        assert after.values.dtype == np.float32
        assert after.steps.dtype == np.int64
        np.testing.assert_array_equal(after.steps, before.steps)
        np.testing.assert_array_equal(after.values, before.values)
        if before.wall_times is None:
            assert after.wall_times is None
        else:
            np.testing.assert_array_equal(after.wall_times, before.wall_times)


def test_reopening_the_database_preserves_everything(temp_database: Path) -> None:
    original = make_experiment()
    with Store(temp_database) as store:
        store.upsert(original, "hash-1")
    with Store(temp_database) as store:
        np.testing.assert_array_equal(
            store.load("run_a").series("train/loss").values,
            original.series("train/loss").values,
        )
        assert store.run_ids() == ["run_a"]
        assert store.tags() == ["train/loss", "val/accuracy"]


def test_upsert_replaces_rather_than_merges(temp_database: Path) -> None:
    """A rewritten source that lost a tag must not leave the old tag behind."""
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
        assert store.load("run_a").tags == ["train/loss", "val/accuracy"]

        shrunk = make_experiment()
        del shrunk.metrics["val/accuracy"]
        store.upsert(shrunk, "hash-2")

        assert store.load("run_a").tags == ["train/loss"]
        assert store.source_hash("run_a") == "hash-2"


def test_unchanged_source_reports_unchanged(temp_database: Path) -> None:
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
        assert store.is_unchanged("run_a", "hash-1")
        assert not store.is_unchanged("run_a", "hash-2")
        assert store.source_hash("missing") is None


def test_delete_removes_the_series_too(temp_database: Path) -> None:
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
        store.delete("run_a")
        assert store.run_ids() == []
        assert store.tags() == []
        with pytest.raises(KeyError):
            store.load("run_a")


def test_statistics_report_real_compression(temp_database: Path) -> None:
    with Store(temp_database) as store:
        store.upsert(make_experiment(n=5000), "hash-1")
        stats = store.statistics()
    assert stats["runs"] == 1
    assert stats["series"] == 2
    assert stats["points"] == 10000
    assert stats["compression_ratio"] > 1.0


# --------------------------------------------------------------- ingest logic


def write_run(root: Path, name: str, rows: int) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["step,loss"] + [f"{i},{1.0 / (i + 1):.6f}" for i in range(rows)]
    (directory / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def test_ingest_skips_unchanged_sources(tmp_path: Path, temp_database: Path) -> None:
    root = tmp_path / "sweep"
    write_run(root, "run_a", 30)
    write_run(root, "run_b", 30)

    with Store(temp_database) as store:
        first = ingest(root, store, show_progress=False)
        assert sorted(first.added) == ["run_a", "run_b"]
        assert first.skipped == []

        second = ingest(root, store, show_progress=False)
        assert second.added == []
        assert sorted(second.skipped) == ["run_a", "run_b"]

        write_run(root, "run_b", 40)
        third = ingest(root, store, show_progress=False)
        assert third.updated == ["run_b"]
        assert third.skipped == ["run_a"]
        assert len(store.load("run_b").series("loss")) == 40


def test_ingest_force_reparses_everything(tmp_path: Path, temp_database: Path) -> None:
    root = tmp_path / "sweep"
    write_run(root, "run_a", 10)
    with Store(temp_database) as store:
        ingest(root, store, show_progress=False)
        forced = ingest(root, store, force=True, show_progress=False)
    assert forced.updated == ["run_a"]
    assert forced.skipped == []


def test_one_bad_run_does_not_stop_the_walk(tmp_path: Path, temp_database: Path) -> None:
    root = tmp_path / "sweep"
    write_run(root, "good_run", 20)
    bad = root / "bad_run"
    bad.mkdir()
    (bad / "metrics.csv").write_text("epoch_number,loss\n0,1.0\n", encoding="utf-8")

    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)
        assert result.added == ["good_run"]
        assert [name for name, _ in result.failed] == ["bad_run"]
        assert store.run_ids() == ["good_run"]
    assert "no step column" in result.failed[0][1]


def test_ingest_reads_a_mixed_format_tree(fixture_root: Path, temp_database: Path) -> None:
    with Store(temp_database) as store:
        result = ingest(fixture_root, store, parsers=DEFAULT_PARSERS, show_progress=False)
        assert len(result.added) == 5
        assert store.run_ids() == [
            "csv_long_run",
            "csv_wide_run",
            "jsonl_run",
            "killed_run",
            "tb_run",
        ]
        formats = {run.run_id: run.source_format for run in store.load_all()}
    assert formats["tb_run"] == "tensorboard"
    assert formats["jsonl_run"] == "jsonl"


def test_store_survives_reingest_of_a_real_fixture(fixture_root: Path, temp_database: Path) -> None:
    """The numbers after ingest match the numbers the parser produced directly."""
    direct = CsvParser().parse(fixture_root / "csv_wide" / "csv_wide_run")
    with Store(temp_database) as store:
        ingest(fixture_root / "csv_wide", store, show_progress=False)
        stored = store.load("csv_wide_run")
    for tag in direct.tags:
        np.testing.assert_array_equal(stored.series(tag).values, direct.series(tag).values)
