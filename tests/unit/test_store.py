"""Store round trip and the ingest skip logic.

The claim the whole design rests on is that parsing once into SQLite loses
nothing. "Loses nothing" has to mean bit for bit, not close, or a comparison
run against the database would not be the same comparison as one run against
the logs. These tests hold the store to that.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from triage.core import Experiment, MetricSeries, Store
from triage.core.store import ID_CHUNK, SCHEMA_VERSION, SQLITE_TIMEOUT_SECONDS, StoreError
from triage.ingest import ingest
from triage.parsers import DEFAULT_PARSERS, CsvParser, JsonlParser


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


def test_upsert_refuses_a_run_id_arriving_from_another_source(temp_database: Path) -> None:
    """D4: two distinct runs sharing one id used to overwrite each other silently."""
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")

        impostor = make_experiment()
        impostor.source_path = "/logs/somewhere_else"
        with pytest.raises(StoreError, match="already stored"):
            store.upsert(impostor, "hash-2")

        # The stored row is untouched by the refused write.
        assert store.load("run_a").source_path == "/logs/run_a"
        assert store.source_hash("run_a") == "hash-1"


def test_upsert_accepts_the_same_run_from_the_same_source(temp_database: Path) -> None:
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
        store.upsert(make_experiment(), "hash-2")
        assert store.source_hash("run_a") == "hash-2"


def test_the_schema_version_is_recorded(temp_database: Path) -> None:
    with Store(temp_database) as store:
        row = store.connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    assert int(row["value"]) == SCHEMA_VERSION
    assert SCHEMA_VERSION == 2, "the D4 identity change is not compatible with a version 1 database"


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
    """Run ids are paths relative to the ingest root, one directory deep here.

    These were bare basenames before D4. The fixture tree groups each run under
    a directory naming its format, so every id now carries that prefix, and
    that is the point of the change: the basenames alone are only unique here
    by luck of the fixture naming.
    """
    with Store(temp_database) as store:
        result = ingest(fixture_root, store, parsers=DEFAULT_PARSERS, show_progress=False)
        assert len(result.added) == 7
        assert store.run_ids() == [
            "csv_long/csv_long_run",
            "csv_wide/csv_wide_run",
            "jsonl/jsonl_run",
            "jsonl_truncated/killed_run",
            "tensorboard/tb_run",
            "tpt_jsonl/healthy_steps",
        ]
        # The consumer's step free file is in this tree too, and it is stored as
        # outcomes rather than as a run (E5): it has no series to compare.
        assert store.outcome_run_ids() == ["outcomes/autofill_run"]
        assert result.outcome_runs == ["outcomes/autofill_run"]
        # The undeclared sweep is refused without --outcomes, and the refusal is
        # a legible ParseError rather than a guess at its schema.
        assert [name for name, _ in result.failed] == ["tpt_sweep/sweep_results"]
        formats = {run.run_id: run.source_format for run in store.load_all()}
    assert formats["tensorboard/tb_run"] == "tensorboard"
    assert formats["jsonl/jsonl_run"] == "jsonl"


def test_store_survives_reingest_of_a_real_fixture(fixture_root: Path, temp_database: Path) -> None:
    """The numbers after ingest match the numbers the parser produced directly."""
    direct = CsvParser().parse(fixture_root / "csv_wide" / "csv_wide_run")
    with Store(temp_database) as store:
        ingest(fixture_root / "csv_wide", store, show_progress=False)
        stored = store.load("csv_wide_run")
    for tag in direct.tags:
        np.testing.assert_array_equal(stored.series(tag).values, direct.series(tag).values)


# ------------------------------------------------------- D14: the version gate

#: The version 1 schema, written out rather than imported, because the whole
#: point of a migration test is to build a database the current code did not.
V1_SCHEMA = """
CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE experiments (
    run_id        TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    source_format TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_hash   TEXT NOT NULL,
    parsed_at     TEXT NOT NULL
);
CREATE TABLE metrics (
    run_id          TEXT NOT NULL,
    tag             TEXT NOT NULL,
    n_points        INTEGER NOT NULL,
    steps_blob      BLOB NOT NULL,
    values_blob     BLOB NOT NULL,
    wall_times_blob BLOB,
    PRIMARY KEY (run_id, tag),
    FOREIGN KEY (run_id) REFERENCES experiments(run_id) ON DELETE CASCADE
);
"""


def write_version_one_database(path: Path, version: int = 1) -> None:
    """A database in the v1 shape, stamped with `version`."""
    connection = sqlite3.connect(str(path))
    connection.executescript(V1_SCHEMA)
    connection.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)", (str(version),)
    )
    connection.execute(
        "INSERT INTO experiments VALUES ('seed0', '/logs/seed0', 'csv', '{}', '{}', 'h', 'then')"
    )
    connection.commit()
    connection.close()


def test_a_newer_schema_version_is_refused_by_name(tmp_path: Path) -> None:
    """A v99 database used to open in silence and be rewritten to the current one."""
    path = tmp_path / "future.db"
    write_version_one_database(path, version=99)
    with pytest.raises(StoreError, match="schema version 99"):
        Store(path)
    # And the refused open left the stored version alone rather than stamping it.
    connection = sqlite3.connect(str(path))
    assert connection.execute("SELECT value FROM schema_meta").fetchone()[0] == "99"
    connection.close()


def test_a_version_one_database_migrates_additively(tmp_path: Path) -> None:
    """The v1 rows survive, the v2 columns appear, and the stamp moves forward."""
    path = tmp_path / "old.db"
    write_version_one_database(path)
    with Store(path) as store:
        assert store.schema_version == SCHEMA_VERSION
        assert store.run_ids() == ["seed0"]
        # D4: the ids in a v1 database are basenames and cannot be assumed to
        # mean what a v2 id means, so the migration says so per row rather than
        # pretending the identity changed with the version stamp.
        assert store.legacy_identity_runs() == ["seed0"]
        columns = {
            row["name"] for row in store.connection.execute("PRAGMA table_info(experiments)")
        }
        assert {"ingest_root", "identity_scheme"} <= columns

        # A run written by this version is on the new identity scheme.
        store.upsert(make_experiment(), "hash-1", ingest_root="/logs")
        assert store.legacy_identity_runs() == ["seed0"]
        assert store.ingest_root("run_a") == "/logs"


def test_a_file_that_is_not_a_database_is_refused_as_one(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("this is not a database\n", encoding="utf-8")
    with pytest.raises(StoreError, match="not a triage database"):
        Store(path)


def test_a_directory_named_as_the_database_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "somewhere"
    directory.mkdir()
    with pytest.raises(StoreError):
        Store(directory)


def test_the_database_runs_in_wal_mode_with_a_timeout(temp_database: Path) -> None:
    with Store(temp_database) as store:
        mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert SQLITE_TIMEOUT_SECONDS == 30


# --------------------------------------------------------- D14: read only open


def test_open_read_only_names_a_missing_database_rather_than_creating_one(
    tmp_path: Path,
) -> None:
    """`triage compare --database /nope/typo.db` used to create the typo."""
    missing = tmp_path / "nope" / "typo.db"
    with pytest.raises(StoreError, match="no such database"):
        Store.open_read_only(missing)
    assert not missing.exists()
    assert not missing.parent.exists(), "a read only open must not create the directory either"


def test_open_read_only_reads_what_the_writer_wrote(temp_database: Path) -> None:
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
    with Store.open_read_only(temp_database) as store:
        assert store.read_only
        assert store.run_ids() == ["run_a"]
        assert len(store.load("run_a").series("train/loss")) == 250


def test_open_read_only_refuses_a_write(temp_database: Path) -> None:
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
    with Store.open_read_only(temp_database) as store:
        with pytest.raises(StoreError, match="read only"):
            store.upsert(make_experiment("run_b"), "hash-2")
        with pytest.raises(StoreError, match="read only"):
            store.delete("run_a")


def test_open_read_only_leaves_the_database_bytes_untouched(temp_database: Path) -> None:
    """The reason read paths stopped opening read write: they dirtied the file.

    Opening read write sets `journal_mode`, which is a write into the database
    header, so `triage compare` on an unchanged database produced a changed
    file. A report is evidence and evidence should not modify what it reports on.
    """
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
    before = hashlib.sha256(temp_database.read_bytes()).hexdigest()
    for _ in range(3):
        with Store.open_read_only(temp_database) as store:
            store.load_all()
    assert hashlib.sha256(temp_database.read_bytes()).hexdigest() == before


def test_open_read_only_refuses_a_version_it_cannot_migrate(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    write_version_one_database(path)
    with pytest.raises(StoreError, match="triage ingest"):
        Store.open_read_only(path)


# ------------------------------------------------------ D14: loading id lists


def test_load_all_chunks_an_id_list_longer_than_sqlite_accepts(temp_database: Path) -> None:
    """A 50,000 id `IN (...)` was a verified failure; the chunk size is 900."""
    wanted = [f"run_{index:05d}" for index in range(ID_CHUNK * 3 + 7)]
    with Store(temp_database) as store:
        with store.connection:
            store.connection.executemany(
                "INSERT INTO experiments (run_id, source_path, source_format, config_json, "
                "metadata_json, source_hash, parsed_at, ingest_root, identity_scheme) "
                "VALUES (?, ?, 'csv', '{}', '{}', 'h', 'now', '', 'path')",
                [(run_id, f"/logs/{run_id}") for run_id in wanted],
            )
        loaded = store.load_all(wanted)
    assert [run.run_id for run in loaded] == sorted(wanted)


def test_load_all_raises_on_an_id_that_is_not_stored(temp_database: Path) -> None:
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
        with pytest.raises(StoreError, match="not_here"):
            store.load_all(["run_a", "not_here"])


def test_a_series_whose_point_count_disagrees_with_its_blob_is_refused(
    temp_database: Path,
) -> None:
    """`n_points` was written and never checked against the blob it describes."""
    with Store(temp_database) as store:
        store.upsert(make_experiment(), "hash-1")
        with store.connection:
            store.connection.execute("UPDATE metrics SET n_points = 3 WHERE tag = 'train/loss'")
        with pytest.raises(StoreError, match="n_points"):
            store.load("run_a")


# ------------------------------------------- D36: concurrency and interruption


def test_two_writers_on_one_database_both_finish(temp_database: Path) -> None:
    """Concurrent writers used to raise an uncaught `sqlite3.OperationalError`."""
    # The first connection creates the file, so the two workers below never race
    # on schema creation itself, which is not what this test is about.
    with Store(temp_database) as store:
        store.upsert(make_experiment("warmup"), "hash-0")

    failures: list[BaseException] = []

    def write(prefix: str) -> None:
        try:
            with Store(temp_database) as store:
                for index in range(15):
                    store.upsert(make_experiment(f"{prefix}_{index}", n=40), f"hash-{index}")
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=write, args=(name,)) for name in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not failures, f"a concurrent writer failed: {failures}"
    with Store(temp_database) as store:
        assert len(store) == 31


def test_an_interrupted_ingest_resumes_where_it_stopped(
    tmp_path: Path, temp_database: Path
) -> None:
    """Each run is committed on its own, so a kill costs the run in flight only."""
    root = tmp_path / "sweep"
    for name in ("run_a", "run_b", "run_c"):
        write_run(root, name, 25)

    killed_at = "run_b"

    class Killer(CsvParser):
        """A parser that dies part way through, the way a Ctrl+C does."""

        def parse(self, path: Path, root: Path | None = None) -> Experiment:
            if path.name == killed_at:
                raise KeyboardInterrupt("stopped by the operator")
            return super().parse(path, root)

    with Store(temp_database) as store:
        with pytest.raises(KeyboardInterrupt):
            ingest(root, store, parsers=[Killer(), JsonlParser()], show_progress=False)
        assert store.run_ids() == ["run_a"], "the run in flight must not be half stored"

    with Store(temp_database) as store:
        second = ingest(root, store, show_progress=False)
    assert second.skipped == ["run_a"]
    assert sorted(second.added) == ["run_b", "run_c"]
