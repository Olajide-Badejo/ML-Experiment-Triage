"""The MLflow parser on both of MLflow's backends (E8).

One database file holds many runs and one file store directory holds one, so
the two layouts exercise different halves of the parser: the SQLite reader has
to expand a single source into several runs with distinct identities, and the
file store reader has to recover the same numbers from plain text files without
a YAML parser.

Both committed fixtures encode the reference series that every other parser
fixture encodes, so the strongest assertion available is cheap here too: an
MLflow run and a TensorBoard run produce identical experiments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.make_fixtures import write_mlflow_database
from triage.parsers import DEFAULT_PARSERS, MlflowParser, ParseError, discover_runs

TAGS = ("train/loss", "val/accuracy")

#: The two runs the committed SQLite fixture holds: the reference series, and a
#: short run carrying one NaN point that MLflow stored with `is_nan` set.
SQLITE_RUN = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"
SQLITE_POISONED_RUN = "9f8e7d6c5b4a30291817263544332211"
FILESTORE_RUN = "f1e2d3c4b5a60718293a4b5c6d7e8f90"


@pytest.fixture
def database(fixture_root: Path) -> Path:
    return fixture_root / "mlflow_sqlite" / "mlflow.db"


@pytest.fixture
def file_store_run(fixture_root: Path) -> Path:
    return fixture_root / "mlflow_filestore" / "mlruns" / "0" / FILESTORE_RUN


def a_run(uuid: str, **overrides: object) -> dict[str, object]:
    """One run for `write_mlflow_database`, with two points on one metric."""
    run: dict[str, object] = {
        "uuid": uuid,
        "name": f"run-{uuid[:4]}",
        "status": "FINISHED",
        "params": {"learning_rate": "0.01"},
        "tags": {"mlflow.runName": f"run-{uuid[:4]}"},
        "metrics": {"loss": [(0, 1.0, 1700000000000, 0), (1, 0.5, 1700000001000, 0)]},
    }
    run.update(overrides)
    return run


# ------------------------------------------------------------ the SQLite backend


def test_one_database_expands_into_one_run_per_row(database: Path) -> None:
    """A tracking database is not a run, and the parser must not pretend it is.

    Every other format in this project is one directory per run, so `ingest`
    walks paths and parses each. MLflow's supported backend puts every run of
    every experiment in one file, and reading it as a single experiment would
    average two conditions into one row without saying so.
    """
    parser = MlflowParser()
    assert parser.can_parse(database)

    runs = parser.runs_in(database)

    assert [path.name for path in runs] == [SQLITE_RUN, SQLITE_POISONED_RUN]
    assert all(path.parent == database for path in runs)


def test_the_sqlite_run_id_is_namespaced_by_its_database(
    database: Path, fixture_root: Path
) -> None:
    """D4. The run uuid alone would collide across two ingested databases."""
    parser = MlflowParser()
    run = parser.runs_in(database)[0]

    assert parser.run_id(run, fixture_root) == f"mlflow_sqlite/mlflow/{SQLITE_RUN}"
    assert parser.run_id(run) == f"mlflow/{SQLITE_RUN}"


def test_the_sqlite_backend_reads_the_reference_series(
    database: Path, reference_series: dict[str, np.ndarray]
) -> None:
    parser = MlflowParser()
    run = next(path for path in parser.runs_in(database) if path.name == SQLITE_RUN)

    experiment = parser.parse(run)

    assert experiment.source_format == "mlflow"
    assert experiment.tags == sorted(TAGS)
    for tag in TAGS:
        series = experiment.series(tag)
        np.testing.assert_array_equal(series.steps, np.arange(len(series), dtype=np.int64))
        np.testing.assert_array_equal(series.values, reference_series[tag])


def test_params_become_the_config_with_numbers_read_as_numbers(database: Path) -> None:
    """MLflow stores every param as a string, and a string cannot be swept.

    `numeric_config` drives the sensitivity ranking and `variant_key` groups
    seed replicates, and both compare values rather than text: leaving the
    learning rate as `'0.001'` would put an MLflow sweep outside the analysis
    this tool exists to do.
    """
    parser = MlflowParser()
    run = next(path for path in parser.runs_in(database) if path.name == SQLITE_RUN)

    experiment = parser.parse(run)

    assert experiment.config["learning_rate"] == 0.001
    assert experiment.config["batch_size"] == 32
    assert experiment.config["optimizer"] == "adamw"
    assert experiment.config["amp"] is True
    assert experiment.seed == 0
    assert experiment.numeric_config()["learning_rate"] == 0.001


def test_the_millisecond_timestamps_become_seconds(database: Path) -> None:
    """MLflow logs epoch milliseconds; every other source here is seconds."""
    parser = MlflowParser()
    run = next(path for path in parser.runs_in(database) if path.name == SQLITE_RUN)

    walls = parser.parse(run).series("train/loss").wall_times

    assert walls is not None
    assert walls[0] == pytest.approx(1700000000.0)
    assert walls[1] == pytest.approx(1700000001.0)


def test_a_metric_mlflow_flagged_as_nan_is_dropped_and_counted(database: Path) -> None:
    """`is_nan` is how MLflow stores a NaN, because the column is a FLOAT.

    The value beside the flag is 0.0. Reading it as a measurement would put a
    fabricated zero in the middle of a loss curve, which is exactly the class of
    silent corruption the non finite filter exists to catch, so the flag is
    turned back into the NaN it stands for and dropped where every other format
    is dropped.
    """
    parser = MlflowParser()
    run = next(path for path in parser.runs_in(database) if path.name == SQLITE_POISONED_RUN)

    experiment = parser.parse(run)

    assert experiment.metadata["n_dropped_non_finite"] == 1
    assert 0.0 not in experiment.series("train/loss").values


def test_a_deleted_run_is_not_ingested(tmp_path: Path) -> None:
    """MLflow deletes softly, and a run somebody deleted is not data."""
    database = tmp_path / "mlflow.db"
    write_mlflow_database(
        database,
        [a_run("a" * 32), a_run("b" * 32, lifecycle_stage="deleted")],
    )

    assert [path.name for path in MlflowParser().runs_in(database)] == ["a" * 32]


def test_a_database_that_is_not_mlflow_fails_with_a_message_naming_the_table(
    tmp_path: Path,
) -> None:
    database = tmp_path / "mlflow.db"
    database.write_bytes(b"not a database at all")

    with pytest.raises(ParseError) as error:
        MlflowParser().runs_in(database)

    assert "mlflow.db" in str(error.value)


def test_an_empty_but_valid_database_yields_no_runs(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    write_mlflow_database(database, [])

    assert MlflowParser().runs_in(database) == []


def test_a_fractional_step_stops_the_parse(tmp_path: Path) -> None:
    """D6, through this parser too: a step is an index and 2.75 is not one."""
    database = tmp_path / "mlflow.db"
    write_mlflow_database(
        database,
        [a_run("c" * 32, metrics={"loss": [(2.75, 1.0, 1700000000000, 0)]})],
    )
    run = MlflowParser().runs_in(database)[0]

    with pytest.raises(ParseError, match="fractional step"):
        MlflowParser().parse(run)


def test_the_last_value_written_at_a_step_is_the_one_kept(tmp_path: Path) -> None:
    """MLflow appends, so a resumed run logs one step twice."""
    database = tmp_path / "mlflow.db"
    write_mlflow_database(
        database,
        [
            a_run(
                "d" * 32,
                metrics={
                    "loss": [
                        (0, 1.0, 1700000000000, 0),
                        (0, 0.25, 1700000009000, 0),
                    ]
                },
            )
        ],
    )
    run = MlflowParser().runs_in(database)[0]

    series = MlflowParser().parse(run).series("loss")

    assert len(series) == 1
    assert series.values[0] == pytest.approx(0.25)


def test_parsing_the_database_itself_says_how_many_runs_it_holds(database: Path) -> None:
    """The whole file is not a run, and the refusal has to say what to do."""
    with pytest.raises(ParseError) as error:
        MlflowParser().parse(database)

    assert "2 run" in str(error.value)


def test_the_fingerprint_is_per_run_and_moves_with_the_database(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    write_mlflow_database(database, [a_run("a" * 32), a_run("b" * 32)])
    parser = MlflowParser()
    first, second = parser.runs_in(database)

    before = parser.fingerprint(first, root=tmp_path)
    assert parser.fingerprint(first, root=tmp_path) == before
    assert parser.fingerprint(second, root=tmp_path) != before

    write_mlflow_database(database, [a_run("a" * 32), a_run("b" * 32), a_run("e" * 32)])
    assert parser.fingerprint(first, root=tmp_path) != before


# --------------------------------------------------------- the file store backend


def test_the_file_store_reads_the_reference_series(
    file_store_run: Path, reference_series: dict[str, np.ndarray]
) -> None:
    """A metric key with a slash is a nested file, which is the whole trap.

    `metrics/train/loss` is one metric named `train/loss`, not a directory
    called `train` holding a metric called `loss`, so the key is the path
    relative to the metrics directory rather than the file name.
    """
    parser = MlflowParser()
    assert parser.can_parse(file_store_run)

    experiment = parser.parse(file_store_run)

    assert experiment.run_id == FILESTORE_RUN
    assert experiment.source_format == "mlflow"
    assert experiment.tags == sorted(TAGS)
    for tag in TAGS:
        np.testing.assert_array_equal(experiment.series(tag).values, reference_series[tag])


def test_the_file_store_reads_the_run_name_without_a_yaml_parser(file_store_run: Path) -> None:
    experiment = MlflowParser().parse(file_store_run)

    assert experiment.metadata["mlflow_run_name"] == "mlflow_filestore_run"
    assert experiment.metadata["mlflow_status"] == "FINISHED"
    assert experiment.metadata["mlflow_backend"] == "file"
    assert experiment.config["batch_size"] == 32


def test_the_two_backends_agree_exactly(database: Path, file_store_run: Path) -> None:
    """One tool, one model: the backend a run was logged to changes nothing."""
    parser = MlflowParser()
    run = next(path for path in parser.runs_in(database) if path.name == SQLITE_RUN)

    from_database = parser.parse(run)
    from_files = parser.parse(file_store_run)

    assert from_database.tags == from_files.tags
    assert from_database.config == from_files.config
    for tag in from_database.tags:
        np.testing.assert_array_equal(
            from_database.series(tag).values, from_files.series(tag).values
        )
        np.testing.assert_array_equal(from_database.series(tag).steps, from_files.series(tag).steps)


def test_a_run_with_no_metrics_directory_still_parses(tmp_path: Path) -> None:
    """A run that crashed before its first metric is a run, and reported empty."""
    run = tmp_path / "mlruns" / "0" / ("a" * 32)
    run.mkdir(parents=True)
    (run / "meta.yaml").write_text("run_id: aaaa\nstatus: 4\n", encoding="utf-8")

    experiment = MlflowParser().parse(run)

    assert experiment.metrics == {}
    assert experiment.metadata["mlflow_status"] == "FAILED"


def test_a_metrics_line_that_is_not_three_fields_is_counted_not_fatal(tmp_path: Path) -> None:
    """A killed writer leaves half a line here as it does in JSONL."""
    run = tmp_path / "mlruns" / "0" / ("a" * 32)
    (run / "metrics").mkdir(parents=True)
    (run / "meta.yaml").write_text("run_id: aaaa\n", encoding="utf-8")
    (run / "metrics" / "loss").write_text(
        "1700000000000 1.0 0\n1700000001000 0.5 1\n1700000002", encoding="utf-8"
    )

    experiment = MlflowParser().parse(run)

    assert len(experiment.series("loss")) == 2
    assert experiment.metadata["malformed_lines"] == 1


# -------------------------------------------------------------------- discovery


def test_discovery_expands_a_database_and_claims_file_store_runs(fixture_root: Path) -> None:
    found = {
        parser.run_id(path): parser.format_name
        for parser, path in discover_runs(fixture_root, DEFAULT_PARSERS)
    }

    assert found[f"mlflow/{SQLITE_RUN}"] == "mlflow"
    assert found[f"mlflow/{SQLITE_POISONED_RUN}"] == "mlflow"
    assert found[FILESTORE_RUN] == "mlflow"


def test_an_experiment_directory_is_a_container_rather_than_a_run(fixture_root: Path) -> None:
    """D5, in MLflow's shape: `mlruns/0` holds runs, so it is not one.

    It answers `can_parse` yes, because it does contain `*/meta.yaml`, and leaf
    claiming is what stops that from swallowing every run beneath it.
    """
    experiment_directory = fixture_root / "mlflow_filestore" / "mlruns" / "0"
    assert MlflowParser().can_parse(experiment_directory)

    found = [path for _parser, path in discover_runs(experiment_directory, DEFAULT_PARSERS)]

    assert found == [experiment_directory / FILESTORE_RUN]
