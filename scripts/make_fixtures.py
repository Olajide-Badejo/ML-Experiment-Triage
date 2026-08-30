"""Regenerate the committed parser fixtures under `tests/fixtures`.

The fixtures are committed rather than built during the test run, so that a
parser regression is caught against bytes that have not moved since the parser
was written. This script exists to rebuild them deliberately, for example after
a TensorBoard format change, and its output is fully determined by the seed
below.

Every fixture carries the same underlying series in a different container, so
the unit suite can assert that all three parsers agree on the numbers.

`--check` regenerates into a temporary directory and reports anything that
differs from the committed tree without touching it, so CI can prove the two
have not drifted apart. Everything is compared byte for byte except the
TensorBoard event file: `EventFileWriter` writes a `file_version` record
stamped with the wall clock at the moment of writing, so identical inputs
produce different bytes. That one is compared through this project's own
parser, which is the claim the fixture is there to support anyway.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SEED = 424242
N_STEPS = 60

#: Compared through the parser rather than by hash, for the reason in the module
#: docstring. Relative to the fixture root.
EVENT_FILE = Path("tensorboard") / "tb_run" / "events.out.tfevents.1700000000.fixture"

#: Compared through the parser too, and for a related reason: SQLite stamps the
#: version of the library that last wrote a database into its header, so a file
#: written by the SQLite in Python 3.13 on Windows and the same file written on
#: a Linux runner differ in bytes that have nothing to do with the fixture.
MLFLOW_DATABASE = Path("mlflow_sqlite") / "mlflow.db"


def reference_series() -> dict[str, np.ndarray]:
    """The one series every fixture encodes, as float32 to match the model."""
    rng = np.random.default_rng(SEED)
    steps = np.arange(N_STEPS, dtype=np.int64)
    loss = (2.5 * np.exp(-steps / 18.0) + 0.20 + rng.normal(0, 0.02, N_STEPS)).astype(np.float32)
    accuracy = (0.95 - 0.8 * np.exp(-steps / 14.0) + rng.normal(0, 0.01, N_STEPS)).astype(
        np.float32
    )
    return {"steps": steps, "train/loss": loss, "val/accuracy": accuracy}


CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adamw",
    "seed": 0,
}


def write_config(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")


def make_tensorboard(series: dict[str, np.ndarray], root: Path) -> None:
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    directory = root / "tensorboard" / "tb_run"
    if directory.exists():
        shutil.rmtree(directory)
    write_config(directory)

    writer = EventFileWriter(str(directory))
    for index, step in enumerate(series["steps"]):
        for tag in ("train/loss", "val/accuracy"):
            summary = summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(series[tag][index]))]
            )
            writer.add_event(
                event_pb2.Event(step=int(step), wall_time=1700000000.0 + index, summary=summary)
            )
    writer.close()

    # The writer names the file after the host and process, which would make the
    # fixture machine specific. Rename it to something stable and committable.
    written = sorted(directory.glob("*tfevents*"))
    if len(written) != 1:
        raise RuntimeError(f"expected one event file, found {len(written)}")
    written[0].rename(directory / "events.out.tfevents.1700000000.fixture")


def make_csv_wide(series: dict[str, np.ndarray], root: Path) -> None:
    directory = root / "csv_wide" / "csv_wide_run"
    write_config(directory)
    lines = ["step,wall_time,train/loss,val/accuracy"]
    for index, step in enumerate(series["steps"]):
        wall = 1700000000.0 + index
        lines.append(
            f"{step},{wall:.1f},{series['train/loss'][index]:.9g},"
            f"{series['val/accuracy'][index]:.9g}"
        )
    (directory / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_csv_long(series: dict[str, np.ndarray], root: Path) -> None:
    directory = root / "csv_long" / "csv_long_run"
    write_config(directory)
    lines = ["step,tag,value"]
    for index, step in enumerate(series["steps"]):
        for tag in ("train/loss", "val/accuracy"):
            lines.append(f"{step},{tag},{series[tag][index]:.9g}")
    (directory / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_jsonl(series: dict[str, np.ndarray], root: Path) -> None:
    directory = root / "jsonl" / "jsonl_run"
    write_config(directory)
    lines = []
    for index, step in enumerate(series["steps"]):
        lines.append(
            json.dumps(
                {
                    "step": int(step),
                    "wall_time": 1700000000.0 + index,
                    "train/loss": float(series["train/loss"][index]),
                    "val/accuracy": float(series["val/accuracy"][index]),
                }
            )
        )
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_truncated_jsonl(series: dict[str, np.ndarray], root: Path) -> None:
    """A JSONL log cut off mid line, which is what a killed job leaves behind."""
    directory = root / "jsonl_truncated" / "killed_run"
    write_config(directory)
    lines = [
        json.dumps({"step": int(step), "train/loss": float(series["train/loss"][index])})
        for index, step in enumerate(series["steps"][:30])
    ]
    text = "\n".join(lines) + '\n{"step": 30, "train/lo'
    (directory / "metrics.jsonl").write_text(text, encoding="utf-8")


def make_tpt_jsonl(series: dict[str, np.ndarray], root: Path) -> None:
    """A second producer's shape: an environment header line, then step rows.

    The PyTorch Performance and Health Toolkit writes its schema v2 logs this
    way, and a resumed sweep writes the header again partway through. Neither
    header carries a step, so a parser that infers its schema from the first
    record alone claims the file and then fails on it. Both headers must be
    skipped and counted, and the eight step rows must ingest cleanly.
    """
    directory = root / "tpt_jsonl" / "healthy_steps"
    write_config(directory)
    header = {
        "type": "environment",
        "run_id": "tpt-healthy-0",
        "row_schema_version": 2,
        "torch_version": "2.9.0",
        "device": "cuda:0",
    }
    lines = [json.dumps(header)]
    for index in range(8):
        if index == 4:
            lines.append(json.dumps(header))
        lines.append(json.dumps({"step": index, "loss": float(series["train/loss"][index])}))
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


#: The field types the outcomes fixture classifies, spelled as the WHATWG
#: autocomplete token values that `Autofill_audit`'s taxonomy uses, so the two
#: repositories agree on the strings without either importing the other (E7b).
LABELS = ("postal-code", "email", "tel", "given-name", "cc-number")

#: How many form templates the outcomes fixture spreads its fields over. The
#: clustered paired test's smallest attainable p value is 2 / 2**n_clusters, so
#: six templates can reach 0.031 and no lower: small enough to enumerate
#: exhaustively, large enough for the result to clear alpha 0.05 at all.
N_TEMPLATES = 6
FIELDS_PER_TEMPLATE = 4


def make_outcomes_jsonl(root: Path) -> None:
    """`Autofill_audit`'s `run.jsonl`, in their filed schema exactly (E5).

    One row per classified field per engine, no step anywhere. The keys and
    their spelling are copied from the schema in their issue and must not drift:
    this fixture is the contract test for the shape they hand us, and a rename
    here would pass while their file stopped parsing.

    The two engines score the SAME fields, which is what makes the comparison a
    paired one, and the fields are clustered by `template_id`, which is what
    makes the pairs non independent: fields on one form template share their
    markup, their locale and their author. The rules engine is given a real
    disadvantage on two of the six templates rather than a uniform one, so the
    cluster structure matters to the answer instead of being decoration.
    """
    rng = np.random.default_rng(SEED)
    directory = root / "outcomes" / "autofill_run"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for template in range(N_TEMPLATES):
        # Two templates the rule engine reads badly, four it reads about as well
        # as the model: a per cluster shift, which is the thing the clustered
        # swap exists to respect.
        rules_rate = 0.45 if template < 2 else 0.86
        ngram_rate = 0.90
        for field_number in range(FIELDS_PER_TEMPLATE):
            label = LABELS[(template + field_number) % len(LABELS)]
            form_id = f"form-{template:02d}-{field_number:02d}"
            for engine, rate in (("rules", rules_rate), ("ngram", ngram_rate)):
                correct = bool(rng.random() < rate)
                rows.append(
                    {
                        "schema_version": "1.0.0",
                        "run_id": "autofill-eval-0",
                        "engine": engine,
                        "split": "test",
                        "form_id": form_id,
                        "template_id": f"template-{template:02d}",
                        "true_label": label,
                        "pred_label": label if correct else "off",
                        "correct": correct,
                        "confidence": round(float(rng.uniform(0.55, 0.99)), 4),
                        "latency_us": round(float(rng.uniform(120.0, 900.0)), 1),
                    }
                )
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (directory / "run.jsonl").write_text(text, encoding="utf-8")


def make_tpt_sweep_jsonl(root: Path) -> None:
    """TPT's `sweep_results.jsonl`: one row per configuration, no step (E6 tail).

    The throughput sweeper writes a row per config rather than a series, so it
    is outcomes shaped and not a training log. It declares no `schema_version`
    on its result rows, which is exactly why this file needs `--outcomes` to be
    read: the strict recognition that claims the autofill file above refuses
    this one rather than guessing at an undeclared format.

    Two runners are measured on the same configurations, joined on `config_key`.
    """
    rng = np.random.default_rng(SEED + 1)
    directory = root / "tpt_sweep" / "sweep_results"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_size in (8, 16, 32, 64):
        for workers in (2, 4):
            key = f"bs{batch_size}_w{workers}"
            base = 220.0 * np.log2(batch_size) + 40.0 * workers
            for runner, gain in (("eager", 1.0), ("compiled", 1.18)):
                rows.append(
                    {
                        "config_key": key,
                        "runner": runner,
                        "batch_size": batch_size,
                        "num_workers": workers,
                        "throughput_samples_per_s": round(
                            float(base * gain + rng.normal(0.0, 12.0)), 3
                        ),
                        "peak_memory_mb": round(float(180.0 * batch_size / 8 + workers * 15), 1),
                        "status": "ok",
                    }
                )
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (directory / "sweep_results.jsonl").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------- MLflow

#: MLflow's tracking schema, in the four tables this project reads and in the
#: spelling MLflow's own SQLAlchemy models produce. Written out rather than
#: generated by MLflow itself, because the point of the fixture is to be
#: readable without MLflow installed, and a schema copied once is the thing the
#: parser is a contract against.
MLFLOW_SCHEMA = """
CREATE TABLE experiments (
    experiment_id INTEGER NOT NULL,
    name VARCHAR(256) NOT NULL,
    artifact_location VARCHAR(256),
    lifecycle_stage VARCHAR(32),
    creation_time BIGINT,
    last_update_time BIGINT,
    CONSTRAINT experiment_pk PRIMARY KEY (experiment_id),
    CONSTRAINT experiments_name_key UNIQUE (name)
);
CREATE TABLE runs (
    run_uuid VARCHAR(32) NOT NULL,
    name VARCHAR(250),
    source_type VARCHAR(20),
    source_name VARCHAR(500),
    entry_point_name VARCHAR(50),
    user_id VARCHAR(256),
    status VARCHAR(9),
    start_time BIGINT,
    end_time BIGINT,
    deleted_time BIGINT,
    source_version VARCHAR(50),
    lifecycle_stage VARCHAR(20),
    artifact_uri VARCHAR(200),
    experiment_id INTEGER,
    CONSTRAINT run_pk PRIMARY KEY (run_uuid)
);
CREATE TABLE metrics (
    key VARCHAR(250) NOT NULL,
    value FLOAT NOT NULL,
    timestamp BIGINT NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    step BIGINT DEFAULT '0' NOT NULL,
    is_nan BOOLEAN DEFAULT '0' NOT NULL,
    CONSTRAINT metric_pk PRIMARY KEY (key, timestamp, step, run_uuid, value, is_nan)
);
CREATE TABLE params (
    key VARCHAR(250) NOT NULL,
    value VARCHAR(8000) NOT NULL,
    run_uuid VARCHAR(32) NOT NULL,
    CONSTRAINT param_pk PRIMARY KEY (key, run_uuid)
);
CREATE TABLE tags (
    key VARCHAR(250) NOT NULL,
    value VARCHAR(5000),
    run_uuid VARCHAR(32) NOT NULL,
    CONSTRAINT tag_pk PRIMARY KEY (key, run_uuid)
);
"""

MLFLOW_EXPERIMENT_ID = 1
MLFLOW_EXPERIMENT_NAME = "triage-fixture"
MLFLOW_START_MS = 1700000000000

#: The reference series, logged to a tracking database.
MLFLOW_SQLITE_RUN = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"
#: A second run in the same database, so the fixture proves that one file is
#: many runs. It carries the NaN that MLflow stores as a zero with `is_nan` set.
MLFLOW_POISONED_RUN = "9f8e7d6c5b4a30291817263544332211"
#: The same reference series again, logged to the file store instead.
MLFLOW_FILESTORE_RUN = "f1e2d3c4b5a60718293a4b5c6d7e8f90"

#: MLflow keeps every param as a string, so the fixture does too. `amp` is here
#: to exercise the one non numeric spelling that must still become a value.
MLFLOW_PARAMS = {
    "learning_rate": "0.001",
    "batch_size": "32",
    "optimizer": "adamw",
    "seed": "0",
    "amp": "True",
}


def write_mlflow_database(
    path: Path,
    runs: list[dict[str, Any]],
    experiments: tuple[tuple[int, str], ...] = ((MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME),),
) -> None:
    """Write an MLflow tracking database at `path`, holding `runs`.

    Each run is a mapping with `uuid` and, optionally, `name`, `status`,
    `lifecycle_stage`, `experiment_id`, `start_time`, `params`, `tags` and
    `metrics`, where `metrics` maps a key to `(step, value, timestamp, is_nan)`
    rows. The unit tests build their edge cases through this function too, so
    the schema above is written down once and every MLflow fixture in the
    project, committed or temporary, is the same shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(MLFLOW_SCHEMA)
        for experiment_id, name in experiments:
            connection.execute(
                "INSERT INTO experiments (experiment_id, name, artifact_location, "
                "lifecycle_stage, creation_time, last_update_time) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    name,
                    f"file:///mlruns/{experiment_id}",
                    "active",
                    MLFLOW_START_MS,
                    MLFLOW_START_MS,
                ),
            )
        for run in runs:
            uuid = str(run["uuid"])
            start = int(run.get("start_time", MLFLOW_START_MS))
            connection.execute(
                "INSERT INTO runs (run_uuid, name, source_type, source_name, entry_point_name, "
                "user_id, status, start_time, end_time, deleted_time, source_version, "
                "lifecycle_stage, artifact_uri, experiment_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    uuid,
                    run.get("name", ""),
                    "LOCAL",
                    "",
                    "",
                    "fixture",
                    run.get("status", "FINISHED"),
                    start,
                    start + 59000,
                    None,
                    "",
                    run.get("lifecycle_stage", "active"),
                    f"file:///mlruns/1/{uuid}/artifacts",
                    int(run.get("experiment_id", MLFLOW_EXPERIMENT_ID)),
                ),
            )
            for key, value in dict(run.get("params", {})).items():
                connection.execute(
                    "INSERT INTO params (key, value, run_uuid) VALUES (?, ?, ?)",
                    (key, value, uuid),
                )
            for key, value in dict(run.get("tags", {})).items():
                connection.execute(
                    "INSERT INTO tags (key, value, run_uuid) VALUES (?, ?, ?)", (key, value, uuid)
                )
            for key, rows in dict(run.get("metrics", {})).items():
                for step, value, timestamp, is_nan in rows:
                    connection.execute(
                        "INSERT INTO metrics (key, value, timestamp, run_uuid, step, is_nan) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (key, value, timestamp, uuid, step, is_nan),
                    )
    connection.close()


def make_mlflow_sqlite(series: dict[str, np.ndarray], root: Path) -> None:
    """MLflow's supported backend: one database file holding two runs."""
    metrics: dict[str, list[tuple[int, float, int, int]]] = {}
    for tag in ("train/loss", "val/accuracy"):
        metrics[tag] = [
            (int(step), float(series[tag][index]), MLFLOW_START_MS + index * 1000, 0)
            for index, step in enumerate(series["steps"])
        ]
    # The poisoned run logs five points, the third of them a NaN. MLflow cannot
    # put a NaN in a FLOAT column, so what it stores is a zero beside a flag,
    # and a reader that took the zero at face value would invent a measurement.
    poisoned = [
        (
            index,
            0.0 if index == 2 else float(series["train/loss"][index]),
            MLFLOW_START_MS + index * 1000,
            int(index == 2),
        )
        for index in range(5)
    ]
    write_mlflow_database(
        root / "mlflow_sqlite" / "mlflow.db",
        [
            {
                "uuid": MLFLOW_SQLITE_RUN,
                "name": "mlflow_sqlite_run",
                "params": MLFLOW_PARAMS,
                "tags": {"mlflow.runName": "mlflow_sqlite_run", "mlflow.source.type": "LOCAL"},
                "metrics": metrics,
            },
            {
                "uuid": MLFLOW_POISONED_RUN,
                "name": "mlflow_diverged_run",
                "start_time": MLFLOW_START_MS + 60000,
                "status": "FAILED",
                "params": MLFLOW_PARAMS,
                "tags": {"mlflow.runName": "mlflow_diverged_run"},
                "metrics": {"train/loss": poisoned},
            },
        ],
    )


def make_mlflow_filestore(series: dict[str, np.ndarray], root: Path) -> None:
    """MLflow's frozen file store: one directory per run, plain text inside.

    A metric key containing a slash becomes a nested file, which is the shape
    of the layout most likely to be read wrongly, so the fixture logs the same
    `train/loss` and `val/accuracy` every other fixture here logs.
    """
    experiment = root / "mlflow_filestore" / "mlruns" / "0"
    run = experiment / MLFLOW_FILESTORE_RUN
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)

    experiment.joinpath("meta.yaml").write_text(
        "artifact_location: file:///mlruns/0\n"
        f"creation_time: {MLFLOW_START_MS}\n"
        "experiment_id: '0'\n"
        f"last_update_time: {MLFLOW_START_MS}\n"
        "lifecycle_stage: active\n"
        f"name: {MLFLOW_EXPERIMENT_NAME}\n",
        encoding="utf-8",
    )
    run.joinpath("meta.yaml").write_text(
        f"artifact_uri: file:///mlruns/0/{MLFLOW_FILESTORE_RUN}/artifacts\n"
        f"end_time: {MLFLOW_START_MS + 59000}\n"
        "entry_point_name: ''\n"
        "experiment_id: '0'\n"
        "lifecycle_stage: active\n"
        f"run_id: {MLFLOW_FILESTORE_RUN}\n"
        "run_name: mlflow_filestore_run\n"
        f"run_uuid: {MLFLOW_FILESTORE_RUN}\n"
        "source_name: ''\n"
        "source_type: 4\n"
        "source_version: ''\n"
        f"start_time: {MLFLOW_START_MS}\n"
        "status: 3\n"
        "tags: []\n"
        "user_id: fixture\n",
        encoding="utf-8",
    )
    for key, value in MLFLOW_PARAMS.items():
        run.joinpath("params").mkdir(exist_ok=True)
        run.joinpath("params", key).write_text(value, encoding="utf-8")
    run.joinpath("tags").mkdir(exist_ok=True)
    run.joinpath("tags", "mlflow.runName").write_text("mlflow_filestore_run", encoding="utf-8")
    for tag in ("train/loss", "val/accuracy"):
        metric_file = run / "metrics" / tag
        metric_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{MLFLOW_START_MS + index * 1000} {float(series[tag][index])!r} {int(step)}"
            for index, step in enumerate(series["steps"])
        ]
        metric_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(root: Path) -> None:
    """Write the whole fixture tree under `root`, which need not exist yet."""
    series = reference_series()
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "reference_series.npy", np.stack([series["train/loss"], series["val/accuracy"]]))
    make_tensorboard(series, root)
    make_csv_wide(series, root)
    make_csv_long(series, root)
    make_jsonl(series, root)
    make_truncated_jsonl(series, root)
    make_tpt_jsonl(series, root)
    make_outcomes_jsonl(root)
    make_tpt_sweep_jsonl(root)
    make_mlflow_sqlite(series, root)
    make_mlflow_filestore(series, root)


def _relative_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def _scalars(experiment: Any) -> dict[str, Any]:
    """One experiment reduced to the things a fixture is a claim about."""
    return {
        "config": experiment.config,
        "metrics": {
            tag: list(zip(series.steps.tolist(), series.values.tolist(), strict=True))
            for tag, series in experiment.metrics.items()
        },
    }


def _event_file_series(directory: Path) -> dict[str, Any]:
    """The event file's scalars, read the way the parser under test reads them."""
    from triage.parsers.tensorboard_parser import TensorBoardParser

    return _scalars(TensorBoardParser().parse(directory))


def _mlflow_database_state(database: Path) -> dict[str, dict[str, Any]]:
    """Every run in a tracking database, read the way the parser reads it."""
    from triage.parsers.mlflow_parser import MlflowParser

    parser = MlflowParser()
    return {path.name: _scalars(parser.parse(path)) for path in parser.runs_in(database)}


def differences(committed: Path) -> list[str]:
    """Every way `committed` differs from a fresh generation, as printable lines.

    An empty list is the whole claim: the committed fixtures are what this
    script produces today. Nothing under `committed` is written to.
    """
    with tempfile.TemporaryDirectory() as work:
        fresh = Path(work) / "fixtures"
        build(fresh)

        found: list[str] = []
        expected = _relative_files(fresh)
        actual = _relative_files(committed)
        for missing in sorted(expected - actual):
            found.append(f"{missing.as_posix()}: missing from the committed fixtures")
        for extra in sorted(actual - expected):
            found.append(f"{extra.as_posix()}: present but no longer generated")

        parsed = {EVENT_FILE, MLFLOW_DATABASE}
        for name in sorted(expected & actual):
            if name in parsed:
                continue
            if (fresh / name).read_bytes() != (committed / name).read_bytes():
                found.append(f"{name.as_posix()}: differs from what the generator writes")

        if EVENT_FILE in expected & actual and _event_file_series(
            fresh / EVENT_FILE.parent
        ) != _event_file_series(committed / EVENT_FILE.parent):
            found.append(
                f"{EVENT_FILE.as_posix()}: decodes to a different series than the generator writes"
            )

        if MLFLOW_DATABASE in expected & actual and _mlflow_database_state(
            fresh / MLFLOW_DATABASE
        ) != _mlflow_database_state(committed / MLFLOW_DATABASE):
            found.append(
                f"{MLFLOW_DATABASE.as_posix()}: holds different runs than the generator writes"
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift between the committed fixtures and the generator, writing nothing",
    )
    args = parser.parse_args()

    if args.check:
        found = differences(FIXTURES)
        for line in found:
            print(line)
        if found:
            print(
                f"\nFAIL: {len(found)} fixture(s) have drifted from "
                f"{Path(__file__).name}. Rerun it without --check and commit the result."
            )
            return 1
        print(f"OK: the fixtures under {FIXTURES.relative_to(ROOT).as_posix()} are current.")
        return 0

    build(FIXTURES)
    print(f"fixtures written under {FIXTURES.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
