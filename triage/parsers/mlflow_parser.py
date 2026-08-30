"""Read MLflow runs from either backend, without depending on MLflow (E8).

MLflow is the one experiment tracker worth reading directly, and reading it
costs nothing: both of its on disk formats are stable, documented, and made of
things the standard library already opens.

**The SQLite backend comes first, because it is the supported one.** A tracking
database holds four tables this parser needs: `runs` (one row per run, keyed by
`run_uuid`), `metrics` (`key`, `value`, `timestamp`, `step`, `is_nan`), `params`
(`key`, `value`) and `tags`. They are read with the standard library's `sqlite3`
in read only mode. The file store (`mlruns/<experiment_id>/<run_id>/`) is read
second: `metrics/<key>` files are newline delimited `timestamp value step` in
plain text, `params/<key>` and `tags/<key>` files hold their value as their
whole contents, and the two or three fields needed out of `meta.yaml` are
extracted with a regular expression rather than by taking a PyYAML dependency
for a file of flat scalars.

**Supported versions, and why this is a safe thing to hand roll.** The SQLite
schema above is MLflow's SQLAlchemy tracking schema, unchanged in the fields
used here since MLflow 1.x, and read here through MLflow 3.x; the `is_nan`
column arrived in MLflow 1.9 and its absence is detected and tolerated. The
file store layout has been frozen for longer still: MLflow put it into
maintenance mode and switched the default backend to SQLite in MLflow 3.7
(December 2025), so the format this reads is one that has stopped moving.
Neither reader is affected by MLflow's Python API, which is why this parser
takes no dependency on it.

**One database is many runs.** Every other format here is one directory per
run, and reading a whole tracking database into one `Experiment` would fold
several conditions into a single row. `runs_in` therefore expands a database
into one path per run, spelled `<database>/<run_uuid>`, and `run_id`,
`fingerprint` and `parse` all understand that spelling. The path does not exist
on disk and is not meant to: it is a name for a row.

**Params become the config, with numbers read as numbers.** MLflow stores every
param as a string. Leaving them as strings would put an MLflow sweep outside
the analysis this tool exists to do, since `numeric_config` drives the
sensitivity ranking and `variant_key` groups seed replicates by comparing
values. Only a canonical spelling is converted, so a param that reads `007` or
`1_0` stays the string it is.

This module imports numpy and the standard library and nothing else, so it
works in a core install exactly as the JSONL parser does (E1).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from triage.core.experiment import Experiment, MetricSeries
from triage.parsers.base import (
    ParseError,
    Parser,
    non_finite_metadata,
    read_text,
    validate_step,
)

LOGGER = logging.getLogger("triage.parsers")

#: The tracking database's file name. Matching on the name rather than on the
#: suffix is deliberate: this project writes its own `triage.db` beside the runs
#: it ingested, and a parser that claimed every SQLite file it walked past would
#: try to read that one back in.
DATABASE_NAME = "mlflow.db"

META_FILENAME = "meta.yaml"
METRICS_DIRNAME = "metrics"
PARAMS_DIRNAME = "params"
TAGS_DIRNAME = "tags"

#: MLflow's `RunStatus` enum. The file store writes the integer and the SQLite
#: backend writes the name, so one of the two has to be translated for a report
#: to be able to say "this run failed" whichever backend it came from.
RUN_STATUS: dict[str, str] = {
    "1": "RUNNING",
    "2": "SCHEDULED",
    "3": "FINISHED",
    "4": "FAILED",
    "5": "KILLED",
}

#: `key: value` in a flat YAML mapping, which is all `meta.yaml` ever is. Block
#: structure is deliberately not handled: a nested value would be silently
#: mis-read by a regular expression, so only lines that are plainly a scalar
#: entry are taken and everything else is ignored.
META_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.\-]*):[ \t]*(.*?)[ \t]*$")

#: Soft deleted runs. MLflow's delete marks the row and leaves it in place, so a
#: reader that ignored this would resurrect every run its user had thrown away.
DELETED = "deleted"


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def config_value(raw: str) -> Any:
    """One MLflow param string as the value it spells, or the string itself.

    Conversion is deliberately narrow. `int()` and `float()` accept spellings
    that are not the number they look like (`007`, `1_000`, `inf`), and a param
    is often an identifier that merely happens to be digits, so a value is
    converted only when the number converts back to exactly the text that was
    stored. Everything else stays a string, where it can still be compared and
    grouped but is never treated as a swept hyperparameter.
    """
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if "_" in text:
        return text
    try:
        as_integer = int(text)
    except ValueError:
        pass
    else:
        if str(as_integer) == text:
            return as_integer
    try:
        as_float = float(text)
    except ValueError:
        return text
    return as_float if math.isfinite(as_float) else text


def _is_database(path: Path) -> bool:
    return path.name.lower() == DATABASE_NAME


def _series_from(
    points: dict[str, list[tuple[int, float, float]]],
) -> dict[str, MetricSeries]:
    """Build one series per metric key, sorted by key for a stable tag order."""
    metrics: dict[str, MetricSeries] = {}
    for tag, rows in sorted(points.items()):
        metrics[tag] = MetricSeries(
            tag=tag,
            steps=np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows)),
            values=np.fromiter((row[1] for row in rows), dtype=np.float32, count=len(rows)),
            wall_times=np.fromiter((row[2] for row in rows), dtype=np.float64, count=len(rows)),
        )
    return metrics


class MlflowParser(Parser):
    """Parses MLflow runs from a tracking database or from the file store."""

    format_name = "mlflow"

    def __init__(self) -> None:
        # `fingerprint` is called once per run and a database holds many, so the
        # digest of the file is computed once per (file, mtime, size) rather
        # than once per run. Without this, ingesting a database of two hundred
        # runs would read and hash the whole file two hundred times.
        self._digests: dict[tuple[str, int, int], str] = {}

    # --------------------------------------------------------------- claiming

    def can_parse(self, path: Path) -> bool:
        if path.is_file():
            return _is_database(path)
        if not path.is_dir():
            # A run inside a database: `<database>/<run_uuid>`, which `runs_in`
            # invented and which has no directory entry of its own.
            return self._split(path) is not None
        if self._database_in(path) is not None:
            return True
        if self._is_run_directory(path):
            return True
        # An experiment directory holds `<run_id>/meta.yaml`. It answers yes so
        # that a user can point `triage ingest` straight at `mlruns/`, and leaf
        # claiming in `discover_runs` is what keeps it a container rather than
        # letting it swallow the runs beneath it (D5).
        return any((child / META_FILENAME).is_file() for child in path.iterdir() if child.is_dir())

    def runs_in(self, path: Path) -> list[Path]:
        """One path per run: a file store run is itself, a database expands."""
        database = self._database_at(path)
        if database is None:
            return [path]
        return [database / uuid for uuid in self._run_uuids(database)]

    # --------------------------------------------------------------- identity

    def run_id(self, path: Path, root: Path | None = None) -> str:
        """`<database stem>/<run_uuid>` for a database run, else the usual rule.

        The uuid alone is unique inside one database and not across two, and two
        databases ingested in one pass is the ordinary case for somebody
        comparing last month's tracking file with this month's. Namespacing by
        the path to the database is the same rule D4 applies to every other
        format, applied to the one source that is a file rather than a
        directory.
        """
        split = self._split(path)
        if split is None:
            return super().run_id(path, root)
        database, run_uuid = split
        return f"{super().run_id(database, root)}/{run_uuid}"

    def fingerprint(self, path: Path, root: Path | None = None) -> str:
        """The database's fingerprint, mixed with which run this is.

        A run inside a database has no file of its own to stat, and the honest
        unit of change is the file: MLflow appends metric rows to shared tables,
        so the cheap per run summaries that could be read instead (a row count,
        a maximum timestamp) would all miss a value that was rewritten in place.
        Every run in a database that changed is therefore reparsed, which is
        conservative in the direction that cannot lose data, and the file is
        hashed once per pass rather than once per run.
        """
        split = self._split(path)
        if split is None:
            return super().fingerprint(path, root)
        database, run_uuid = split
        digest = hashlib.sha256()
        digest.update(self._database_fingerprint(database, root).encode("utf-8"))
        digest.update(run_uuid.encode("utf-8"))
        return digest.hexdigest()

    # ------------------------------------------------------------------ parse

    def parse(self, path: Path, root: Path | None = None) -> Experiment:
        split = self._split(path)
        if split is not None:
            database, run_uuid = split
            return self._parse_database_run(database, run_uuid, path, root)

        whole_database = self._database_at(path)
        if whole_database is not None:
            # Only reachable when somebody calls the parser directly: `ingest`
            # expands a database through `runs_in` first. Saying how many runs
            # are in there is the whole of the remedy.
            uuids = self._run_uuids(whole_database)
            example = (whole_database / uuids[0]).as_posix() if uuids else "<database>/<run_uuid>"
            raise ParseError(
                f"{whole_database} is an MLflow tracking database holding {len(uuids)} run(s), "
                f"not a single run. Parse one of them by name, for example {example}, "
                f"or point `triage ingest` at the database and it will read them all"
            )
        return self._parse_file_store_run(path, root)

    # -------------------------------------------------------- the SQLite side

    def _split(self, path: Path) -> tuple[Path, str] | None:
        """`(database, run_uuid)` when `path` names a run inside a database."""
        parent = path.parent
        if parent.is_file() and _is_database(parent):
            return parent, path.name
        return None

    def _database_in(self, directory: Path) -> Path | None:
        """The tracking database directly inside `directory`, if there is one."""
        candidate = directory / DATABASE_NAME
        return candidate if candidate.is_file() else None

    def _database_at(self, path: Path) -> Path | None:
        """The database `path` names, whether as the file or as its directory."""
        if path.is_file():
            return path if _is_database(path) else None
        if path.is_dir():
            return self._database_in(path)
        return None

    def _connect(self, database: Path) -> sqlite3.Connection:
        """Open the tracking database read only, or raise `ParseError`.

        Read only is not a precaution against this code writing; it is a
        precaution against SQLite writing. Opening a database read/write creates
        the journal files beside it and can recover a hot journal, which is a
        surprising thing for a tool that was asked to read somebody's records to
        do to them.
        """
        try:
            connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
        except (sqlite3.Error, ValueError) as error:
            raise ParseError(f"{database} could not be opened as SQLite: {error}") from error
        connection.row_factory = sqlite3.Row
        return connection

    def _query(
        self, connection: sqlite3.Connection, database: Path, sql: str, *arguments: object
    ) -> list[sqlite3.Row]:
        try:
            return connection.execute(sql, arguments).fetchall()
        except sqlite3.Error as error:
            raise ParseError(
                f"{database} is not an MLflow tracking database: {error}. MLflow's SQLite "
                f"backend holds the tables runs, metrics, params and tags"
            ) from error

    def _run_uuids(self, database: Path) -> list[str]:
        """Every run in the database that its owner has not deleted."""
        with closing(self._connect(database)) as connection:
            rows = self._query(
                connection,
                database,
                "SELECT run_uuid, lifecycle_stage FROM runs ORDER BY start_time, run_uuid",
            )
        return [
            str(row["run_uuid"])
            for row in rows
            if str(row["lifecycle_stage"] or "").lower() != DELETED
        ]

    def _database_fingerprint(self, database: Path, root: Path | None) -> str:
        stat = database.stat()
        key = (str(database.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = self._digests.get(key)
        if cached is None:
            cached = super().fingerprint(database, root)
            self._digests[key] = cached
        return cached

    def _parse_database_run(
        self, database: Path, run_uuid: str, path: Path, root: Path | None
    ) -> Experiment:
        with closing(self._connect(database)) as connection:
            runs = self._query(
                connection, database, "SELECT * FROM runs WHERE run_uuid = ?", run_uuid
            )
            if not runs:
                raise ParseError(f"{database} holds no run with the id {run_uuid!r}")
            run = runs[0]
            columns = {
                str(row["name"])
                for row in self._query(connection, database, "PRAGMA table_info(metrics)")
            }
            # `is_nan` arrived in MLflow 1.9. An older database has the column
            # missing rather than empty, so it is selected only when it is there.
            nan_column = "is_nan" if "is_nan" in columns else "0 AS is_nan"
            metric_rows = self._query(
                connection,
                database,
                f"SELECT key, value, timestamp, step, {nan_column} FROM metrics "
                f"WHERE run_uuid = ? ORDER BY key, step, timestamp",
                run_uuid,
            )
            params = self._query(
                connection, database, "SELECT key, value FROM params WHERE run_uuid = ?", run_uuid
            )
            tags = self._query(
                connection, database, "SELECT key, value FROM tags WHERE run_uuid = ?", run_uuid
            )
            experiments = self._query(
                connection,
                database,
                "SELECT name FROM experiments WHERE experiment_id = ?",
                run["experiment_id"],
            )

        source = f"{database.name} run {run_uuid}"
        points: dict[str, list[tuple[int, float, float]]] = {}
        for row in metric_rows:
            step = validate_step(row["step"] if row["step"] is not None else 0, source, "step")
            # MLflow cannot store a NaN in a FLOAT column, so it stores zero and
            # sets the flag. Reading the zero would drop a fabricated point into
            # the middle of a loss curve; the NaN is restored instead and the
            # filter in `MetricSeries` counts it like every other lost point.
            value = math.nan if row["is_nan"] else float(row["value"])
            timestamp = float(row["timestamp"] or 0) / 1000.0
            points.setdefault(str(row["key"]), []).append((step, value, timestamp))

        metrics = _series_from(points)
        run_tags = {str(row["key"]): str(row["value"]) for row in tags}
        return Experiment(
            run_id=self.run_id(path, root),
            source_path=str(path),
            source_format=self.format_name,
            config={str(row["key"]): config_value(str(row["value"])) for row in params},
            metrics=metrics,
            metadata={
                "mlflow_backend": "sqlite",
                "mlflow_database": database.name,
                "mlflow_run_uuid": run_uuid,
                "mlflow_run_name": self._run_name(run["name"], run_tags),
                "mlflow_experiment_id": str(run["experiment_id"]),
                "mlflow_experiment_name": str(experiments[0]["name"]) if experiments else "",
                "mlflow_status": self._status(run["status"]),
                "mlflow_tags": dict(sorted(run_tags.items())),
                "scalar_tags": sorted(metrics),
                **non_finite_metadata(metrics),
            },
        )

    # ----------------------------------------------------- the file store side

    def _is_run_directory(self, directory: Path) -> bool:
        """A run directory is a `meta.yaml` beside metrics, params or tags.

        `meta.yaml` alone is not enough: an experiment directory has one too,
        and claiming that as a run would produce an experiment with no series
        and hide every real run under it.
        """
        if not (directory / META_FILENAME).is_file():
            return False
        return any(
            (directory / name).is_dir() for name in (METRICS_DIRNAME, PARAMS_DIRNAME, TAGS_DIRNAME)
        )

    def _meta(self, path: Path) -> dict[str, str]:
        """The flat scalar entries of a `meta.yaml`, by key."""
        found: dict[str, str] = {}
        for line in read_text(path).splitlines():
            match = META_LINE.match(line)
            if match is not None:
                found[match.group(1)] = _unquote(match.group(2))
        return found

    def _leaf_values(self, directory: Path) -> dict[str, str]:
        """`{key: contents}` for `params/` and `tags/`, keys nested by slash."""
        if not directory.is_dir():
            return {}
        found: dict[str, str] = {}
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                found[path.relative_to(directory).as_posix()] = read_text(path).strip()
        return found

    def _parse_file_store_run(self, path: Path, root: Path | None) -> Experiment:
        if not path.is_dir():
            raise ParseError(f"{path} is not an MLflow run directory")
        meta_path = path / META_FILENAME
        if not meta_path.is_file():
            raise ParseError(f"{path} has no {META_FILENAME}, so it is not an MLflow run directory")
        meta = self._meta(meta_path)

        points: dict[str, list[tuple[int, float, float]]] = {}
        malformed = 0
        metrics_root = path / METRICS_DIRNAME
        if metrics_root.is_dir():
            for metric_file in sorted(p for p in metrics_root.rglob("*") if p.is_file()):
                tag = metric_file.relative_to(metrics_root).as_posix()
                rows, bad = self._read_metric_file(metric_file, tag)
                points.setdefault(tag, []).extend(rows)
                malformed += bad
        # Sorting by step keeps the "later write wins" rule of `MetricSeries`
        # pointed at the later TIMESTAMP, which is what a resumed run means by
        # writing one step twice.
        for rows in points.values():
            rows.sort(key=lambda row: (row[0], row[2]))

        metrics = _series_from(points)
        tags = self._leaf_values(path / TAGS_DIRNAME)
        return Experiment(
            run_id=self.run_id(path, root),
            source_path=str(path),
            source_format=self.format_name,
            config={
                key: config_value(value)
                for key, value in self._leaf_values(path / PARAMS_DIRNAME).items()
            },
            metrics=metrics,
            metadata={
                "mlflow_backend": "file",
                "mlflow_run_uuid": meta.get("run_uuid") or meta.get("run_id", ""),
                "mlflow_run_name": self._run_name(meta.get("run_name"), tags),
                "mlflow_experiment_id": meta.get("experiment_id", ""),
                "mlflow_status": self._status(meta.get("status")),
                "mlflow_tags": dict(sorted(tags.items())),
                "malformed_lines": malformed,
                "scalar_tags": sorted(metrics),
                **non_finite_metadata(metrics),
            },
        )

    def _read_metric_file(self, path: Path, tag: str) -> tuple[list[tuple[int, float, float]], int]:
        """One `metrics/<key>` file: `timestamp value step` per line.

        A line that is not three fields is counted rather than fatal, for the
        reason the JSONL parser counts a truncated record: a killed writer
        leaves half a line behind, and losing the run over it would be wrong.
        A line that IS three fields but holds a step that is not an integer is
        not that case, and stops the parse like every other poisoned step (D6).
        """
        rows: list[tuple[int, float, float]] = []
        malformed = 0
        source = f"{path.name} in {path.parent}"
        for line in read_text(path).splitlines():
            fields = line.split()
            if len(fields) < 2:
                if line.strip():
                    malformed += 1
                continue
            try:
                timestamp = float(fields[0])
                value = float(fields[1])
            except ValueError:
                malformed += 1
                continue
            # The step column was added to the file store in MLflow 1.0 and an
            # older file has two fields, where every point is step zero.
            raw_step: Any = fields[2] if len(fields) > 2 else 0
            if isinstance(raw_step, str):
                try:
                    raw_step = float(raw_step)
                except ValueError:
                    malformed += 1
                    continue
            rows.append((validate_step(raw_step, source, tag), value, timestamp / 1000.0))
        return rows, malformed

    # ----------------------------------------------------------------- shared

    @staticmethod
    def _run_name(recorded: object, tags: dict[str, str]) -> str:
        """The run's name, which MLflow keeps in two places and drops from one.

        Runs created before MLflow 2.0 have an empty `name` column and carry
        the name in the `mlflow.runName` tag instead; runs created since have
        both. Preferring the column and falling back to the tag reads either.
        """
        name = str(recorded or "")
        return name or tags.get("mlflow.runName", "")

    @staticmethod
    def _status(recorded: object) -> str:
        """The run status as a name, from either the integer or the name."""
        text = str(recorded or "").strip()
        return RUN_STATUS.get(text, text)
