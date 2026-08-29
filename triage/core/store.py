"""SQLite persistence so a run is parsed once and compared thereafter.

Reading a TensorBoard event directory costs hundreds of milliseconds. Reading
the same series back out of SQLite as a compressed float32 blob costs under a
millisecond, and every later comparison, report and sensitivity pass reads from
here. That is the whole argument for a database in a project whose datasets are
a few megabytes; a server based database would add operational cost and buy
nothing at this scale.

Two properties matter and both are tested:

* **Round trip is exact.** Series are stored as raw float32 and int64 buffers,
  zlib compressed. What comes out is bit for bit what went in.
* **Ingest is resumable.** Every run records a fingerprint of its source. An
  unchanged source is skipped, a changed one replaces its rows in a single
  transaction, and interrupting an ingest can lose only the run in flight.

Config and metadata are stored as strict JSON: `allow_nan=False`, with non
finite numbers replaced by `null` first. RFC 8259 has no `NaN` or `Infinity`
literal, so writing one would produce a database that only Python can read,
and null is already how the rest of this codebase spells "no usable value".
"""

from __future__ import annotations

import json
import math
import sqlite3
import zlib
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from triage.core.experiment import Experiment, MetricSeries

#: Version 2 is the D4 identity change: `run_id` is the run's path relative to
#: the ingest root rather than its directory basename, so the ids in a version
#: 1 database do not mean what the ids in a version 2 one mean. Detecting that
#: and migrating or refusing is D14's job; this constant records which of the
#: two a database was written by, which is what a migration will need.
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    run_id        TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    source_format TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    source_hash   TEXT NOT NULL,
    parsed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id          TEXT NOT NULL,
    tag             TEXT NOT NULL,
    n_points        INTEGER NOT NULL,
    steps_blob      BLOB NOT NULL,
    values_blob     BLOB NOT NULL,
    wall_times_blob BLOB,
    PRIMARY KEY (run_id, tag),
    FOREIGN KEY (run_id) REFERENCES experiments(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS metrics_tag_idx ON metrics(tag);
"""


class StoreError(RuntimeError):
    """Raised when a write would corrupt what is already stored."""


def _sanitize(value: Any) -> Any:
    """Replace non finite numbers with null, recursively.

    Python's `json` module emits the bare tokens `NaN`, `Infinity` and
    `-Infinity` by default. Those are not JSON: RFC 8259 has no non finite
    literals, so a config written that way parses back in Python and fails in
    every other reader, and this database is meant to be readable by anything
    that speaks SQLite.

    **The convention, chosen here and applied everywhere:** a non finite
    configuration or metadata value is stored as `null`. Null already means
    "no usable value" throughout this codebase (`Experiment.numeric_config`
    drops non finite entries for exactly the same reason, and a null is
    dropped by the same rule), so a NaN learning rate reads back as an absent
    learning rate rather than as the string "nan", which would silently become
    a categorical value and split a variant key.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _dump_json(value: Any) -> str:
    """Serialise config or metadata as strict, standards compliant JSON."""
    return json.dumps(_sanitize(value), sort_keys=True, default=str, allow_nan=False)


def _compress(array: np.ndarray, dtype: str) -> bytes:
    return zlib.compress(np.ascontiguousarray(array, dtype=dtype).tobytes(), level=6)


def _decompress(blob: bytes, dtype: str) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=dtype)


class Store:
    """A connection to a triage database, usable as a context manager."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    # ---------------------------------------------------------------- writing

    def upsert(self, experiment: Experiment, source_hash: str) -> None:
        """Insert or replace one run and all of its series in one transaction.

        Replacing rather than merging is deliberate. A source whose fingerprint
        changed may have gained tags, lost tags or been rewritten entirely, and
        a merge would leave stale series behind with no way to notice.

        Replacing is only safe when the incoming run IS the stored one, which
        is why the source path is checked first. `run_id` is the PRIMARY KEY,
        and when it was the directory basename two genuinely different runs
        could share it: `sweep_a/seed0` and `sweep_b/seed0` ingested in one
        pass produced a single row holding the second run's values, reported as
        "1 added, 1 updated", with no warning anywhere. D4 made identity
        path relative, which removes that collision; this check catches what
        remains, such as one database fed from two different ingest roots, and
        it raises rather than overwriting so the cost is one run rather than a
        silently wrong comparison.
        """
        stored_path = self.source_path(experiment.run_id)
        if stored_path is not None and stored_path != experiment.source_path:
            raise StoreError(
                f"run id {experiment.run_id!r} is already stored from a different source: "
                f"{stored_path!r} is on record and {experiment.source_path!r} was offered. "
                f"Two runs cannot share one id. Ingest each root into its own database, or "
                f"delete the stored run first if it really was moved"
            )
        parsed_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self.connection:
            self.connection.execute("DELETE FROM metrics WHERE run_id = ?", (experiment.run_id,))
            self.connection.execute(
                """
                INSERT OR REPLACE INTO experiments
                    (run_id, source_path, source_format, config_json,
                     metadata_json, source_hash, parsed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.run_id,
                    experiment.source_path,
                    experiment.source_format,
                    _dump_json(experiment.config),
                    _dump_json(experiment.metadata),
                    source_hash,
                    parsed_at,
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO metrics
                    (run_id, tag, n_points, steps_blob, values_blob, wall_times_blob)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        experiment.run_id,
                        tag,
                        len(series),
                        _compress(series.steps, "<i8"),
                        _compress(series.values, "<f4"),
                        None if series.wall_times is None else _compress(series.wall_times, "<f8"),
                    )
                    for tag, series in sorted(experiment.metrics.items())
                ],
            )

    def delete(self, run_id: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            self.connection.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))

    # ---------------------------------------------------------------- reading

    def source_hash(self, run_id: str) -> str | None:
        """The fingerprint recorded for a run, or None when it is not stored."""
        row = self.connection.execute(
            "SELECT source_hash FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else str(row["source_hash"])

    def source_path(self, run_id: str) -> str | None:
        """The source path recorded for a run, or None when it is not stored."""
        row = self.connection.execute(
            "SELECT source_path FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else str(row["source_path"])

    def is_unchanged(self, run_id: str, source_hash: str) -> bool:
        """True when this run is already stored with exactly this fingerprint."""
        return self.source_hash(run_id) == source_hash

    def run_ids(self) -> list[str]:
        rows = self.connection.execute("SELECT run_id FROM experiments ORDER BY run_id").fetchall()
        return [str(row["run_id"]) for row in rows]

    def tags(self) -> list[str]:
        rows = self.connection.execute("SELECT DISTINCT tag FROM metrics ORDER BY tag").fetchall()
        return [str(row["tag"]) for row in rows]

    def load(self, run_id: str) -> Experiment:
        row = self.connection.execute(
            "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id!r} in {self.path}")
        return self._build(row)

    def load_all(self, run_ids: Iterable[str] | None = None) -> list[Experiment]:
        if run_ids is None:
            rows = self.connection.execute("SELECT * FROM experiments ORDER BY run_id").fetchall()
        else:
            wanted = list(run_ids)
            if not wanted:
                return []
            placeholders = ",".join("?" for _ in wanted)
            rows = self.connection.execute(
                f"SELECT * FROM experiments WHERE run_id IN ({placeholders}) ORDER BY run_id",
                wanted,
            ).fetchall()
        return [self._build(row) for row in rows]

    def __iter__(self) -> Iterator[Experiment]:
        return iter(self.load_all())

    def __len__(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()
        return int(row["n"])

    def _build(self, row: sqlite3.Row) -> Experiment:
        metrics: dict[str, MetricSeries] = {}
        for metric_row in self.connection.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY tag", (row["run_id"],)
        ):
            wall_blob = metric_row["wall_times_blob"]
            metrics[str(metric_row["tag"])] = MetricSeries(
                tag=str(metric_row["tag"]),
                steps=_decompress(metric_row["steps_blob"], "<i8"),
                values=_decompress(metric_row["values_blob"], "<f4"),
                wall_times=None if wall_blob is None else _decompress(wall_blob, "<f8"),
            )
        return Experiment(
            run_id=str(row["run_id"]),
            source_path=str(row["source_path"]),
            source_format=str(row["source_format"]),
            config=json.loads(row["config_json"]),
            metrics=metrics,
            metadata=json.loads(row["metadata_json"]),
        )

    # --------------------------------------------------------------- reporting

    def statistics(self) -> dict[str, Any]:
        """Counts and byte totals, used by the CLI and the report footer."""
        runs = len(self)
        row = self.connection.execute(
            "SELECT COUNT(*) AS series, COALESCE(SUM(n_points), 0) AS points FROM metrics"
        ).fetchone()
        stored = self.connection.execute(
            "SELECT COALESCE(SUM(LENGTH(steps_blob) + LENGTH(values_blob)), 0) AS b FROM metrics"
        ).fetchone()
        points = int(row["points"])
        compressed = int(stored["b"])
        raw = points * 12  # int64 step plus float32 value per point
        return {
            "runs": runs,
            "series": int(row["series"]),
            "points": points,
            "compressed_bytes": compressed,
            "compression_ratio": (raw / compressed) if compressed else 0.0,
            "database_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }
