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

**Reading does not write.** `Store(path)` is the read write connection ingest
uses: it creates the file, sets the journal mode and stamps the schema version.
`Store.open_read_only(path)` is what `compare` and `report` use, over a SQLite
URI with `mode=ro`, and it is the only honest way to run an analysis: opening
read write named a database that does not exist into existence, so a typo in
`--database` created an empty file and was then diagnosed as an empty sweep,
and setting the journal mode is a write into the database header, so producing
a report changed the bytes of the thing it was reporting on.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import zlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from triage.core.experiment import Experiment, MetricSeries

LOGGER = logging.getLogger("triage.store")

#: Version 2 is the D4 identity change: `run_id` is the run's path relative to
#: the ingest root rather than its directory basename, so the ids in a version
#: 1 database do not mean what the ids in a version 2 one mean, plus the two
#: columns that record which of the two a row was written under.
SCHEMA_VERSION = 2

#: How long a writer waits for another writer's lock before giving up. Without
#: it the default is zero and a second `triage ingest` against one database
#: raised `sqlite3.OperationalError: database is locked` immediately; thirty
#: seconds is far longer than any single run's transaction takes here, so the
#: error now means a genuinely stuck writer rather than a lost race.
SQLITE_TIMEOUT_SECONDS = 30

#: How many ids go into one `IN (...)`. SQLite's compiled default limit on host
#: parameters is 999 (raised to 32766 in 3.32 but still capped, and the cap is a
#: build option nobody controls), and `load_all` built one placeholder per id:
#: verified to fail outright at 50,000 ids. 900 leaves room for the rest of a
#: statement and is small enough to be safe on any build.
ID_CHUNK = 900

#: `identity_scheme` values. `path` is D4's ingest root relative id; `basename`
#: is what a version 1 database holds, and the migration says so rather than
#: relabelling rows whose ids it cannot recompute.
IDENTITY_PATH = "path"
IDENTITY_BASENAME = "basename"

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    run_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    source_format   TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    source_hash     TEXT NOT NULL,
    parsed_at       TEXT NOT NULL,
    ingest_root     TEXT NOT NULL DEFAULT '',
    identity_scheme TEXT NOT NULL DEFAULT 'basename'
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

#: Columns version 2 added to `experiments`, with the DDL to add each to a
#: version 1 table. Both carry a DEFAULT, which is what makes the migration
#: additive: the rows already there acquire truthful values without being
#: rewritten, and `basename` is the truth about a version 1 id.
V2_COLUMNS = {
    "ingest_root": "ALTER TABLE experiments ADD COLUMN ingest_root TEXT NOT NULL DEFAULT ''",
    "identity_scheme": (
        "ALTER TABLE experiments ADD COLUMN identity_scheme TEXT NOT NULL DEFAULT 'basename'"
    ),
}


class StoreError(RuntimeError):
    """Raised when the database cannot be used as one, or a write would corrupt it.

    Every `sqlite3.Error` this module can provoke is translated into this, so a
    caller catches one exception type rather than the union of a database
    driver's exception tree and this package's own. The CLI's error boundary is
    written against exactly that list.
    """


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
        self.read_only = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._wrapping(f"opening {self.path}"):
            self.connection = sqlite3.connect(str(self.path), timeout=SQLITE_TIMEOUT_SECONDS)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            # WAL lets a reader run while a writer holds the lock, which is the
            # shape of every concurrent use this tool has: an ingest finishing
            # while a compare reads. It is a property of the file rather than
            # the connection, so it is set once and inherited thereafter, which
            # is also why a read only open must never try.
            self.connection.execute("PRAGMA journal_mode = WAL")
        self.schema_version = self._read_schema_version()
        self._refuse_a_newer_version()
        with self._wrapping(f"preparing {self.path}"):
            self.connection.executescript(SCHEMA)
            self._migrate()
            self.connection.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.connection.commit()
        self.schema_version = SCHEMA_VERSION

    @classmethod
    def open_read_only(cls, path: str | Path) -> Store:
        """Open an existing database for reading, creating and changing nothing.

        This is what `compare` and `report` use, and both defects it fixes were
        the same mistake seen from two sides. A read write open of a path that
        does not exist CREATES it, so `--database /nope/typo.db` made the typo,
        found no runs in it, and reported an empty sweep rather than a missing
        file; the message here names the file instead. And a read write open
        sets the journal mode, which is a write into the database header, so
        rendering a report modified the database it was rendering, which is not
        a thing evidence should do to its own source.

        A version older than the current one is refused rather than migrated:
        migrating is a write, and a read path that quietly rewrote the database
        would be the defect above wearing a different hat.
        """
        store = cls.__new__(cls)
        store.path = Path(path)
        store.read_only = True
        if not store.path.is_file():
            raise StoreError(
                f"no such database: {store.path}. `compare` and `report` read a database that "
                f"`triage ingest` has already written; check the --database path, or ingest a "
                f"directory of runs into it first"
            )
        uri = f"{store.path.resolve().as_uri()}?mode=ro"
        with store._wrapping(f"opening {store.path} for reading"):
            store.connection = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT_SECONDS)
            store.connection.row_factory = sqlite3.Row
        store.schema_version = store._read_schema_version()
        store._refuse_a_newer_version()
        if store.schema_version < SCHEMA_VERSION:
            store.close()
            raise StoreError(
                f"{store.path} was written at schema version {store.schema_version} and this "
                f"is version {SCHEMA_VERSION}. Reading cannot migrate it, because migrating is "
                f"a write: run `triage ingest` against it once to bring it forward"
            )
        return store

    # ------------------------------------------------------------ housekeeping

    @contextmanager
    def _wrapping(self, action: str) -> Iterator[None]:
        """Translate any `sqlite3.Error` raised inside into a `StoreError`.

        Callers of this module should not have to know that SQLite is what is
        underneath, and the CLI's error boundary is a fixed list of exception
        types. A `sqlite3.DatabaseError` on the first read is also how a file
        that is not a database announces itself, since `connect` is lazy, so
        that case is named explicitly rather than passed through as driver
        wording nobody can act on.
        """
        try:
            yield
        except sqlite3.DatabaseError as error:
            if "file is not a database" in str(error) or "encrypted" in str(error):
                raise StoreError(
                    f"{self.path} is not a triage database: SQLite cannot read it as one "
                    f"({error}). Point --database at a file written by `triage ingest`"
                ) from error
            raise StoreError(f"{action}: {error}") from error
        except sqlite3.Error as error:
            raise StoreError(f"{action}: {error}") from error

    def _read_schema_version(self) -> int:
        """The version stamped in `schema_meta`, or the current one for a new file.

        Read on EVERY open, which it was not: `SCHEMA_VERSION` was written and
        never looked at, so a version 99 database opened in silence and was
        rewritten to version 1, discarding whatever a later version had meant by
        the rows it found.
        """
        with self._wrapping(f"reading the schema version of {self.path}"):
            tables = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_meta'"
            ).fetchone()
            if tables is None:
                # No metadata table at all: either a database this tool has
                # never written, or an empty file about to become one. An empty
                # file is version current by construction; anything else with
                # tables in it is not ours, and `executescript` would say so.
                return SCHEMA_VERSION
            row = self.connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
        if row is None:
            return 1
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 1

    def _refuse_a_newer_version(self) -> None:
        if self.schema_version <= SCHEMA_VERSION:
            return
        self.close()
        raise StoreError(
            f"{self.path} was written at schema version {self.schema_version} and this build of "
            f"triage understands version {SCHEMA_VERSION}. Refusing to open it: a newer version "
            f"may mean something different by the rows it holds. Upgrade triage, or ingest into "
            f"a new database"
        )

    def _migrate(self) -> None:
        """Bring a version 1 database forward, additively and in place.

        Additive means every change is an `ADD COLUMN` with a default, so no row
        is rewritten and no data is discarded. The one thing a migration cannot
        do is recompute identity: D4 made `run_id` the run's path relative to
        the ingest root, and a version 1 database holds basenames whose ingest
        root is not recorded anywhere. So the rows keep the ids they have and
        are labelled `basename`, which `legacy_identity_runs` reports; a later
        `triage ingest` writes the same runs under their path relative ids, and
        the labelling is what lets a reader tell the two apart instead of
        finding out through a comparison that silently paired the wrong runs.
        """
        if self.schema_version >= SCHEMA_VERSION:
            return
        existing = {
            row["name"] for row in self.connection.execute("PRAGMA table_info(experiments)")
        }
        added = [name for name in V2_COLUMNS if name not in existing]
        for name in added:
            self.connection.execute(V2_COLUMNS[name])
        legacy = self.connection.execute(
            "SELECT COUNT(*) AS n FROM experiments WHERE identity_scheme = ?", (IDENTITY_BASENAME,)
        ).fetchone()
        LOGGER.warning(
            "%s migrated from schema version %d to %d (columns added: %s); %d run(s) keep the "
            "version 1 basename identity and will be re ingested under their path relative ids",
            self.path,
            self.schema_version,
            SCHEMA_VERSION,
            ", ".join(added) or "none",
            int(legacy["n"]),
        )

    def _refuse_a_write(self, action: str) -> None:
        if self.read_only:
            raise StoreError(
                f"this database was opened read only and {action} would write to it. "
                f"`compare` and `report` never write; use `triage ingest` to change a database"
            )

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

    def upsert(self, experiment: Experiment, source_hash: str, ingest_root: str = "") -> None:
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

        `ingest_root` is recorded beside the run because a path relative id is
        only interpretable against the root it is relative to, and a database
        can legitimately be fed from more than one.
        """
        self._refuse_a_write("upsert")
        stored_path = self.source_path(experiment.run_id)
        if stored_path is not None and stored_path != experiment.source_path:
            raise StoreError(
                f"run id {experiment.run_id!r} is already stored from a different source: "
                f"{stored_path!r} is on record and {experiment.source_path!r} was offered. "
                f"Two runs cannot share one id. Ingest each root into its own database, or "
                f"delete the stored run first if it really was moved"
            )
        parsed_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._wrapping(f"storing run {experiment.run_id!r}"), self.connection:
            self.connection.execute("DELETE FROM metrics WHERE run_id = ?", (experiment.run_id,))
            self.connection.execute(
                """
                INSERT OR REPLACE INTO experiments
                    (run_id, source_path, source_format, config_json,
                     metadata_json, source_hash, parsed_at, ingest_root, identity_scheme)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.run_id,
                    experiment.source_path,
                    experiment.source_format,
                    _dump_json(experiment.config),
                    _dump_json(experiment.metadata),
                    source_hash,
                    parsed_at,
                    ingest_root,
                    IDENTITY_PATH,
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
        self._refuse_a_write("delete")
        with self._wrapping(f"deleting run {run_id!r}"), self.connection:
            self.connection.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            self.connection.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))

    # ---------------------------------------------------------------- reading

    def source_hash(self, run_id: str) -> str | None:
        """The fingerprint recorded for a run, or None when it is not stored."""
        return self._column("source_hash", run_id)

    def source_path(self, run_id: str) -> str | None:
        """The source path recorded for a run, or None when it is not stored."""
        return self._column("source_path", run_id)

    def ingest_root(self, run_id: str) -> str | None:
        """The ingest root this run's id was taken relative to (D4)."""
        return self._column("ingest_root", run_id)

    def identity_scheme(self, run_id: str) -> str | None:
        """`path` for a version 2 id, `basename` for one carried over from v1."""
        return self._column("identity_scheme", run_id)

    def legacy_identity_runs(self) -> list[str]:
        """Runs whose ids predate D4 and so cannot be compared with new ones.

        A migrated database holds ids that were directory basenames. They are
        still unique inside that file, but two of them may name runs a version 2
        ingest would keep apart, and none of them match the id the same run gets
        today. Naming them is what makes the difference visible before a
        comparison quietly pairs the wrong runs.
        """
        with self._wrapping("listing runs on the version 1 identity scheme"):
            rows = self.connection.execute(
                "SELECT run_id FROM experiments WHERE identity_scheme = ? ORDER BY run_id",
                (IDENTITY_BASENAME,),
            ).fetchall()
        return [str(row["run_id"]) for row in rows]

    def _column(self, column: str, run_id: str) -> str | None:
        with self._wrapping(f"reading {column} for run {run_id!r}"):
            row = self.connection.execute(
                f"SELECT {column} FROM experiments WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else str(row[column])

    def is_empty(self, run_id: str) -> bool:
        """True when this run is stored and holds no series at all.

        Used by ingest to re warn about a run with no scalars on every pass,
        not only the pass that parsed it. A stored fingerprint made the second
        ingest skip such a run in silence, which is how an empty run stopped
        being visible at the point a reader was most likely to trust the tool.
        """
        if self.source_hash(run_id) is None:
            return False
        with self._wrapping(f"counting series for run {run_id!r}"):
            row = self.connection.execute(
                "SELECT COUNT(*) AS n FROM metrics WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["n"]) == 0

    def is_unchanged(self, run_id: str, source_hash: str) -> bool:
        """True when this run is already stored with exactly this fingerprint."""
        return self.source_hash(run_id) == source_hash

    def run_ids(self) -> list[str]:
        with self._wrapping("listing runs"):
            rows = self.connection.execute(
                "SELECT run_id FROM experiments ORDER BY run_id"
            ).fetchall()
        return [str(row["run_id"]) for row in rows]

    def tags(self) -> list[str]:
        with self._wrapping("listing tags"):
            rows = self.connection.execute(
                "SELECT DISTINCT tag FROM metrics ORDER BY tag"
            ).fetchall()
        return [str(row["tag"]) for row in rows]

    def load(self, run_id: str) -> Experiment:
        with self._wrapping(f"loading run {run_id!r}"):
            row = self.connection.execute(
                "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id!r} in {self.path}")
        return self._build(row)

    def load_all(self, run_ids: Iterable[str] | None = None) -> list[Experiment]:
        """Load every run, or the named ones, raising on an id that is not stored.

        The id list is read in chunks of `ID_CHUNK`. It used to be built into a
        single `IN (...)` with one placeholder per id, which SQLite refuses past
        its host parameter limit: verified to fail outright at 50,000 ids, which
        is a size a real sweep archive reaches.

        An id that is not in the database is an error rather than an absence.
        Returning the rows that happened to match let a caller ask for forty runs
        and analyse thirty nine without being told, and the missing one is
        exactly the run somebody would have been looking for.
        """
        if run_ids is None:
            with self._wrapping("loading every run"):
                rows = self.connection.execute(
                    "SELECT * FROM experiments ORDER BY run_id"
                ).fetchall()
            return [self._build(row) for row in rows]

        wanted = list(dict.fromkeys(run_ids))
        if not wanted:
            return []
        rows = []
        with self._wrapping("loading runs by id"):
            for start in range(0, len(wanted), ID_CHUNK):
                chunk = wanted[start : start + ID_CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    self.connection.execute(
                        f"SELECT * FROM experiments WHERE run_id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )
        found = {str(row["run_id"]) for row in rows}
        missing = [run_id for run_id in wanted if run_id not in found]
        if missing:
            shown = ", ".join(repr(run_id) for run_id in missing[:8])
            more = f" and {len(missing) - 8} more" if len(missing) > 8 else ""
            raise StoreError(
                f"{len(missing)} of the {len(wanted)} requested run(s) are not in {self.path}: "
                f"{shown}{more}. Ingest them first, or ask for the ids `run_ids()` reports"
            )
        rows.sort(key=lambda row: str(row["run_id"]))
        return [self._build(row) for row in rows]

    def __iter__(self) -> Iterator[Experiment]:
        return iter(self.load_all())

    def __len__(self) -> int:
        with self._wrapping("counting runs"):
            row = self.connection.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()
        return int(row["n"])

    def _build(self, row: sqlite3.Row) -> Experiment:
        metrics: dict[str, MetricSeries] = {}
        with self._wrapping(f"loading the series of run {str(row['run_id'])!r}"):
            metric_rows = self.connection.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY tag", (row["run_id"],)
            ).fetchall()
        for metric_row in metric_rows:
            wall_blob = metric_row["wall_times_blob"]
            tag = str(metric_row["tag"])
            steps = _decompress(metric_row["steps_blob"], "<i8")
            values = _decompress(metric_row["values_blob"], "<f4")
            # `n_points` was written on every insert and read by nothing, so a
            # truncated or partially rewritten blob came back as a shorter
            # series and every statistic computed from it was quietly wrong. The
            # column is a checksum of length; this is where it earns its keep.
            expected = int(metric_row["n_points"])
            if steps.size != expected or values.size != expected:
                raise StoreError(
                    f"run {str(row['run_id'])!r} tag {tag!r} records n_points = {expected} but "
                    f"its blobs hold {steps.size} step(s) and {values.size} value(s); the row is "
                    f"damaged. Delete the run and ingest it again"
                )
            metrics[tag] = MetricSeries(
                tag=tag,
                steps=steps,
                values=values,
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
        with self._wrapping("summarising the database"):
            row = self.connection.execute(
                "SELECT COUNT(*) AS series, COALESCE(SUM(n_points), 0) AS points FROM metrics"
            ).fetchone()
            stored = self.connection.execute(
                "SELECT COALESCE(SUM(LENGTH(steps_blob) + LENGTH(values_blob)), 0) AS b "
                "FROM metrics"
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
