"""Walk a directory of runs into the database, skipping work already done.

Ingest is the only place that touches raw log files. Everything after it reads
SQLite. The contract it keeps:

* a source whose fingerprint matches the stored one is skipped, not reparsed;
* a source whose fingerprint changed replaces its rows wholesale;
* each run is committed on its own, so an interrupted ingest loses only the run
  in flight and rerunning picks up where it stopped;
* a run that fails to parse is recorded as a failure and the walk continues,
  because one corrupt event file should not cost a sweep of forty runs;
* every point the non finite filter discarded is counted and reported, because
  a series that quietly lost its diverged tail looks exactly like one that
  never diverged;
* a run that produced no scalar series is reported EVERY pass, including the
  passes that skip it as unchanged, because the second ingest is where an
  empty run used to become invisible.

Diagnostics go through `logging.getLogger("triage.ingest")` and results go to
stdout, so piping a run somewhere does not mix the two.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from triage.core.store import Store
from triage.parsers import DEFAULT_PARSERS, Parser, discover_runs
from triage.progress import track

LOGGER = logging.getLogger("triage.ingest")


@dataclass
class IngestResult:
    """What one ingest pass did, for the CLI to print and tests to assert on."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    #: Non finite points discarded while parsing, per run that lost any. The
    #: filter itself lives in `MetricSeries`; this is where the loss becomes
    #: visible, because a point dropped without a count is a point lost.
    dropped_by_run: dict[str, int] = field(default_factory=dict)
    #: Full traceback per failed run, keyed by run id. `str(error)` alone loses
    #: both which exception it was and where it came from, which is most of
    #: what is needed to tell a corrupt log from a bug in this tool.
    tracebacks: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.updated) + len(self.skipped) + len(self.failed)

    @property
    def dropped_non_finite(self) -> int:
        """Non finite points discarded across every run parsed in this pass."""
        return sum(self.dropped_by_run.values())

    def summary(self) -> str:
        parts = [
            f"{len(self.added)} added",
            f"{len(self.updated)} updated",
            f"{len(self.skipped)} unchanged",
        ]
        if self.empty:
            parts.append(f"{len(self.empty)} with no scalars")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.dropped_non_finite:
            parts.append(
                f"{self.dropped_non_finite} non finite points dropped "
                f"across {len(self.dropped_by_run)} run(s)"
            )
        return ", ".join(parts)


def ingest(
    root: str | Path,
    store: Store,
    parsers: list[Parser] | None = None,
    force: bool = False,
    show_progress: bool = True,
) -> IngestResult:
    """Parse every run under `root` into `store` and report what changed.

    Args:
        root: directory holding one subdirectory per run, or a single log file.
        store: open database to write into.
        parsers: parser order to try; defaults to `DEFAULT_PARSERS`.
        force: reparse even when the source fingerprint is unchanged.
        show_progress: draw a tqdm bar; set False in CI and tests.
    """
    parsers = parsers or DEFAULT_PARSERS
    root = Path(root)
    runs = discover_runs(root, parsers)
    result = IngestResult()
    # Identity and the fingerprint are both taken relative to the ingest root,
    # so a sweep that moves keeps both. A single file passed as the root has
    # its parent as the base, since a file cannot contain the run it is.
    identity_root = root if root.is_dir() else root.parent

    tracked = track(
        runs,
        "ingest",
        enabled=show_progress and bool(runs),
        label=lambda pair: pair[0].run_id(pair[1], identity_root),
    )
    for parser, path in tracked:
        run_id = parser.run_id(path, identity_root)
        try:
            fingerprint = parser.fingerprint(path, root=identity_root)
            known = store.source_hash(run_id)
            if not force and known == fingerprint:
                result.skipped.append(run_id)
                # A skipped run is still checked for emptiness. Storing the
                # fingerprint of a run with no scalars used to silence its
                # warning from the second pass onward, so a run that logged
                # nothing became indistinguishable from a healthy one exactly
                # when a reader had stopped watching for it.
                if store.is_empty(run_id):
                    result.empty.append(run_id)
                    LOGGER.warning("%s is stored with no scalar metrics", run_id)
                continue
            experiment = parser.parse(path, identity_root)
            store.upsert(experiment, fingerprint, ingest_root=str(identity_root))
            dropped = int(experiment.metadata.get("n_dropped_non_finite", 0) or 0)
            if dropped:
                result.dropped_by_run[run_id] = dropped
                LOGGER.warning("%s: dropped %d non finite point(s)", run_id, dropped)
            if not experiment.metrics:
                result.empty.append(run_id)
                LOGGER.warning("%s parsed but carries no scalar metrics", run_id)
            if known is None:
                result.added.append(run_id)
            else:
                result.updated.append(run_id)
        # Exception, not a tuple of the failures anyone thought of. The tuple
        # here was (ParseError, OSError, ValueError), and every gap in it was
        # reachable: OverflowError from a poisoned step, RecursionError from a
        # symlink cycle, sqlite3.Error from a locked database, zlib.error from
        # a truncated blob. Each of those turned one bad run into a dead sweep,
        # which is the exact promise this module's docstring makes and breaks.
        # KeyboardInterrupt and SystemExit inherit from BaseException and so
        # are NOT caught: stopping the ingest has to remain possible.
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            result.failed.append((run_id, detail))
            result.tracebacks[run_id] = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            LOGGER.error("%s failed: %s", run_id, detail, exc_info=error)
    return result
