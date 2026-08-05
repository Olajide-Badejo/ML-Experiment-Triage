"""Walk a directory of runs into the database, skipping work already done.

Ingest is the only place that touches raw log files. Everything after it reads
SQLite. The contract it keeps:

* a source whose fingerprint matches the stored one is skipped, not reparsed;
* a source whose fingerprint changed replaces its rows wholesale;
* each run is committed on its own, so an interrupted ingest loses only the run
  in flight and rerunning picks up where it stopped;
* a run that fails to parse is recorded as a failure and the walk continues,
  because one corrupt event file should not cost a sweep of forty runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from triage.core.store import Store
from triage.parsers import DEFAULT_PARSERS, ParseError, Parser, discover_runs
from triage.progress import track


@dataclass
class IngestResult:
    """What one ingest pass did, for the CLI to print and tests to assert on."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.updated) + len(self.skipped) + len(self.failed)

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
    runs = discover_runs(Path(root), parsers)
    result = IngestResult()

    tracked = track(
        runs,
        "ingest",
        enabled=show_progress and bool(runs),
        label=lambda pair: pair[0].run_id(pair[1]),
    )
    for parser, path in tracked:
        run_id = parser.run_id(path)
        try:
            fingerprint = parser.fingerprint(path)
            known = store.source_hash(run_id)
            if not force and known == fingerprint:
                result.skipped.append(run_id)
                continue
            experiment = parser.parse(path)
            store.upsert(experiment, fingerprint)
            if not experiment.metrics:
                result.empty.append(run_id)
            if known is None:
                result.added.append(run_id)
            else:
                result.updated.append(run_id)
        except (ParseError, OSError, ValueError) as error:
            result.failed.append((run_id, str(error)))
    return result
