"""The parser contract and the run discovery walk.

Every parser answers two questions about a path: can I read this, and what
`Experiment` does it contain. A third method produces a fingerprint of the
source so that `ingest` can skip work that has already been done.

Run identity comes from the directory name. The layout this assumes is one
directory per run, holding whichever log format that run produced plus an
optional `config.json`:

    demo_sweep/
        lr0.001_bs32_seed0/
            config.json
            events.out.tfevents.1234567890.host
        lr0.010_bs32_seed0/
            config.json
            metrics.csv

A bare file is also accepted, in which case the run takes its name from the
file stem. That is a convenience for one off comparisons, not the layout the
demo or the tests use.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from triage.core.experiment import Experiment, MetricSeries

CONFIG_FILENAME = "config.json"


#: Widest step this tool accepts. int64 is the storage type, and a value beyond
#: this silently casts to INT64_MIN behind a numpy warning and then sorts to the
#: front of the series, so it is refused instead.
MAX_STEP = 2**62


def validate_step(raw: Any, source: str, column: str) -> int:
    """Check one step value and return it as an int, or raise `ParseError`.

    The step axis is integral by definition here: a series is indexed by
    training step, and two points cannot share one index. A fractional value
    such as an epoch of 2.75 is therefore not a step, and casting it would
    truncate 12 fractional epochs into 3 duplicate integers and destroy the
    series, which is what used to happen. Infinity and NaN are refused for the
    same reason, and so is a magnitude int64 cannot hold.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ParseError(
            f"{source}: column {column!r} holds {raw!r}, which is not a step; "
            f"steps must be integers"
        )
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ParseError(
                f"{source}: column {column!r} holds the non finite step {raw!r}; "
                f"steps must be finite integers"
            )
        if raw != int(raw):
            raise ParseError(
                f"{source}: column {column!r} holds the fractional step {raw!r}; "
                f"steps must be integers. Log an integer step alongside the "
                f"fractional value, or scale it (for example epoch 2.75 at 400 "
                f"steps per epoch is step 1100)"
            )
    step = int(raw)
    if abs(step) > MAX_STEP:
        raise ParseError(
            f"{source}: column {column!r} holds the step {raw!r}, out of range for "
            f"the int64 step axis; steps must satisfy abs(step) <= {MAX_STEP}"
        )
    return step


def read_text(path: Path) -> str:
    """Read a UTF-8 file, translating a decode failure into `ParseError`.

    The documented contract is that a parser raises `ParseError` when a source
    it matched cannot be read. A bare `UnicodeDecodeError` breaks that: it is
    not caught by callers that catch `ParseError`, and it names an offset into
    an anonymous buffer rather than a file. The byte offset is the useful part
    of the message, so it is kept and the file is named alongside it.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ParseError(
            f"{path} is not valid UTF-8: byte {error.start} ("
            f"0x{path.read_bytes()[error.start]:02x}) cannot be decoded ({error.reason}). "
            f"Re encode the file as UTF-8"
        ) from error
    except OSError as error:
        raise ParseError(f"{path} could not be read: {error}") from error


def non_finite_metadata(metrics: dict[str, MetricSeries]) -> dict[str, Any]:
    """Metadata entries reporting what the non finite filter discarded.

    Every parser merges this into its experiment metadata, so the loss is
    visible in the same place whatever format the run was written in, and the
    ingest summary can report it without knowing which parser ran.
    """
    dropped = {tag: series.dropped_non_finite for tag, series in metrics.items()}
    dropped = {tag: count for tag, count in sorted(dropped.items()) if count}
    return {"dropped_non_finite": dropped, "n_dropped_non_finite": sum(dropped.values())}


class ParseError(RuntimeError):
    """Raised when a source matched a parser but could not be read."""


class Parser(ABC):
    """Reads one log format into the shared `Experiment` model."""

    #: Short name recorded on the experiment, for example `tensorboard`.
    format_name: str = "unknown"

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """True when this parser recognises `path` as one run it can read."""

    @abstractmethod
    def parse(self, path: Path) -> Experiment:
        """Read `path` into an `Experiment`. Raises `ParseError` on bad input."""

    # ------------------------------------------------------------- shared help

    def run_id(self, path: Path) -> str:
        return path.name if path.is_dir() else path.stem

    def config_for(self, path: Path) -> dict[str, Any]:
        """Load the `config.json` sitting beside the run, or an empty config."""
        directory = path if path.is_dir() else path.parent
        config_path = directory / CONFIG_FILENAME
        if not config_path.exists():
            return {}
        try:
            loaded = json.loads(read_text(config_path))
        except json.JSONDecodeError as error:
            raise ParseError(f"{config_path} is not readable JSON: {error}") from error
        if not isinstance(loaded, dict):
            raise ParseError(f"{config_path} must hold a JSON object, got {type(loaded).__name__}")
        return loaded

    def fingerprint(self, path: Path) -> str:
        """Hash of the source path, modification times and sizes.

        This is what `ingest` compares against the stored value to decide
        whether a run needs reparsing. Content is deliberately not hashed: for
        a directory of event files that would cost as much as parsing, which
        would defeat the purpose of the check.
        """
        digest = hashlib.sha256()
        digest.update(str(path.resolve()).encode("utf-8"))
        targets = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for target in targets:
            stat = target.stat()
            digest.update(target.name.encode("utf-8"))
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
        return digest.hexdigest()


def discover_runs(root: Path, parsers: list[Parser]) -> list[tuple[Parser, Path]]:
    """Find every run under `root`, pairing each with the parser that claims it.

    The walk is top down and does not descend into a directory that has already
    been claimed, so a run directory containing subdirectories of checkpoints
    yields one run rather than several. Parsers are tried in the order given,
    and the first match wins.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no such path: {root}")

    found: list[tuple[Parser, Path]] = []
    if root.is_file():
        for parser in parsers:
            if parser.can_parse(root):
                return [(parser, root)]
        return []

    def walk(directory: Path) -> None:
        for parser in parsers:
            if parser.can_parse(directory):
                found.append((parser, directory))
                return
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                walk(child)

    walk(root)
    return found
