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

A directory is claimed as a run only when nothing beneath it is a run: a sweep
root that happens to hold a stray parseable file is a container, not a run.
See `discover_runs`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from triage.core.experiment import Experiment, MetricSeries

CONFIG_FILENAME = "config.json"

LOGGER = logging.getLogger("triage.parsers")


#: Widest step this tool accepts. int64 is the storage type, and a value beyond
#: this silently casts to INT64_MIN behind a numpy warning and then sorts to the
#: front of the series, so it is refused instead.
MAX_STEP = 2**62

#: Files at or below this size have their contents hashed into the fingerprint.
#: 8 MiB covers every CSV and JSONL log this tool has been pointed at and the
#: great majority of event files, while keeping the cost of the skip check well
#: under the cost of the parse it is there to avoid.
CONTENT_HASH_MAX_BYTES = 8 * 1024 * 1024

#: Read size for the content hash. One page sized buffer, reused.
_CONTENT_CHUNK_BYTES = 1024 * 1024


def _relative_key(path: Path, root: Path) -> str:
    """`path` relative to `root`, spelled with forward slashes.

    POSIX separators are used whatever the platform, so a sweep ingested on
    Windows and the same sweep ingested on Linux produce the same identity and
    the same fingerprint. A path that does not sit under `root` falls back to
    its resolved form, which cannot collide with a relative one.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _content_hash(path: Path) -> bytes:
    """SHA-256 of a file's bytes, read in chunks, or a marker if unreadable.

    An unreadable file must not raise here: `fingerprint` runs before the parse
    that would report the problem properly, and a permission error at this
    point should cost the run a skip, not the sweep a crash. The error is
    folded into the digest instead, so the run reparses and fails where it can
    be reported.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_CONTENT_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        return hashlib.sha256(f"unreadable:{error}".encode()).digest()
    return digest.digest()


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


def files_with_suffix(directory: Path, *suffixes: str) -> list[Path]:
    """Files in `directory` whose extension matches, ignoring case.

    `Path.glob` is case sensitive on Linux and case insensitive on Windows, so
    a run holding `METRICS.CSV` parsed here and was invisible in CI. Matching
    on `suffix.lower()` makes the two platforms agree, which is the only
    behaviour worth having: the same sweep must ingest the same way wherever
    it is read. Results are sorted for a deterministic parse order.
    """
    if not directory.is_dir():
        return []
    wanted = {suffix.lower() for suffix in suffixes}
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in wanted
    )


def is_config_file(path: Path) -> bool:
    """True when `path` is the run's `config.json`, whatever its casing."""
    return path.name.lower() == CONFIG_FILENAME


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
        found = [p for p in files_with_suffix(directory, ".json") if is_config_file(p)]
        if not found:
            return {}
        config_path = found[0]
        try:
            loaded = json.loads(read_text(config_path))
        except json.JSONDecodeError as error:
            raise ParseError(f"{config_path} is not readable JSON: {error}") from error
        if not isinstance(loaded, dict):
            raise ParseError(f"{config_path} must hold a JSON object, got {type(loaded).__name__}")
        return loaded

    def fingerprint(self, path: Path, root: Path | None = None) -> str:
        """Hash identifying this run's source, for the ingest skip check.

        Two things go in, and the choice of each was a defect before it was a
        decision.

        **The run relative path, never the absolute one.** Hashing the resolved
        absolute path meant moving a sweep, or checking it out on another
        machine, changed every fingerprint and forced a full reparse of data
        that had not changed by a byte; it also made the database non portable,
        which is the opposite of what a cache is for. The path still has to
        take part, because `sweep_a/seed0` and `sweep_b/seed0` are two runs
        that may hold identical bytes, so what is hashed is the path relative
        to the ingest root. `root` defaults to the run's parent, which is the
        conservative choice for a caller that fingerprints a run on its own.

        **Content, for a file small enough to afford it.** Stat alone was
        defeated by anything that restores modification times, which is to say
        by rsync, by git checkout, and by every archive tool with a flag for
        it: the content changed, the stat did not, and the run was skipped as
        unchanged. Files at or under `CONTENT_HASH_MAX_BYTES` are therefore
        read and hashed. Larger ones keep the stat only check, because hashing
        a directory of event files would cost about what parsing costs and
        leave the skip check buying nothing.
        """
        base = Path(root) if root is not None else path.parent
        digest = hashlib.sha256()
        digest.update(_relative_key(path, base).encode("utf-8"))
        targets = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for target in targets:
            stat = target.stat()
            digest.update(_relative_key(target, base).encode("utf-8"))
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
            if stat.st_size <= CONTENT_HASH_MAX_BYTES:
                digest.update(_content_hash(target))
        return digest.hexdigest()


def _claiming_parser(directory: Path, parsers: list[Parser]) -> Parser | None:
    """The first parser that recognises `directory`, or None."""
    for parser in parsers:
        if parser.can_parse(directory):
            return parser
    return None


def _has_parseable_subdirectory(directory: Path, parsers: list[Parser]) -> bool:
    """True when any subdirectory below `directory` is itself a run.

    This is the whole of leaf claiming. The search goes all the way down rather
    than one level, because a sweep is often nested (`sweep/group/run`), and
    stopping at the first level would let a stray file at the sweep root claim
    the sweep exactly as before.
    """
    try:
        children = sorted(child for child in directory.iterdir() if child.is_dir())
    except OSError:
        return False
    for child in children:
        if _claiming_parser(child, parsers) is not None:
            return True
        if _has_parseable_subdirectory(child, parsers):
            return True
    return False


def discover_runs(root: Path, parsers: list[Parser]) -> list[tuple[Parser, Path]]:
    """Find every run under `root`, pairing each with the parser that claims it.

    **Leaf claiming.** A directory becomes a run only when no directory beneath
    it is itself parseable. The walk used to claim the first directory a parser
    recognised and stop there, which meant one stray `index.csv` written at a
    sweep root claimed the sweep: both real runs under it were never looked at,
    and the tool exited 0 having ingested one row. Descending past a claimable
    parent costs a little discovery time and is the only rule under which a
    nested TensorBoard layout (`runX/train`, `runX/val`) and a run directory
    holding a `checkpoints/` subdirectory both come out right.

    Parsers are tried in the order given and the first match wins. Every claim
    and every descend decision is logged at debug level, so `--log-level debug`
    answers "why was this directory not a run" without a rebuild.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no such path: {root}")

    found: list[tuple[Parser, Path]] = []
    if root.is_file():
        parser = _claiming_parser(root, parsers)
        if parser is not None:
            LOGGER.debug("claim %s as a run (%s, single file)", root, parser.format_name)
            return [(parser, root)]
        LOGGER.debug("skip %s: no parser claims it", root)
        return []

    def walk(directory: Path) -> None:
        parser = _claiming_parser(directory, parsers)
        if parser is not None and not _has_parseable_subdirectory(directory, parsers):
            LOGGER.debug("claim %s as a run (%s)", directory, parser.format_name)
            found.append((parser, directory))
            return
        if parser is not None:
            LOGGER.debug(
                "descend into %s: it parses as %s, but a subdirectory is a run too, so it is a "
                "container rather than a leaf",
                directory,
                parser.format_name,
            )
        else:
            LOGGER.debug("descend into %s: no parser claims it", directory)
        for child in sorted(directory.iterdir()):
            if child.is_dir():
                walk(child)

    walk(root)
    return found
