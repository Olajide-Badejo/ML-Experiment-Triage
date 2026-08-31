"""Read TensorBoard event files through the official event accumulator.

Scope is scalars only, and that is a decision rather than an omission.
Histograms, images, audio and graph definitions carry no information this tool
can act on: every statistic here operates on a tagged scalar step series, so
loading the rest would cost memory and parse time to no end. A run that
contains only histograms parses to an experiment with no metrics, which the
ingest reports rather than hides.

Reading is delegated to `tensorboard.backend.event_processing` rather than
decoding the record protocol here. The record framing has changed before, and
the accumulator is the reference implementation that tracks it.

A run is claimed only when it holds a file named with the full
`events.out.tfevents.` prefix whose first bytes frame as a TFRecord. Both
halves matter: this parser runs first, so claiming a directory on the strength
of the substring `tfevents` alone hid a real `metrics.csv` sitting beside a
file called `notes_tfevents.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from triage._extras import require
from triage.core.experiment import Experiment, MetricSeries
from triage.parsers.base import ParseError, Parser, non_finite_metadata

#: TensorBoard names every event file `events.out.tfevents.<time>.<host>...`.
#: Matching the full prefix rather than the substring `tfevents` is the whole
#: of defect D21a: `notes_tfevents.md` used to claim the run, and because this
#: parser is first in DEFAULT_PARSERS, a real `metrics.csv` sitting beside that
#: note was never read.
EVENT_FILE_PREFIX = "events.out.tfevents."

#: A TFRecord frames each record as an 8 byte little endian payload length, a
#: 4 byte masked CRC of those 8 bytes, the payload, and a 4 byte payload CRC.
#: The length alone is a strong enough check to reject a text file that merely
#: carries the right name: an arbitrary 8 bytes of prose declares a payload far
#: larger than the file. The CRCs are left to the accumulator, which is the
#: reference implementation and already verifies them.
RECORD_HEADER_BYTES = 12
RECORD_FOOTER_BYTES = 4

# 0 means keep every scalar rather than the accumulator's sampling default,
# which silently downsamples long runs and would corrupt a window statistic.
SIZE_GUIDANCE = {"scalars": 0, "histograms": 0, "images": 0, "audio": 0, "tensors": 0}


def _event_files(directory: Path) -> list[Path]:
    """Every readable event file in `directory`, matched without regard to case.

    Name matching is case insensitive so that a run reads the same way on
    Windows, where the filesystem ignores case, and on Linux, where it does
    not; a sweep that ingests on one and not the other is the worse of the two
    outcomes whichever way it falls.
    """
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if _is_event_file(path))


def _is_event_file(path: Path) -> bool:
    """True when `path` is named like an event file AND frames like one.

    The name is necessary but not sufficient. A file that merely carries the
    prefix, whether by accident or because someone copied a log next to it,
    must not claim the run: this parser runs first, and a false claim hides
    every other format in the same directory.
    """
    if not path.name.lower().startswith(EVENT_FILE_PREFIX) or not path.is_file():
        return False
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(RECORD_HEADER_BYTES)
    except OSError:
        return False
    if len(header) < RECORD_HEADER_BYTES:
        return False
    payload = int.from_bytes(header[:8], "little")
    return 0 < payload <= size - RECORD_HEADER_BYTES - RECORD_FOOTER_BYTES


class TensorBoardParser(Parser):
    """Parses a directory of TensorBoard event files into one experiment."""

    format_name = "tensorboard"

    def can_parse(self, path: Path) -> bool:
        if path.is_dir():
            return bool(_event_files(path))
        return _is_event_file(path)

    def parse(self, path: Path, root: Path | None = None) -> Experiment:
        # Imported here, and through `require`, so that a core install can be
        # offered this parser during discovery and only meets tensorboard if it
        # actually points at an event directory (E1).
        accumulator_module = require(
            "tensorboard.backend.event_processing.event_accumulator",
            extra="parsers",
            purpose="reading TensorBoard event files",
        )
        event_accumulator = accumulator_module.EventAccumulator

        directory = path if path.is_dir() else path.parent
        event_files = [p.name for p in _event_files(directory)]
        if not event_files:
            raise ParseError(
                f"no readable TensorBoard event file in {directory}; a file named "
                f"{EVENT_FILE_PREFIX}... must begin with a valid TFRecord header"
            )
        accumulator = event_accumulator(str(directory), size_guidance=SIZE_GUIDANCE)
        try:
            accumulator.Reload()
        except Exception as error:
            raise ParseError(f"could not read event files in {directory}: {error}") from error

        metrics: dict[str, MetricSeries] = {}
        for tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            if not events:
                continue
            metrics[tag] = MetricSeries(
                tag=tag,
                steps=np.fromiter((event.step for event in events), dtype=np.int64),
                values=np.fromiter((event.value for event in events), dtype=np.float32),
                wall_times=np.fromiter((event.wall_time for event in events), dtype=np.float64),
            )

        return Experiment(
            # Identity is the run DIRECTORY, not the event file inside it: a
            # bare event file passed on its own belongs to the directory that
            # holds it, and every other event file there is part of the same run.
            run_id=self.run_id(directory, root),
            source_path=str(directory),
            source_format=self.format_name,
            config=self.config_for(directory),
            metrics=metrics,
            metadata={
                "event_files": event_files,
                "n_event_files": len(event_files),
                "scalar_tags": sorted(metrics),
                **non_finite_metadata(metrics),
            },
        )
