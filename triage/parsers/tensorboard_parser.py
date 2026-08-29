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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from triage.core.experiment import Experiment, MetricSeries
from triage.parsers.base import ParseError, Parser, non_finite_metadata

EVENT_FILE_GLOB = "*tfevents*"

# 0 means keep every scalar rather than the accumulator's sampling default,
# which silently downsamples long runs and would corrupt a window statistic.
SIZE_GUIDANCE = {"scalars": 0, "histograms": 0, "images": 0, "audio": 0, "tensors": 0}


class TensorBoardParser(Parser):
    """Parses a directory of TensorBoard event files into one experiment."""

    format_name = "tensorboard"

    def can_parse(self, path: Path) -> bool:
        if path.is_dir():
            return any(path.glob(EVENT_FILE_GLOB))
        return EVENT_FILE_GLOB.strip("*") in path.name

    def parse(self, path: Path) -> Experiment:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        directory = path if path.is_dir() else path.parent
        accumulator = EventAccumulator(str(directory), size_guidance=SIZE_GUIDANCE)
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

        event_files = sorted(p.name for p in directory.glob(EVENT_FILE_GLOB))
        return Experiment(
            run_id=self.run_id(directory),
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
