"""Read CSV training logs in either of the two shapes people actually write.

**Wide.** One row per step, one column per metric. This is what a hand rolled
`logger.log_csv(...)` produces and what pandas writes by default:

    step,train/loss,val/loss,val/acc
    0,2.30,2.29,0.11

**Long.** One row per observation, with the metric name in a column. This is
what a framework callback writes when the set of metrics is not known up front:

    step,tag,value
    0,train/loss,2.30
    0,val/loss,2.29

The shape is detected from the header rather than configured, because getting
it wrong is loud (a metric named `value` with steps repeated) and asking the
user to declare it would be one more thing to get wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from triage.core.experiment import Experiment, MetricSeries
from triage.parsers.base import ParseError, Parser

STEP_COLUMNS = ("step", "global_step", "iteration", "iter", "epoch")
TAG_COLUMNS = ("tag", "metric", "name", "key")
VALUE_COLUMNS = ("value", "val")
WALL_COLUMNS = ("wall_time", "wall", "timestamp", "time")


def _first_match(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


class CsvParser(Parser):
    """Parses one or more CSV files into a single experiment."""

    format_name = "csv"

    def can_parse(self, path: Path) -> bool:
        if path.is_dir():
            return any(path.glob("*.csv"))
        return path.suffix.lower() == ".csv"

    def parse(self, path: Path) -> Experiment:
        directory = path if path.is_dir() else path.parent
        files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        if not files:
            raise ParseError(f"no CSV files under {path}")

        metrics: dict[str, MetricSeries] = {}
        shapes: dict[str, str] = {}
        for csv_path in files:
            try:
                frame = pd.read_csv(csv_path)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                raise ParseError(f"{csv_path} is not readable CSV: {error}") from error
            parsed, shape = self._parse_frame(frame, csv_path)
            shapes[csv_path.name] = shape
            metrics.update(parsed)

        return Experiment(
            run_id=self.run_id(path),
            source_path=str(path),
            source_format=self.format_name,
            config=self.config_for(directory),
            metrics=metrics,
            metadata={
                "csv_files": [p.name for p in files],
                "csv_shapes": shapes,
                "scalar_tags": sorted(metrics),
            },
        )

    def _parse_frame(
        self, frame: pd.DataFrame, source: Path
    ) -> tuple[dict[str, MetricSeries], str]:
        columns = [str(name) for name in frame.columns]
        step_column = _first_match(columns, STEP_COLUMNS)
        if step_column is None:
            raise ParseError(
                f"{source} has no step column; expected one of {', '.join(STEP_COLUMNS)}"
            )
        tag_column = _first_match(columns, TAG_COLUMNS)
        value_column = _first_match(columns, VALUE_COLUMNS)
        wall_column = _first_match(columns, WALL_COLUMNS)

        steps = pd.to_numeric(frame[step_column], errors="coerce")
        walls = pd.to_numeric(frame[wall_column], errors="coerce") if wall_column else None

        if tag_column is not None and value_column is not None:
            return self._parse_long(frame, steps, walls, tag_column, value_column), "long"
        return self._parse_wide(frame, steps, walls, {step_column, wall_column or ""}), "wide"

    def _parse_long(
        self,
        frame: pd.DataFrame,
        steps: pd.Series,
        walls: pd.Series | None,
        tag_column: str,
        value_column: str,
    ) -> dict[str, MetricSeries]:
        values = pd.to_numeric(frame[value_column], errors="coerce")
        metrics: dict[str, MetricSeries] = {}
        for tag, index in frame.groupby(frame[tag_column].astype(str)).groups.items():
            rows = frame.index.isin(index)
            keep = rows & steps.notna().to_numpy() & values.notna().to_numpy()
            if not keep.any():
                continue
            metrics[str(tag)] = MetricSeries(
                tag=str(tag),
                steps=steps.to_numpy()[keep].astype(np.int64),
                values=values.to_numpy()[keep].astype(np.float32),
                wall_times=None if walls is None else walls.to_numpy()[keep].astype(np.float64),
            )
        return metrics

    def _parse_wide(
        self,
        frame: pd.DataFrame,
        steps: pd.Series,
        walls: pd.Series | None,
        skip: set[str],
    ) -> dict[str, MetricSeries]:
        metrics: dict[str, MetricSeries] = {}
        for name in frame.columns:
            if str(name) in skip:
                continue
            values = pd.to_numeric(frame[name], errors="coerce")
            keep = (steps.notna() & values.notna()).to_numpy()
            if not keep.any():
                continue  # a text column such as a note or a checkpoint path
            metrics[str(name)] = MetricSeries(
                tag=str(name),
                steps=steps.to_numpy()[keep].astype(np.int64),
                values=values.to_numpy()[keep].astype(np.float32),
                wall_times=None if walls is None else walls.to_numpy()[keep].astype(np.float64),
            )
        return metrics
