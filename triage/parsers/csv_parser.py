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

**Steps must be integers.** The step axis indexes the series, so two points
cannot share one index and a fractional value is not a step. `epoch` is an
accepted step column name because most logs write whole epochs, but a column
of fractional epochs (0.00, 0.25, 0.50, ...) stops the parse with a message
naming the file and the column rather than truncating twelve points into three
duplicates. Log an integer step beside the fractional value, or scale it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from triage.core.experiment import Experiment, MetricSeries
from triage.parsers.base import (
    MAX_STEP,
    ParseError,
    Parser,
    non_finite_metadata,
    read_text,
    validate_step,
)

STEP_COLUMNS = ("step", "global_step", "iteration", "iter", "epoch")
TAG_COLUMNS = ("tag", "metric", "name", "key")
VALUE_COLUMNS = ("value", "val")
WALL_COLUMNS = ("wall_time", "wall", "timestamp", "time")

# Spellings that mean "this point was measured and the measurement is not a
# finite number", as opposed to an empty cell, which means the point is absent.
# The distinction matters: the first is counted as a dropped non finite point
# and reported, the second is simply not part of the series. pandas collapses
# both to NaN by default, so the file is read with NA detection off and these
# are recognised here instead.
NON_FINITE_SPELLINGS = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)


def _first_match(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _validated_steps(column: pd.Series, source: Path, name: str) -> pd.Series:
    """Coerce the step column to numbers, refusing anything that is not a step.

    A cell that is not a number at all is left as NaN and its row is dropped
    later, because a log with a blank step on one line is still usable. A cell
    that IS a number but not an integral, in range one is a different matter:
    it would truncate into a duplicate of another step and silently destroy the
    series, so it stops the parse with a message naming the file and column.
    """
    numeric = pd.to_numeric(column, errors="coerce")
    present = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(present)
    bad = finite & ((present != np.trunc(present)) | (np.abs(present) > MAX_STEP))
    infinite = ~finite & ~np.isnan(present)
    offenders = np.flatnonzero(bad | infinite)
    if offenders.size:
        # Re raise through the shared validator so the message, the remedy and
        # the wording are identical to the JSON parser's.
        validate_step(float(present[offenders[0]]), f"{source.name} in {source.parent}", name)
    return numeric


def _numeric(column: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Convert a raw column to floats, and say which cells held a measurement.

    Returns the numeric column and a boolean mask that is True for a cell that
    parsed to a number, finite or not. A cell that is blank, or holds text that
    is not a number, is False: that point was never measured.
    """
    numeric = pd.to_numeric(column, errors="coerce")
    text = column.astype(str).str.strip().str.lower()
    measured = numeric.notna().to_numpy() | text.isin(NON_FINITE_SPELLINGS).to_numpy()
    return numeric, measured


class _Points:
    """Points for one tag, accumulated across every CSV file in a run.

    Wall times survive only when every contributing file carried them; a run
    that logged them in one file and not another has no consistent time axis,
    and inventing one would be worse than having none.
    """

    def __init__(self) -> None:
        self.steps: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.walls: list[np.ndarray | None] = []

    def add(
        self, steps: pd.Series, values: pd.Series, walls: pd.Series | None, keep: np.ndarray
    ) -> None:
        self.steps.append(steps.to_numpy()[keep].astype(np.int64))
        self.values.append(values.to_numpy()[keep].astype(np.float32))
        self.walls.append(None if walls is None else walls.to_numpy()[keep].astype(np.float64))

    def to_series(self, tag: str) -> MetricSeries:
        wall_times = (
            np.concatenate([w for w in self.walls if w is not None])
            if all(w is not None for w in self.walls)
            else None
        )
        return MetricSeries(
            tag=tag,
            steps=np.concatenate(self.steps),
            values=np.concatenate(self.values),
            wall_times=wall_times,
        )


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

        # Points accumulate across every file in the run directory before any
        # series is built. A run split into `part1.csv` and `part2.csv` is one
        # series of five points, not the second file's two: the previous code
        # did `metrics.update(...)` per file, so the later file replaced the
        # earlier one under the same tag and 60% of the data vanished. The
        # JSONL parser has always concatenated, so this also makes the two
        # parsers agree on the same logical input.
        collected: dict[str, _Points] = {}
        shapes: dict[str, str] = {}
        for csv_path in files:
            # Decoding is done here rather than inside pandas so that a non
            # UTF-8 file raises ParseError naming the byte offset, which is the
            # documented contract, instead of a bare UnicodeDecodeError that no
            # caller catches.
            try:
                frame = pd.read_csv(
                    io.StringIO(read_text(csv_path)), keep_default_na=False, na_values=[]
                )
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                raise ParseError(f"{csv_path} is not readable CSV: {error}") from error
            shapes[csv_path.name] = self._collect_frame(frame, csv_path, collected)

        # `MetricSeries` sorts by step and keeps the later write for a repeated
        # step, so building each tag once here is all the ordering needed.
        metrics = {tag: points.to_series(tag) for tag, points in sorted(collected.items())}

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
                **non_finite_metadata(metrics),
            },
        )

    def _collect_frame(
        self, frame: pd.DataFrame, source: Path, collected: dict[str, _Points]
    ) -> str:
        """Add one file's points to the run wide accumulator, return its shape."""
        columns = [str(name) for name in frame.columns]
        step_column = _first_match(columns, STEP_COLUMNS)
        if step_column is None:
            raise ParseError(
                f"{source} has no step column; expected one of {', '.join(STEP_COLUMNS)}"
            )
        tag_column = _first_match(columns, TAG_COLUMNS)
        value_column = _first_match(columns, VALUE_COLUMNS)
        wall_column = _first_match(columns, WALL_COLUMNS)

        steps = _validated_steps(frame[step_column], source, step_column)
        walls = pd.to_numeric(frame[wall_column], errors="coerce") if wall_column else None

        if tag_column is not None and value_column is not None:
            self._collect_long(frame, steps, walls, tag_column, value_column, collected)
            return "long"
        self._collect_wide(frame, steps, walls, {step_column, wall_column or ""}, collected)
        return "wide"

    def _collect_long(
        self,
        frame: pd.DataFrame,
        steps: pd.Series,
        walls: pd.Series | None,
        tag_column: str,
        value_column: str,
        collected: dict[str, _Points],
    ) -> None:
        # A cell that is not a number at all (blank, or text such as "n/a") is
        # an absent point and never reaches the series. A cell that spells a
        # non finite number is a poisoned point: it is passed through so that
        # `MetricSeries` drops it and counts it, which is what makes the three
        # formats agree on the same poisoned run.
        values, measured = _numeric(frame[value_column])
        for tag, index in frame.groupby(frame[tag_column].astype(str)).groups.items():
            rows = frame.index.isin(index)
            keep = rows & steps.notna().to_numpy() & measured
            if not keep.any():
                continue
            collected.setdefault(str(tag), _Points()).add(steps, values, walls, keep)

    def _collect_wide(
        self,
        frame: pd.DataFrame,
        steps: pd.Series,
        walls: pd.Series | None,
        skip: set[str],
        collected: dict[str, _Points],
    ) -> None:
        for name in frame.columns:
            if str(name) in skip:
                continue
            values, measured = _numeric(frame[name])
            keep = steps.notna().to_numpy() & measured
            if not keep.any():
                continue  # a text column such as a note or a checkpoint path
            collected.setdefault(str(name), _Points()).add(steps, values, walls, keep)
