"""Read JSONL and JSON training logs.

JSONL is the format that appears when someone appends a dictionary per logging
call, which is the most common thing a training loop does without a framework.
Both record shapes are accepted, mirroring the CSV parser:

    {"step": 0, "train/loss": 2.30, "val/acc": 0.11}
    {"step": 0, "tag": "train/loss", "value": 2.30}

A `.json` file is accepted too, holding either a list of such records or an
object with a `metrics` or `history` list. Malformed lines are counted and
reported in the experiment metadata rather than silently dropped or allowed to
abort the parse: a truncated final line is what a killed training job leaves
behind, and losing the other ten thousand records over it would be wrong.

**Steps must be integers**, as in the CSV parser and for the same reason: the
step axis indexes the series. A record whose step is fractional, non finite or
out of int64 range stops the parse with a message naming the file and the key.
A `null` step, and any record that carries no step key at all, is skipped and
counted instead, because a header line is a legitimate part of these files.

File matching is on `suffix.lower()`, and `config.json` is recognised by a
case insensitive name, so a run reads the same way on Linux and on Windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from triage.core.experiment import Experiment, MetricSeries
from triage.parsers.base import (
    ParseError,
    Parser,
    files_with_suffix,
    is_config_file,
    non_finite_metadata,
    read_text,
    validate_step,
)
from triage.parsers.csv_parser import STEP_COLUMNS, TAG_COLUMNS, VALUE_COLUMNS, WALL_COLUMNS

RECORD_KEYS = ("metrics", "history", "records", "logs")

#: JSON has no non finite literals, so producers that need them write strings.
#: The PyTorch Performance and Health Toolkit writes exactly these three in its
#: schema v2 logs. They are read as the numbers they name, at which point the
#: non finite filter in `MetricSeries` drops and counts them like any other
#: NaN: the point was measured, and the measurement is unusable.
NON_FINITE_LITERALS = {
    "NaN": float("nan"),
    "Infinity": float("inf"),
    "-Infinity": float("-inf"),
}


def _pick(record: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(key).lower(): str(key) for key in record}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _as_float(raw: Any) -> float | None:
    """The numeric value of a logged field, or None when there is no point.

    `null` returns None, and that is the contract: a producer writing null
    means "not measured at this step", never zero, so the point is absent from
    the series rather than dragging a fabricated zero into a window mean.
    Booleans are also None: `True` is a flag, not a metric. Any other string is
    text (a checkpoint path, a note) and is not a measurement either.
    """
    if isinstance(raw, str):
        return NON_FINITE_LITERALS.get(raw)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


class JsonlParser(Parser):
    """Parses JSONL or JSON training logs into a single experiment."""

    format_name = "jsonl"

    def can_parse(self, path: Path) -> bool:
        if path.is_dir():
            return bool(self._log_files(path))
        if path.suffix.lower() == ".jsonl":
            return True
        return path.suffix.lower() == ".json" and not is_config_file(path)

    @staticmethod
    def _log_files(directory: Path) -> list[Path]:
        """Every log file in a run directory, JSONL first, config.json excluded."""
        return files_with_suffix(directory, ".jsonl") + [
            p for p in files_with_suffix(directory, ".json") if not is_config_file(p)
        ]

    def parse(self, path: Path) -> Experiment:
        directory = path if path.is_dir() else path.parent
        files = self._log_files(path) if path.is_dir() else [path]
        if not files:
            raise ParseError(f"no JSON or JSONL files under {path}")

        records: list[dict[str, Any]] = []
        malformed = 0
        for source in files:
            new_records, bad = self._read(source)
            records.extend(new_records)
            malformed += bad

        if not records:
            raise ParseError(f"{path} contained no usable records")

        metrics, counts = self._to_metrics(records, ", ".join(p.name for p in files))
        return Experiment(
            run_id=self.run_id(path),
            source_path=str(path),
            source_format=self.format_name,
            config=self.config_for(directory),
            metrics=metrics,
            metadata={
                "json_files": [p.name for p in files],
                "records": len(records),
                "malformed_lines": malformed,
                "scalar_tags": sorted(metrics),
                **counts,
                **non_finite_metadata(metrics),
            },
        )

    def _read(self, source: Path) -> tuple[list[dict[str, Any]], int]:
        text = read_text(source)

        if source.suffix.lower() == ".json":
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as error:
                raise ParseError(f"{source} is not valid JSON: {error}") from error
            if isinstance(loaded, dict):
                for key in RECORD_KEYS:
                    if isinstance(loaded.get(key), list):
                        loaded = loaded[key]
                        break
            if not isinstance(loaded, list):
                raise ParseError(f"{source} holds no list of records")
            return [item for item in loaded if isinstance(item, dict)], 0

        records: list[dict[str, Any]] = []
        malformed = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                malformed += 1
        return records, malformed

    def _to_metrics(
        self, records: list[dict[str, Any]], source_label: str
    ) -> tuple[dict[str, MetricSeries], dict[str, int]]:
        """Turn records into series, and count what could not become a point.

        Schema inference scans for the FIRST record that carries a step key
        rather than trusting `records[0]`. That single change is what makes a
        second producer's logs readable: the PyTorch Performance and Health
        Toolkit opens every file with an environment header line, and a resumed
        sweep writes several of them, so the first record routinely carries no
        step at all. A step free record is skipped and counted here rather than
        aborting the parse, because those header lines are legitimate content.
        """
        schema_record: dict[str, Any] = {}
        step_key: str | None = None
        for record in records:
            step_key = _pick(record, STEP_COLUMNS)
            if step_key is not None:
                schema_record = record
                break
        if step_key is None:
            raise ParseError(
                f"{source_label}: no record carries a step field; expected one of "
                f"{', '.join(STEP_COLUMNS)}"
            )
        tag_key = _pick(schema_record, TAG_COLUMNS)
        value_key = _pick(schema_record, VALUE_COLUMNS)
        wall_key = _pick(schema_record, WALL_COLUMNS)
        long_form = tag_key is not None and value_key is not None

        counts = {"records_without_step": 0, "records_without_tag": 0}
        collected: dict[str, list[tuple[int, float, float | None]]] = {}
        for record in records:
            if _pick(record, STEP_COLUMNS) is None or record.get(step_key) is None:
                # A header line, or a record whose step is explicitly null.
                counts["records_without_step"] += 1
                continue
            step = validate_step(record[step_key], source_label, step_key)
            wall = record.get(wall_key) if wall_key else None
            wall_value = float(wall) if isinstance(wall, (int, float)) else None

            if long_form:
                value = _as_float(record.get(value_key))
                if value is None:
                    continue
                raw_tag = record.get(tag_key)
                if raw_tag is None:
                    # Without a tag there is nothing to call the series, and
                    # naming it `None` (which is what used to happen) invents a
                    # metric out of a logging bug.
                    counts["records_without_tag"] += 1
                    continue
                collected.setdefault(str(raw_tag), []).append((step, value, wall_value))
                continue

            for key, raw in record.items():
                if str(key) in {step_key, wall_key}:
                    continue
                value = _as_float(raw)
                if value is not None:
                    collected.setdefault(str(key), []).append((step, value, wall_value))

        metrics: dict[str, MetricSeries] = {}
        for tag, points in collected.items():
            steps = np.fromiter((p[0] for p in points), dtype=np.int64, count=len(points))
            values = np.fromiter((p[1] for p in points), dtype=np.float32, count=len(points))
            walls = [p[2] for p in points]
            wall_times = (
                np.asarray(walls, dtype=np.float64) if all(w is not None for w in walls) else None
            )
            metrics[tag] = MetricSeries(tag=tag, steps=steps, values=values, wall_times=wall_times)
        return metrics, counts
