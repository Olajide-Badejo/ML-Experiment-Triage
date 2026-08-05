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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from triage.core.experiment import Experiment, MetricSeries
from triage.parsers.base import ParseError, Parser
from triage.parsers.csv_parser import STEP_COLUMNS, TAG_COLUMNS, VALUE_COLUMNS, WALL_COLUMNS

RECORD_KEYS = ("metrics", "history", "records", "logs")


def _pick(record: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(key).lower(): str(key) for key in record}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


class JsonlParser(Parser):
    """Parses JSONL or JSON training logs into a single experiment."""

    format_name = "jsonl"

    def can_parse(self, path: Path) -> bool:
        if path.is_dir():
            return any(path.glob("*.jsonl")) or any(
                p.name != "config.json" for p in path.glob("*.json")
            )
        if path.suffix.lower() == ".jsonl":
            return True
        return path.suffix.lower() == ".json" and path.name != "config.json"

    def parse(self, path: Path) -> Experiment:
        directory = path if path.is_dir() else path.parent
        if path.is_dir():
            files = sorted(path.glob("*.jsonl")) + sorted(
                p for p in path.glob("*.json") if p.name != "config.json"
            )
        else:
            files = [path]
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

        metrics = self._to_metrics(records)
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
            },
        )

    def _read(self, source: Path) -> tuple[list[dict[str, Any]], int]:
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as error:
            raise ParseError(f"{source} could not be read: {error}") from error

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

    def _to_metrics(self, records: list[dict[str, Any]]) -> dict[str, MetricSeries]:
        step_key = _pick(records[0], STEP_COLUMNS)
        if step_key is None:
            raise ParseError(
                f"records carry no step field; expected one of {', '.join(STEP_COLUMNS)}"
            )
        tag_key = _pick(records[0], TAG_COLUMNS)
        value_key = _pick(records[0], VALUE_COLUMNS)
        wall_key = _pick(records[0], WALL_COLUMNS)
        long_form = tag_key is not None and value_key is not None

        collected: dict[str, list[tuple[int, float, float | None]]] = {}
        for record in records:
            raw_step = record.get(step_key)
            if not isinstance(raw_step, (int, float)) or isinstance(raw_step, bool):
                continue
            step = int(raw_step)
            wall = record.get(wall_key) if wall_key else None
            wall_value = float(wall) if isinstance(wall, (int, float)) else None

            if long_form:
                value = record.get(value_key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    tag = str(record.get(tag_key))
                    collected.setdefault(tag, []).append((step, float(value), wall_value))
                continue

            for key, value in record.items():
                if str(key) in {step_key, wall_key}:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                collected.setdefault(str(key), []).append((step, float(value), wall_value))

        metrics: dict[str, MetricSeries] = {}
        for tag, points in collected.items():
            steps = np.fromiter((p[0] for p in points), dtype=np.int64, count=len(points))
            values = np.fromiter((p[1] for p in points), dtype=np.float32, count=len(points))
            walls = [p[2] for p in points]
            wall_times = (
                np.asarray(walls, dtype=np.float64) if all(w is not None for w in walls) else None
            )
            metrics[tag] = MetricSeries(tag=tag, steps=steps, values=values, wall_times=wall_times)
        return metrics
