"""Read step free JSONL evaluation output into `Outcomes`.

A training log is indexed by step. An evaluation output is not: it is one row
per classified field, per example, per configuration, and asking for its final
window is a category error rather than a hard question. The JSONL parser said so
truthfully and unhelpfully, by raising `ParseError` on a file it had already
claimed; this parser is the path that file should have had.

**Two levels of recognition, and only one of them is automatic.**

*Strict*, which is what `can_parse` does inside `DEFAULT_PARSERS`: the file
carries no step key on any record AND every record declares a `schema_version`.
That pair of conditions is narrow on purpose. Claiming any step free JSONL by
default would mean claiming a file of configuration, a file of notes, or a log
whose step column this tool simply failed to recognise, and turning a legible
`ParseError` into a table of nonsense is a much worse failure than refusing.
A `schema_version` is a producer saying "this is a record format I am declaring",
which is exactly the signal the consumer's own schema leads with.

*Generic*, requested with `triage ingest --outcomes`: the `schema_version`
condition is dropped and any step free JSONL of flat records is read. That is an
explicit instruction from somebody who knows what the file is, which is the only
basis on which guessing is reasonable.

**Fields and groups are separated by what the values are**, not by a list of
names this file would have to keep in step with two other repositories. A column
whose values are numbers or booleans in every row that has one is a measurement;
anything else is a categorical group key. A column that is numeric in some rows
and text in others is a group key, because a statistic over it would be a
statistic over a column that does not mean one thing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triage.core.outcomes import Outcomes
from triage.parsers.base import (
    ParseError,
    Parser,
    files_with_suffix,
    is_config_file,
    read_text,
)
from triage.parsers.csv_parser import STEP_COLUMNS

#: The key a producer uses to declare that its rows follow a record schema. Its
#: presence is what makes strict recognition safe: see the module docstring.
SCHEMA_KEYS = ("schema_version", "row_schema_version")

#: How many records `can_parse` reads before deciding. Enough to get past a
#: header block and see real rows; small enough that discovery over a large
#: sweep does not turn into a full parse of every file in it.
PROBE_RECORDS = 50


def _has_step(record: dict[str, Any]) -> bool:
    lowered = {str(key).lower() for key in record}
    return any(candidate in lowered for candidate in STEP_COLUMNS)


def _is_measurement(value: Any) -> bool:
    """True when this value is a number to compute over rather than a label.

    `bool` passes because it is a subclass of `int` and because that is right:
    `correct: true` is a measurement whose mean is an accuracy.
    """
    return isinstance(value, (int, float))


class OutcomesParser(Parser):
    """Parses step free JSONL evaluation rows into an `Outcomes`."""

    format_name = "outcomes"

    def __init__(self, strict: bool = True) -> None:
        #: Strict requires a declared `schema_version`; see the module docstring.
        self.strict = strict

    def can_parse(self, path: Path) -> bool:
        files = self._log_files(path) if path.is_dir() else [path]
        if not files or any(file.suffix.lower() not in {".jsonl", ".json"} for file in files):
            return False
        records = []
        for file in files:
            records.extend(self._probe(file))
            if len(records) >= PROBE_RECORDS:
                break
        if not records:
            return False
        if any(_has_step(record) for record in records):
            return False
        if self.strict and not all(any(key in record for key in SCHEMA_KEYS) for record in records):
            return False
        # A row of outcomes has to hold at least one number to be worth reading
        # as one. A file of pure labels is a manifest, not a measurement.
        return any(_is_measurement(value) for record in records for value in record.values())

    @staticmethod
    def _log_files(directory: Path) -> list[Path]:
        return files_with_suffix(directory, ".jsonl") + [
            path for path in files_with_suffix(directory, ".json") if not is_config_file(path)
        ]

    @staticmethod
    def _probe(path: Path) -> list[dict[str, Any]]:
        """The first `PROBE_RECORDS` objects in a file, or none if unreadable.

        `can_parse` answers a question and never raises: a file it cannot read is
        a file it does not claim, and the parser that does claim it is the one
        that should report why it could not be read.
        """
        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                if path.suffix.lower() == ".json":
                    loaded = json.load(handle)
                    items = loaded if isinstance(loaded, list) else []
                    return [item for item in items if isinstance(item, dict)][:PROBE_RECORDS]
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
                    if len(records) >= PROBE_RECORDS:
                        break
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        return records

    def parse(self, path: Path, root: Path | None = None) -> Outcomes:
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
            raise ParseError(f"{path} contained no usable outcome rows")

        stepped = sum(1 for record in records if _has_step(record))
        if stepped:
            raise ParseError(
                f"{path}: {stepped} of {len(records)} record(s) carry a step field, so this is a "
                f"time series rather than a set of outcomes. Let the JSONL parser read it: an "
                f"outcomes file has no step axis, and a file with one has no rows to pair"
            )

        fields, groups = self._columns(records)
        return Outcomes(
            run_id=self.run_id(path, root),
            source_path=str(path),
            source_format=self.format_name,
            fields=fields,
            groups=groups,
            config=self.config_for(directory),
            metadata={
                "json_files": [file.name for file in files],
                "rows": len(records),
                "malformed_lines": malformed,
                "outcome_fields": sorted(fields),
                "group_keys": sorted(groups),
                "strict_schema": self.strict,
            },
        )

    def _read(self, source: Path) -> tuple[list[dict[str, Any]], int]:
        """Every object in one file, counting the lines that were not objects."""
        text = read_text(source)
        if source.suffix.lower() == ".json":
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as error:
                raise ParseError(f"{source} is not valid JSON: {error}") from error
            if not isinstance(loaded, list):
                raise ParseError(f"{source} holds no list of outcome rows")
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
                # Counted, not fatal, exactly as in the JSONL parser: a
                # truncated last line is what a killed job leaves behind.
                malformed += 1
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                malformed += 1
        return records, malformed

    @staticmethod
    def _columns(
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, list[float]], dict[str, list[Any]]]:
        """Sort every key into a measurement or a group key, over all rows.

        Over ALL rows rather than the first one. Inferring a schema from
        `records[0]` is the defect E6 documents in the JSONL parser, and it is
        worse here: a confidence that happens to be `null` in the first row would
        make the whole column categorical, and an accuracy would silently become
        a set of labels.

        A key missing from a row is a hole, not a zero. A missing measurement
        becomes NaN, which every statistic in this package already refuses to
        compute over silently; a missing group key becomes the empty string,
        which is a value and never merges with a real one.
        """
        names: list[str] = []
        for record in records:
            for key in record:
                if str(key) not in names:
                    names.append(str(key))

        numeric: dict[str, bool] = {}
        for name in names:
            seen = [record[name] for record in records if record.get(name) is not None]
            numeric[name] = bool(seen) and all(_is_measurement(value) for value in seen)

        fields: dict[str, list[float]] = {}
        groups: dict[str, list[Any]] = {}
        for name in names:
            if numeric[name]:
                fields[name] = [
                    float(record[name]) if record.get(name) is not None else float("nan")
                    for record in records
                ]
            else:
                groups[name] = [record.get(name) for record in records]
        return fields, groups
