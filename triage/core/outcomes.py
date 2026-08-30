"""Cross sectional results: one row per scored unit, and no step axis at all.

`Experiment` is a training run: tagged series indexed by step. A great deal of
real evaluation output is not that shape. An offline evaluation writes one row
per classified field, per example, per configuration, and the rows are not
ordered by anything; asking for "the final window" of such a file is a category
error, not a hard question.

The consumer this model was built for writes exactly that, 25 keys to the row:

    {"schema_version": "1.0.0", "run_id": "...", "engine": "ngram",
     "engine_describe": {...}, "prompt_version": null,
     "corpus_manifest_sha": "...", "split": "test", "form_id": "...",
     "form_family": "address", "locale": "de-DE", "tier": "clean",
     "template_id": "...", "selector": "#address-plz",
     "true_label": "postal-code", "pred_label": "postal-code",
     "correct": true, "confidence": 0.91, "confidence_kind": "calibrated",
     "runner_up_label": "...", "runner_up_confidence": 0.04,
     "signals": [...], "latency_us": 412.0, "declared_token": "postal-code",
     "finding_codes": [], "extraction_warnings": []}

They handed that file to the JSONL parser as a probe and recorded the refusal in
their own analysis output as a checkable finding, which was the honest thing to
do with a tool that had no shape for it. `Outcomes` is the shape.

**One row is one FIELD, so the unit of analysis is the `(form_id, selector)`
pair.** A form holds many fields; `form_id` names the form. Pairing on it raises
rather than guessing, which is `pair_on` doing its job, and the composite key is
built as a group key first: see `pair_on` below.

**Two kinds of column, and the difference is not cosmetic.** A *field* is a
measurement: a float, or a boolean that is really a zero or a one. A *group* is a
categorical key: which engine, which split, which form template. Fields are what
a statistic is computed over; groups are what pairs are matched on and what
clusters are drawn from. Keeping them apart is what makes `pair_on` able to
refuse an ambiguous join instead of guessing at one.

**What this model deliberately does not have** is a step, a window, or an order
that means anything. Every windowed statistic in `triage.analysis.comparison`
refuses an `Outcomes` by construction and names `paired_permutation` instead,
because the right test for two conditions scored on the same units is a paired
one, and a window mean of an arbitrary row order is a number with no referent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from triage.core.experiment import SeriesError


def _as_field(name: str, values: Any) -> np.ndarray:
    """One outcome column as float64, with booleans read as zero and one.

    Booleans are folded here and nowhere else. `correct: true` is a measurement
    whose mean is an accuracy, so it has to arrive as a number for any statistic
    to run over it; a group key spelled `true` would be a different thing and
    lands in `groups` instead.
    """
    array = np.asarray(values)
    if array.dtype == np.bool_:
        return array.astype(np.float64)
    try:
        return np.asarray(array, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise SeriesError(
            f"outcome field {name!r} holds values that are not numbers ({error}); "
            f"a categorical column belongs in `groups`"
        ) from error


def _as_group(name: str, values: Any) -> np.ndarray:
    """One categorical column as an object array of strings.

    Strings rather than whatever came in, because a group key is used as a
    dictionary key and as a cluster label, and `32` and `32.0` arriving from two
    producers must not become two templates. `None` becomes the empty string,
    which is a value like any other and never silently merges with a real one.
    """
    return np.asarray(["" if item is None else str(item) for item in values], dtype=object)


@dataclass
class Outcomes:
    """One evaluation's rows: named measurements plus categorical group keys.

    Attributes:
        run_id: identity, taken the same way an `Experiment`'s is (D4).
        source_path: where it was read from.
        source_format: short name of the parser that read it.
        fields: `{name: float64 array}`, one value per row. Booleans arrive here
            as zero and one.
        groups: `{name: object array of str}`, one value per row.
        config: the run's configuration, from a `config.json` beside it.
        metadata: provenance and counts, as every parser records.
    """

    run_id: str
    source_path: str
    source_format: str
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fields = {name: _as_field(name, values) for name, values in self.fields.items()}
        groups = {name: _as_group(name, values) for name, values in self.groups.items()}
        lengths = {len(column) for column in (*fields.values(), *groups.values())}
        if len(lengths) > 1:
            spelled = ", ".join(
                f"{name} ({len(column)})"
                for name, column in sorted({**fields, **groups}.items())
                # sorted over a merged dict is fine: the two name spaces are
                # disjoint, which the check below enforces.
            )
            raise SeriesError(
                f"run {self.run_id!r}: every column must hold one value per row, got {spelled}"
            )
        collision = sorted(set(fields) & set(groups))
        if collision:
            raise SeriesError(
                f"run {self.run_id!r}: {', '.join(collision)} is both a measurement and a "
                f"group key; one name means one thing"
            )
        self.fields = fields
        self.groups = groups

    def __len__(self) -> int:
        return self.n_rows

    @property
    def n_rows(self) -> int:
        for column in (*self.fields.values(), *self.groups.values()):
            return int(column.size)
        return 0

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0

    @property
    def field_names(self) -> list[str]:
        return sorted(self.fields)

    @property
    def group_names(self) -> list[str]:
        return sorted(self.groups)

    def field(self, name: str) -> np.ndarray:
        """One measurement column. Raises `KeyError` naming what is available."""
        try:
            return self.fields[name]
        except KeyError as error:
            available = ", ".join(self.field_names) or "none"
            raise KeyError(
                f"run {self.run_id!r} has no outcome field {name!r}; available: {available}"
            ) from error

    def group(self, name: str) -> np.ndarray:
        """One categorical column. Raises `KeyError` naming what is available."""
        try:
            return self.groups[name]
        except KeyError as error:
            available = ", ".join(self.group_names) or "none"
            raise KeyError(
                f"run {self.run_id!r} has no group key {name!r}; available: {available}"
            ) from error

    def take(self, index: np.ndarray) -> Outcomes:
        """The same outcomes with rows reordered or restricted by `index`."""
        return Outcomes(
            run_id=self.run_id,
            source_path=self.source_path,
            source_format=self.source_format,
            fields={name: column[index] for name, column in self.fields.items()},
            groups={name: column[index] for name, column in self.groups.items()},
            config=dict(self.config),
            metadata=dict(self.metadata),
        )

    def select(self, **equals: str) -> Outcomes:
        """The rows whose group keys all equal the values given, in row order."""
        keep = np.ones(self.n_rows, dtype=bool)
        for name, wanted in equals.items():
            keep &= self.group(name) == str(wanted)
        return self.take(np.flatnonzero(keep))

    def pair_on(
        self, unit: str, condition: str, baseline: str, candidate: str
    ) -> tuple[Outcomes, Outcomes]:
        """Split into two row aligned halves, matched unit by unit.

        This is the join a paired test needs and the one a caller should not be
        writing by hand. Two conditions scored on the same units are only
        comparable pair by pair, and the pairing has to be established from a key
        rather than from the order the rows happen to sit in: a file written by
        two processes, or filtered by anything, is not in a matching order, and
        two vectors of the same length are not thereby paired.

        Refuses rather than guesses. A unit missing from one side, or appearing
        twice on either, is an error naming the units, because the alternatives
        are dropping evidence in silence or pairing a row with the wrong one.

        **`unit` is one key, and a real unit is often two columns.** One row per
        classified field under a `form_id` that names the form is not identified
        by the form: `pair_on("form_id", ...)` on such a file raises, correctly.
        Build the composite as a group key first, and pair on that:

            from dataclasses import replace

            forms = outcomes.group("form_id")
            selectors = outcomes.group("selector")
            field_id = [
                f"{form}|{selector}" for form, selector in zip(forms, selectors, strict=True)
            ]
            paired = replace(outcomes, groups={**outcomes.groups, "field_id": field_id})
            rules, ngram = paired.pair_on("field_id", "engine", "rules", "ngram")

        Returns `(baseline_rows, candidate_rows)` in a common unit order, ready
        for `paired_permutation` along with a cluster key taken from either.
        """
        self.group(unit)  # raises here, naming the available keys, if it is absent
        left = self.select(**{condition: baseline})
        right = self.select(**{condition: candidate})
        if left.is_empty or right.is_empty:
            seen = ", ".join(sorted(set(self.group(condition)))) or "none"
            raise SeriesError(
                f"run {self.run_id!r}: {condition} = {baseline!r} has {left.n_rows} row(s) and "
                f"{condition} = {candidate!r} has {right.n_rows}; values present: {seen}"
            )
        left_index = _unit_index(left.group(unit), unit, baseline, self.run_id)
        right_index = _unit_index(right.group(unit), unit, candidate, self.run_id)
        shared = sorted(set(left_index) & set(right_index))
        unmatched = sorted(set(left_index) ^ set(right_index))
        if unmatched:
            shown = ", ".join(repr(key) for key in unmatched[:6])
            more = f" and {len(unmatched) - 6} more" if len(unmatched) > 6 else ""
            raise SeriesError(
                f"run {self.run_id!r}: {len(unmatched)} value(s) of {unit!r} appear under only "
                f"one of {baseline!r} and {candidate!r} ({shown}{more}), so those rows have no "
                f"pair. A paired test needs both conditions scored on every unit"
            )
        if not shared:
            raise SeriesError(
                f"run {self.run_id!r}: {baseline!r} and {candidate!r} share no value of {unit!r}, "
                f"so there is nothing to pair"
            )
        return (
            left.take(np.array([left_index[key] for key in shared], dtype=np.int64)),
            right.take(np.array([right_index[key] for key in shared], dtype=np.int64)),
        )

    def numeric_config(self) -> dict[str, float]:
        """Config entries usable as numbers, matching `Experiment.numeric_config`."""
        return {
            name: float(value)
            for name, value in self.config.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }

    def summary(self) -> str:
        fields = ", ".join(self.field_names) or "no measurements"
        groups = ", ".join(self.group_names) or "no group keys"
        return (
            f"{self.run_id} [{self.source_format}] {self.n_rows} rows; "
            f"fields: {fields}; groups: {groups}"
        )


def _unit_index(keys: np.ndarray, unit: str, condition: str, run_id: str) -> dict[str, int]:
    """`{unit key: row}`, refusing a key that appears more than once."""
    index: dict[str, int] = {}
    duplicates: list[str] = []
    for position, key in enumerate(keys):
        if key in index:
            duplicates.append(str(key))
            continue
        index[str(key)] = position
    if duplicates:
        shown = ", ".join(repr(key) for key in sorted(set(duplicates))[:6])
        raise SeriesError(
            f"run {run_id!r}: {unit!r} is not unique within {condition!r} ({shown}), so a row "
            f"cannot be paired with one row on the other side. Pick a key that identifies a "
            f"unit, or aggregate the duplicates first"
        )
    return index
