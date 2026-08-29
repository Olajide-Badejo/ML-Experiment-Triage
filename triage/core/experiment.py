"""The one model every parser produces and every analysis consumes.

A single training run is an `Experiment`: an identifier, the configuration it
was launched with, and a set of tagged step series. Everything downstream, the
permutation tests, the regression gates, the sensitivity ranking and the
reports, reads this model and nothing else, which is what lets a TensorBoard
run and a JSONL log be compared without either analysis or report knowing
where the numbers came from.

Values are held as float32 on purpose. That is the precision the store writes,
so keeping the in memory model at the same precision makes a store round trip
exact rather than merely close, and the unit suite asserts that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Configuration keys that identify a replicate rather than a condition. Two runs
# that differ only in these are seed replicates of the same variant.
SEED_KEYS: tuple[str, ...] = ("seed", "random_seed", "rng_seed", "replicate")


class SeriesError(ValueError):
    """Raised when a series cannot be built or is unusable for a statistic."""


def canonical_config_value(value: Any) -> str:
    """Spell one config value so that equal values spell the same.

    `repr()` does not have that property over the types a config actually
    holds. A batch size written `32` by one launcher and `32.0` by another is
    one condition, but `repr()` spells them `32` and `32.0`, which split a five
    seed condition into five singletons; every one of them then failed the two
    runs per side test and silently fell back to the weaker single run mode,
    where a difference explained entirely by the seed can be reported as
    significant. So an integral number is spelled as the integer it equals
    whatever Python type carries it.

    Booleans are deliberately NOT folded into 0 and 1 even though `bool` is a
    subclass of `int`: `amp: true` and `amp: 1` are the same run in practice,
    but a config that carries both spellings is more likely to be describing
    two different things than one, and keeping them apart costs at worst an
    unnecessary split, while folding them costs a wrong grouping.
    """
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number) and number == int(number):
            return str(int(number))
        return repr(value)
    return repr(value)


@dataclass(frozen=True)
class MetricSeries:
    """One tagged scalar series from one run, sorted by step and deduplicated.

    Every value in a constructed series is finite. Non finite points are
    dropped on construction and counted, which is the single chokepoint the
    whole tool relies on: no statistic downstream has to defend itself against
    a NaN, because one cannot get this far.

    Attributes:
        tag: metric name as it appeared in the source, for example `val/loss`.
        steps: strictly increasing int64 training steps.
        values: finite float32 metric values, one per step.
        wall_times: optional float64 seconds since the epoch, one per step.
        dropped_non_finite: how many points were discarded as NaN or infinite.
            Callers pass a starting count when points were already dropped
            upstream, for example a parser that discarded a `null`.
    """

    tag: str
    steps: np.ndarray
    values: np.ndarray
    wall_times: np.ndarray | None = None
    dropped_non_finite: int = 0

    def __post_init__(self) -> None:
        raw_steps = np.asarray(self.steps).reshape(-1)
        if raw_steps.size and not np.issubdtype(raw_steps.dtype, np.integer):
            as_float = np.asarray(raw_steps, dtype=np.float64)
            if not bool(np.isfinite(as_float).all()):
                raise SeriesError(f"tag {self.tag!r} has non finite steps")
        steps = np.asarray(raw_steps, dtype=np.int64).reshape(-1)
        values = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if steps.size != values.size:
            raise SeriesError(f"tag {self.tag!r} has {steps.size} steps but {values.size} values")
        wall_times = self.wall_times
        if wall_times is not None:
            wall_times = np.asarray(wall_times, dtype=np.float64).reshape(-1)
            if wall_times.size != steps.size:
                raise SeriesError(
                    f"tag {self.tag!r} has {wall_times.size} wall times for {steps.size} steps"
                )

        # The non finite chokepoint. NaN and +/- infinity reach a series from
        # divergence, from a JSON `NaN` literal, and from a CSV cell reading
        # `inf`, and every one of them fabricates a result downstream: a NaN in
        # a final window makes the permutation comparison return an exact p of
        # 0.0, and an infinite effect ranks a diverged run first. There is no
        # honest arithmetic to do with such a point, so it is dropped here,
        # once, where every parser and every store load passes through, and the
        # count is kept so the loss is reported rather than hidden.
        finite = np.isfinite(values)
        dropped = int(finite.size - int(finite.sum()))
        if dropped:
            steps, values = steps[finite], values[finite]
            if wall_times is not None:
                wall_times = wall_times[finite]

        # Sort by step, then keep the last record for any repeated step. A step
        # is written more than once when a job is resumed from a checkpoint, and
        # the later write is the one that survived.
        order = np.argsort(steps, kind="stable")
        steps, values = steps[order], values[order]
        if wall_times is not None:
            wall_times = wall_times[order]
        keep = np.ones(steps.size, dtype=bool)
        if steps.size > 1:
            keep[:-1] = steps[:-1] != steps[1:]
        object.__setattr__(self, "steps", steps[keep])
        object.__setattr__(self, "values", values[keep])
        object.__setattr__(self, "wall_times", None if wall_times is None else wall_times[keep])
        object.__setattr__(self, "dropped_non_finite", int(self.dropped_non_finite) + dropped)

    def __len__(self) -> int:
        return int(self.steps.size)

    @property
    def is_empty(self) -> bool:
        return self.steps.size == 0

    @property
    def final_value(self) -> float:
        if self.is_empty:
            raise SeriesError(f"tag {self.tag!r} has no points")
        return float(self.values[-1])

    def window_size(self, fraction: float, minimum: int) -> int:
        """Size of the final window: `fraction` of the run, at least `minimum`.

        The window is clipped to the length of the series, so a run shorter than
        `minimum` contributes every point it has. Callers that need a hard floor
        must check `len(series)` themselves; `comparison.py` does exactly that.
        """
        if not 0.0 < fraction <= 1.0:
            raise SeriesError(f"window fraction must be in (0, 1], got {fraction}")
        wanted = max(minimum, math.ceil(fraction * len(self)))
        return min(wanted, len(self))

    def final_window(self, fraction: float = 0.1, minimum: int = 20) -> np.ndarray:
        """The last `window_size` values, as float64 for downstream arithmetic."""
        if self.is_empty:
            raise SeriesError(f"tag {self.tag!r} has no points")
        return self.values[-self.window_size(fraction, minimum) :].astype(np.float64)

    def smoothed(self, window: int = 9) -> np.ndarray:
        """Centred moving average, edge padded so the output length is preserved.

        Used for plotting and for the comparison statistic. Smoothing before
        taking the window mean lowers the variance of the statistic without
        moving its expectation, because the filter is symmetric.
        """
        values = self.values.astype(np.float64)
        if window <= 1 or values.size == 0:
            return values
        window = min(window, values.size)
        if window % 2 == 0:
            window -= 1
        if window <= 1:
            return values
        pad = window // 2
        padded = np.pad(values, pad, mode="edge")
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(padded, kernel, mode="valid")


@dataclass
class Experiment:
    """One training run: identity, configuration, tagged series, provenance."""

    run_id: str
    source_path: str
    source_format: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, MetricSeries] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tags(self) -> list[str]:
        return sorted(self.metrics)

    def series(self, tag: str) -> MetricSeries:
        try:
            return self.metrics[tag]
        except KeyError as error:
            available = ", ".join(self.tags) or "none"
            raise KeyError(
                f"run {self.run_id!r} has no tag {tag!r}; available tags: {available}"
            ) from error

    def has(self, tag: str) -> bool:
        return tag in self.metrics and not self.metrics[tag].is_empty

    @property
    def seed(self) -> int | None:
        """The replicate seed from the config, or None when the run declares none."""
        for key in SEED_KEYS:
            if key in self.config:
                try:
                    return int(self.config[key])
                except (TypeError, ValueError):
                    return None
        return None

    @property
    def variant_key(self) -> str:
        """Identity of the *condition* this run belongs to, ignoring the seed.

        Runs sharing a variant key are seed replicates of one another, which is
        what makes the strong permutation mode available. An explicit
        `variant` or `group` entry in the config wins; otherwise the key is
        built from every non seed configuration entry, sorted for stability and
        spelled through `canonical_config_value` so that two spellings of one
        number do not split one condition in two.
        """
        for key in ("variant", "group"):
            if key in self.config:
                return str(self.config[key])
        parts = [
            f"{name}={canonical_config_value(self.config[name])}"
            for name in sorted(self.config)
            if name not in SEED_KEYS
        ]
        return ", ".join(parts) if parts else self.run_id

    def numeric_config(self) -> dict[str, float]:
        """Config entries usable as hyperparameters in the sensitivity analysis.

        Booleans arrive as 0 and 1 for free, since `bool` is a subclass of `int`.
        Seeds are excluded because correlating a metric against the seed tests
        the harness rather than a hyperparameter. Strings, nulls and non finite
        values are dropped, because none of them can be rank correlated.
        """
        return {
            name: float(value)
            for name, value in self.config.items()
            if name not in SEED_KEYS
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }

    def summary(self) -> str:
        tags = ", ".join(f"{tag} ({len(series)})" for tag, series in sorted(self.metrics.items()))
        return f"{self.run_id} [{self.source_format}] {tags or 'no metrics'}"
