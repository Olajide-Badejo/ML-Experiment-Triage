"""Which hyperparameters actually moved the metric, and on how much evidence.

Spearman rank correlation between each numeric hyperparameter and the final
window statistic, one correlation per parameter per metric.

Rank correlation rather than an ANOVA, for two reasons. Sweeps are almost never
balanced, because people add runs where the results looked interesting, and
ANOVA wants a designed grid. And hyperparameter effects are usually monotonic
but strongly nonlinear: halving the learning rate does not halve the loss, and
learning rate is normally swept on a log scale in the first place. Rank
correlation handles both without asking anyone to pick a transform.

Null hypothesis for each correlation: the parameter and the metric are
statistically independent, so the observed rank correlation is what shuffling
one against the other would give. Assumption: runs are independent draws.

The count `n` is reported beside every correlation and is not optional. A
correlation of 0.9 over four points is nearly meaningless, and the only defence
against reading it as a finding is putting the four in front of the reader.

**The limitation this method has, stated plainly.** Rank correlation sees
monotone relationships and is blind to everything else. A hyperparameter with an
optimum in the middle of its swept range, which is the normal shape for a
learning rate, produces a U shaped relationship with the loss, and the rank
correlation of a U shape is near zero no matter how strong the effect is. The
demo sweep in this repository is built with exactly that shape, and the tool
duly reports a weak learning rate correlation beside a strong batch size one,
because batch size was swept monotonically and learning rate was not. A near
zero correlation here means "not monotone", never "no effect", and the reports
say so where the table is printed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from triage.analysis.comparison import ComparisonConfig, direction_for, window_statistic
from triage.core.experiment import Experiment

# Below this many distinct parameter values a rank correlation says nothing, so
# the parameter is skipped rather than reported with a caveat nobody reads.
MIN_RUNS = 4
MIN_DISTINCT_VALUES = 3


@dataclass(frozen=True)
class SensitivityResult:
    """One hyperparameter against one metric."""

    parameter: str
    tag: str
    correlation: float
    p_value: float
    n_runs: int
    n_distinct_values: int
    value_range: tuple[float, float]
    higher_is_better: bool
    test_name: str = "Spearman rank correlation, two sided"

    @property
    def direction(self) -> str:
        """What raising this parameter does to the metric, in plain words."""
        if abs(self.correlation) < 0.1:
            return "no clear monotone relationship"
        rises = self.correlation > 0
        better = rises == self.higher_is_better
        return f"raising it {'raises' if rises else 'lowers'} the metric, which is " + (
            "better" if better else "worse"
        )

    @property
    def strength(self) -> str:
        magnitude = abs(self.correlation)
        if magnitude >= 0.7:
            return "strong"
        if magnitude >= 0.4:
            return "moderate"
        if magnitude >= 0.1:
            return "weak"
        return "negligible"

    def p_value_label(self) -> str:
        return f"p = {self.p_value:.4f} ({self.test_name}, n = {self.n_runs})"


def analyse(
    experiments: list[Experiment],
    tags: list[str] | None = None,
    config: ComparisonConfig | None = None,
) -> list[SensitivityResult]:
    """Rank correlate every numeric hyperparameter against every metric.

    Results are sorted by the size of the correlation, strongest first, with the
    number of runs behind each one carried through to the report.
    """
    config = config or ComparisonConfig()
    if tags is None:
        tags = sorted({tag for run in experiments for tag in run.tags})

    results: list[SensitivityResult] = []
    for tag in tags:
        usable = [run for run in experiments if run.has(tag)]
        if len(usable) < MIN_RUNS:
            continue
        statistics = np.array(
            [window_statistic(run.series(tag), config) for run in usable], dtype=np.float64
        )

        by_parameter: dict[str, list[float | None]] = defaultdict(list)
        for run in usable:
            numeric = run.numeric_config()
            for parameter in {name for other in usable for name in other.numeric_config()}:
                by_parameter[parameter].append(numeric.get(parameter))

        for parameter, raw_values in sorted(by_parameter.items()):
            present = np.array([value is not None for value in raw_values], dtype=bool)
            if present.sum() < MIN_RUNS:
                continue
            values = np.array(
                [value for value in raw_values if value is not None], dtype=np.float64
            )
            metric = statistics[present]
            distinct = int(np.unique(values).size)
            if distinct < MIN_DISTINCT_VALUES:
                continue  # a constant, or an on off switch, is not a sweep

            correlation, p_value = scipy_stats.spearmanr(values, metric)
            if not np.isfinite(correlation):
                continue

            results.append(
                SensitivityResult(
                    parameter=parameter,
                    tag=tag,
                    correlation=float(correlation),
                    p_value=float(p_value),
                    n_runs=int(present.sum()),
                    n_distinct_values=distinct,
                    value_range=(float(values.min()), float(values.max())),
                    higher_is_better=direction_for(tag, config),
                )
            )

    return sorted(results, key=lambda r: (-abs(r.correlation), r.parameter, r.tag))


@dataclass
class SensitivityReport:
    """Sensitivity results plus the caveat that has to travel with them."""

    results: list[SensitivityResult] = field(default_factory=list)

    @property
    def strongest(self) -> SensitivityResult | None:
        return self.results[0] if self.results else None

    def caveat(self) -> str:
        if not self.results:
            return "No hyperparameter varied over enough runs to correlate against."
        smallest = min(result.n_runs for result in self.results)
        return (
            f"Rank correlation over as few as {smallest} runs in places. These are "
            f"associations across a sweep, not controlled effects: a parameter that was only "
            f"ever changed alongside another cannot be separated from it here."
        )
