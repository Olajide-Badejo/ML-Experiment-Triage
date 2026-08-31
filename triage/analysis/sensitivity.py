"""Which hyperparameters actually moved the metric, and on how much evidence.

Spearman rank correlation between each numeric hyperparameter and the final
window statistic, one correlation per parameter per metric.

Rank correlation rather than an ANOVA, for two reasons. Sweeps are almost never
balanced, because people add runs where the results looked interesting, and
ANOVA wants a designed grid. And hyperparameter effects are usually monotonic
but strongly nonlinear: halving the learning rate does not halve the loss, and
learning rate is normally swept on a log scale in the first place. Rank
correlation handles both without asking anyone to pick a transform.

**The unit of analysis is the condition, not the run.** Eight seed replicates of
one learning rate are eight measurements of one point, not eight independent
draws of the learning rate, so the runs of a variant are averaged before anything
is correlated. Feeding replicates in as independent points is the single most
expensive mistake available here: measured on five learning rate levels with
eight seeds each, it reported `rho = -0.594, p = 5.3e-05, n = 40` where the
honest answer over five points cannot be smaller than 2/120 = 0.0167. That is
about four orders of magnitude of fabricated confidence, and it looks exactly
like a finding.

Null hypothesis for each correlation: the parameter and the metric are
statistically independent, so the observed rank correlation is what pairing one
against the other at random would give. The p value is that permutation
distribution, enumerated exactly when the number of conditions permits and
resampled with a recorded seed when it does not. `scipy.stats.spearmanr`'s p
value is not used: it is a t approximation that returns exactly 0.0 at n = 4
(verified) and is invalid under ties, which a swept grid produces constantly.

The counts `n_variants` and `n_runs` are reported beside every correlation and
are not optional. A correlation of 0.9 over four conditions is nearly
meaningless, and the only defence against reading it as a finding is putting the
four in front of the reader. Below `MIN_VARIANTS_FOR_P` conditions no p value is
reported at all, because none could be small: the smallest two sided p over 4!
pairings is 2/24 = 0.083.

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

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from triage.analysis.comparison import ComparisonConfig, direction_for, window_statistic
from triage.analysis.regression import benjamini_hochberg
from triage.core.experiment import Experiment

# Below this many distinct parameter values a rank correlation says nothing, so
# the parameter is skipped rather than reported with a caveat nobody reads.
MIN_RUNS = 4
MIN_VARIANTS = 4
MIN_DISTINCT_VALUES = 3

#: Conditions needed before a permutation p value is reported at all. At four the
#: exact two sided floor over 4! pairings is 2/24 = 0.083, so no result from four
#: conditions can be significant at any conventional rate and a p value printed
#: beside one is decoration. The correlation and its counts are still reported.
MIN_VARIANTS_FOR_P = 5

#: Enumerate the pairing permutations exactly while `n_variants!` is no larger
#: than this; resample above it. Matches the spirit of the comparison layer's
#: exhaustive limit: exact where exactness is affordable, and said either way.
EXHAUSTIVE_PAIRINGS = 50_000

#: The false discovery rate the parameter by metric grid is corrected at. The
#: grid is a family: a sweep with six parameters and four metrics asks
#: twenty four questions at once, and answering them uncorrected guarantees a
#: "finding".
GRID_FALSE_DISCOVERY_RATE = 0.05


@dataclass(frozen=True)
class SensitivityResult:
    """One hyperparameter against one metric."""

    parameter: str
    tag: str
    correlation: float
    #: The permutation p value, or None when there were too few conditions to
    #: compute one that could mean anything.
    p_value: float | None
    n_runs: int
    n_variants: int
    n_distinct_values: int
    value_range: tuple[float, float]
    higher_is_better: bool
    test_name: str = "Spearman rank correlation, two sided"
    #: Benjamini Hochberg adjusted across the parameter by metric grid, or None
    #: when no p value was reported.
    adjusted_p: float | None = None
    #: Whether the adjusted p cleared the grid's false discovery rate.
    significant: bool = False

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
        """The p value with the test and the evidence behind it, never bare."""
        if self.p_value is None:
            return (
                f"p not reported: {self.n_variants} variants, and at least "
                f"{MIN_VARIANTS_FOR_P} are needed before a permutation p value can be "
                f"small enough to mean anything ({self.test_name}, over "
                f"{self.n_runs} runs)"
            )
        return (
            f"p = {self.p_value:.4f} ({self.test_name}, n variants = {self.n_variants}, "
            f"n runs = {self.n_runs})"
        )


def _spearman_permutation_p(
    values: np.ndarray, metric: np.ndarray, config: ComparisonConfig
) -> tuple[float, str]:
    """Two sided p for the rank correlation, by permuting the pairings.

    The null is that the parameter and the metric are paired at random, so the
    reference distribution is the correlation over every reassignment of one to
    the other. It is enumerated exactly while that is affordable and resampled
    from the configured seed above it, and the returned name says which.

    Spearman's rho on ranks is Pearson's r on ranks, so the ranks are taken once
    and the statistic over each permutation is a vectorised correlation. That
    keeps the exact enumeration cheap enough to be the default.
    """
    ranked_values = scipy_stats.rankdata(values)
    ranked_metric = scipy_stats.rankdata(metric)
    centred_metric = ranked_metric - ranked_metric.mean()
    metric_norm = float(np.sqrt(np.sum(centred_metric**2)))

    def statistic(sample: np.ndarray, axis: int = -1) -> np.ndarray:
        centred = sample - sample.mean(axis=axis, keepdims=True)
        numerator = np.sum(centred * centred_metric, axis=axis)
        denominator = np.sqrt(np.sum(centred**2, axis=axis)) * metric_norm
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denominator > 0, numerator / denominator, 0.0)

    arrangements = math.factorial(values.size)
    exact = arrangements <= EXHAUSTIVE_PAIRINGS
    n_resamples = arrangements if exact else config.n_permutations
    test = scipy_stats.permutation_test(
        (ranked_values,),
        statistic,
        permutation_type="pairings",
        alternative="two-sided",
        n_resamples=n_resamples,
        vectorized=True,
        rng=np.random.default_rng(config.seed),
    )
    how = (
        f"exact over {arrangements} pairings"
        if exact
        else f"{config.n_permutations} resamples of the pairings"
    )
    return float(test.pvalue), f"Spearman rank correlation, two sided permutation test, {how}"


def _variant_means(
    experiments: list[Experiment], tag: str, config: ComparisonConfig
) -> dict[str, tuple[float, dict[str, float], int]]:
    """One point per condition: its mean final window statistic and its config.

    Runs sharing a variant key are seed replicates of one another, so they are
    one point on the parameter axis and are averaged onto it. Their configs are
    identical apart from the seed by construction of the key, so the first run's
    numeric config describes the whole condition.
    """
    grouped: dict[str, list[Experiment]] = defaultdict(list)
    for run in experiments:
        grouped[run.variant_key].append(run)

    points: dict[str, tuple[float, dict[str, float], int]] = {}
    for variant, runs in grouped.items():
        statistics = [window_statistic(run.series(tag), config) for run in runs]
        points[variant] = (
            float(np.mean(statistics)),
            runs[0].numeric_config(),
            len(runs),
        )
    return points


def analyse(
    experiments: list[Experiment],
    tags: list[str] | None = None,
    config: ComparisonConfig | None = None,
) -> list[SensitivityResult]:
    """Rank correlate every numeric hyperparameter against every metric.

    Results are sorted by evidence and then by the size of the correlation:
    everything that survives the correction across the parameter by metric grid
    first, strongest correlation first within each group. Ranking by the size of
    rho alone put a four point artefact above a corrected finding of the same
    magnitude, which is the table's whole job inverted.
    """
    config = config or ComparisonConfig()
    if tags is None:
        tags = sorted({tag for run in experiments for tag in run.tags})

    results: list[SensitivityResult] = []
    for tag in tags:
        usable = [run for run in experiments if run.has(tag)]
        if len(usable) < MIN_RUNS:
            continue
        points = _variant_means(usable, tag, config)
        variants = sorted(points)
        parameters = {name for _, numeric, _ in points.values() for name in numeric}

        for parameter in sorted(parameters):
            present = [variant for variant in variants if parameter in points[variant][1]]
            if len(present) < MIN_VARIANTS:
                continue
            values = np.array(
                [points[variant][1][parameter] for variant in present], dtype=np.float64
            )
            metric = np.array([points[variant][0] for variant in present], dtype=np.float64)
            distinct = int(np.unique(values).size)
            if distinct < MIN_DISTINCT_VALUES:
                continue  # a constant, or an on off switch, is not a sweep

            correlation = float(scipy_stats.spearmanr(values, metric).statistic)
            if not np.isfinite(correlation):
                continue

            if len(present) >= MIN_VARIANTS_FOR_P:
                p_value, test_name = _spearman_permutation_p(values, metric, config)
            else:
                p_value, test_name = None, "Spearman rank correlation, two sided"

            results.append(
                SensitivityResult(
                    parameter=parameter,
                    tag=tag,
                    correlation=correlation,
                    p_value=p_value,
                    n_runs=sum(points[variant][2] for variant in present),
                    n_variants=len(present),
                    n_distinct_values=distinct,
                    value_range=(float(values.min()), float(values.max())),
                    higher_is_better=direction_for(tag, config),
                    test_name=test_name,
                )
            )

    return _correct_the_grid(results)


def _correct_the_grid(results: list[SensitivityResult]) -> list[SensitivityResult]:
    """Benjamini Hochberg across the whole parameter by metric grid, then rank.

    Every correlation in the table was asked at the same time, off the same
    sweep, so the table is the family. Results carrying no p value are held out
    of the denominator on the same principle the regression layer holds out an
    inadmissible comparison: a question that cannot be answered should not cost
    the answerable ones power.
    """
    tested = [index for index, result in enumerate(results) if result.p_value is not None]
    corrected = list(results)
    if tested:
        adjusted = benjamini_hochberg(
            [results[index].p_value for index in tested],  # type: ignore[misc]
            GRID_FALSE_DISCOVERY_RATE,
        )
        for index, value in zip(tested, adjusted, strict=True):
            corrected[index] = SensitivityResult(
                **{
                    **results[index].__dict__,
                    "adjusted_p": float(value),
                    "significant": bool(value <= GRID_FALSE_DISCOVERY_RATE),
                }
            )
    return sorted(
        corrected,
        key=lambda r: (not r.significant, -abs(r.correlation), r.parameter, r.tag),
    )


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
        fewest = min(result.n_variants for result in self.results)
        return (
            f"Rank correlation over as few as {fewest} conditions ({smallest} runs) in "
            f"places, the condition being the unit because seed replicates of one "
            f"setting are one point and not several. These are associations across a "
            f"sweep, not controlled effects: a parameter that was only ever changed "
            f"alongside another cannot be separated from it here."
        )
