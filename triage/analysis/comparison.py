"""Does this run beat the baseline, by how much, and how sure are we.

Everything here reduces a training curve to one number and then asks whether
two sets of that number could plausibly have come from the same source.

**The statistic.** For a tagged series, take a centred moving average, then the
mean of the final window: the last `max(minimum, ceil(fraction * n))` points.
The final window is what people actually mean by "how good did this run get",
and averaging over it rather than reading the last point alone removes most of
the variance that makes single point comparisons useless.

**The test.** A two sided permutation test. Training curve values are
autocorrelated and rarely normal, which breaks the assumptions a t test needs.
A permutation test needs only exchangeability under the null, which is exactly
what the null hypothesis asserts (Good, *Permutation, Parametric, and Bootstrap
Tests of Hypotheses*, 3rd ed., Springer 2005).

The quantity permuted is the Welch t rather than the raw difference of means.
Full exchangeability is more than the null actually claims once the two
conditions have different spreads, and a raw mean difference is exact only
under the stronger assumption: measured at a nominal 5 percent, seven runs
against three with a fivefold spread ratio rejected 17.9 percent of true nulls.
Studentizing the permuted statistic restores the guarantee asymptotically
(Janssen, *Statistics and Probability Letters* 36(1), 1997) and changes nothing
at all when the two conditions hold the same number of runs. The effect and the
interval reported beside the p value are still on the mean difference, which is
the quantity a reader can act on.

**Three modes, answering three different questions.**

`seed_replicate`
    Several seeds per condition. The unit of analysis is the run, and the
    labels shuffle across runs. Null hypothesis: the condition label is
    exchangeable across runs, that is, the two conditions draw their final
    window means from the same distribution. This is the strong claim, because
    the seed to seed variance that dominates deep learning results is inside
    the null distribution rather than ignored by it.

`window_block`
    One run per side. There is no seed variance to work with, so the test falls
    back to permuting *blocks* of the final window between the two runs, with a
    block length taken from the estimated autocorrelation time. Null
    hypothesis: the two final windows are exchangeable blockwise, that is, they
    are segments of the same stationary process. This is a weaker claim and is
    labelled as such everywhere it appears. It cannot see run to run variance,
    so when seed variance is present it is anticonservative. The calibration
    suite measures exactly how anticonservative, and the number is published
    rather than buried.

`paired_cluster`
    Two conditions scored on the SAME units, which is the usual shape of an
    offline evaluation rather than a training sweep. The unit of analysis is
    the pair, the labels swap within a pair rather than shuffling across
    units, and whole clusters of pairs swap together when the units are not
    independent of one another. The statistic is whatever the caller passes,
    because what is being compared is often not a mean of anything.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from triage.core.experiment import Experiment, MetricSeries, SeriesError

DEFAULT_SEED = 20260805

# Tag name words that settle which direction is an improvement. Matched against
# the WORDS of the tag, longest first, so `val/loss` and `top1_accuracy` both
# resolve without configuration while `val/mape` is not read as a mean average
# precision because the three letters of `map` happen to start it.
LOWER_IS_BETTER = (
    "perplexity",
    "loss",
    "error",
    "err",
    "nll",
    "mse",
    "rmse",
    "mae",
    "mape",
    "smape",
    "wer",
    "cer",
    "fid",
    "regret",
)
HIGHER_IS_BETTER = (
    "accuracy",
    "acc",
    "f1",
    "auc",
    "auroc",
    "precision",
    "recall",
    "reward",
    "bleu",
    "rouge",
    "iou",
    "map",
    "score",
    "dice",
    "psnr",
    "ssim",
)

MODE_SEED_REPLICATE = "seed_replicate"
MODE_WINDOW_BLOCK = "window_block"
MODE_PAIRED_CLUSTER = "paired_cluster"

# Block length as a multiple of the estimated autocorrelation time, and the
# fewest blocks per run the window block mode will accept. Both numbers were
# set by measurement rather than taste: see `block_length` and the calibration
# suite in `tests/statistics`.
BLOCK_TAU_MULTIPLIER = 3.0
MIN_BLOCKS_PER_RUN = 8

#: Labels for the modes this module implements. The vocabulary is OPEN: a
#: caller that builds a `ComparisonResult` with a mode of its own gets the
#: fallback below rather than a `KeyError` out of `mode_label` and `to_dict`,
#: which between them are every reporting path there is. A consumer describing
#: its own design truthfully must not be punished for it.
MODE_LABELS = {
    MODE_SEED_REPLICATE: "seed replicated permutation test (strong claim)",
    MODE_WINDOW_BLOCK: (
        "single run window block permutation test (WEAKER CLAIM: cannot see seed to seed variance)"
    ),
    MODE_PAIRED_CLUSTER: (
        "paired permutation test with clustered label swaps (paired claim, clustered resampling)"
    ),
}
MODE_LABEL_FALLBACK = "{mode} (a mode this version of triage does not describe)"


class ComparisonError(ValueError):
    """Raised when a comparison cannot be made from the data supplied."""


@dataclass(frozen=True)
class ComparisonConfig:
    """Every knob of the comparison, in one place so reports can print it."""

    window_fraction: float = 0.10
    window_minimum: int = 20
    smoothing_window: int = 9
    n_permutations: int = 10000
    exhaustive_limit: int = 50000
    confidence_level: float = 0.95
    n_bootstrap: int = 2000
    alpha: float = 0.05
    min_replicates_for_seed_mode: int = 3
    seed: int = DEFAULT_SEED
    #: Per tag direction overrides, `{tag: higher_is_better}`. A tag listed here
    #: is never guessed at from its name, at any entry point.
    directions: Mapping[str, bool] = field(default_factory=dict)

    def describe(self) -> str:
        overrides = ""
        if self.directions:
            named = ", ".join(
                f"{tag} ({'higher' if better else 'lower'} is better)"
                for tag, better in sorted(self.directions.items())
            )
            overrides = f"; direction set by the caller for {named}"
        return (
            f"final window mean over the last max({self.window_minimum}, "
            f"{self.window_fraction:.0%} of steps) points of a "
            f"{self.smoothing_window} point moving average; "
            f"two sided permutation test at alpha {self.alpha:g}; "
            f"seed {self.seed}{overrides}"
        )


@dataclass(frozen=True)
class ComparisonResult:
    """One candidate against one baseline on one metric.

    Every field a reader needs to judge the claim is here, which is why the
    reports never have to reach back into the analysis to explain themselves.
    """

    tag: str
    baseline: str
    candidate: str
    mode: str
    test_name: str
    baseline_statistic: float
    candidate_statistic: float
    effect: float
    relative_effect_pct: float
    effect_size: float
    effect_size_name: str
    ci_low: float
    ci_high: float
    ci_method: str
    ci_level: float
    p_value: float
    n_permutations: int
    exact: bool
    min_attainable_p: float
    n_baseline: int
    n_candidate: int
    window_points: int
    higher_is_better: bool
    seed: int
    baseline_runs: tuple[str, ...] = ()
    candidate_runs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: Block length used by the window block mode, in steps; 0 in seed mode.
    block_length: int = 0

    @property
    def mode_label(self) -> str:
        """The claim this mode makes, or the mode itself when it is not ours."""
        return MODE_LABELS.get(self.mode, MODE_LABEL_FALLBACK.format(mode=self.mode))

    @property
    def is_weak_mode(self) -> bool:
        return self.mode == MODE_WINDOW_BLOCK

    @property
    def improved(self) -> bool:
        """True when the candidate moved the metric in the good direction."""
        return self.effect > 0 if self.higher_is_better else self.effect < 0

    @property
    def signed_improvement_pct(self) -> float:
        """Relative change oriented so that positive always means better."""
        return self.relative_effect_pct if self.higher_is_better else -self.relative_effect_pct

    def p_value_label(self) -> str:
        """The p value with the test and mode that produced it, never bare."""
        exactness = "exact" if self.exact else f"{self.n_permutations} resamples"
        return f"p = {self.p_value:.4f} ({self.test_name}, {self.mode_label}, {exactness})"

    def to_dict(self) -> dict[str, Any]:
        data = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        data["mode_label"] = self.mode_label
        data["improved"] = self.improved
        data["signed_improvement_pct"] = self.signed_improvement_pct
        data["p_value_label"] = self.p_value_label()
        return data


# --------------------------------------------------------------------- helpers


#: Word boundaries in a metric tag: any run of characters that is neither a
#: letter nor a digit. A camel case hump is a boundary too, so `valMAP` and
#: `val/map` produce the same words.
_NOT_A_WORD = re.compile(r"[^a-z0-9]+")
_CAMEL_HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tag_words(tag: str) -> frozenset[str]:
    """The words of a metric tag, for direction matching.

    Splitting matters more than it looks. Matching the table as substrings read
    `val/mape` and `val/smape` as mean average precision, because the three
    letters of `map` start both of them, and reported a rising error as an
    improvement. Words cannot do that: `mape` is a word, `map` is not in it.

    Camel case is read both ways, and deliberately. Splitting the humps is what
    finds the `loss` in `valLoss`; not splitting them is what finds the `map` in
    `mAP@50`, which is how that metric is conventionally spelled. Both spellings
    contribute words, because a word that matches under either reading is
    evidence, and neither reading can invent a word the tag does not contain.
    """
    plain = _NOT_A_WORD.split(tag.lower())
    humped = _NOT_A_WORD.split(_CAMEL_HUMP.sub(" ", tag).lower())
    return frozenset(word for word in plain + humped if word)


def infer_direction(tag: str) -> bool:
    """True when a larger value of this metric is better.

    Resolved from the tag name so that the common case needs no configuration.
    The longest matching word wins, so `val/top1_accuracy` is not decided by the
    `acc` entry before `accuracy` itself has been considered, and a tag matching
    nothing defaults to lower is better, which is what a loss is.
    """
    words = tag_words(tag)
    best_higher = max((len(f) for f in HIGHER_IS_BETTER if f in words), default=0)
    best_lower = max((len(f) for f in LOWER_IS_BETTER if f in words), default=0)
    return best_higher > best_lower


def direction_for(tag: str, config: ComparisonConfig | None = None) -> bool:
    """The direction in force for a tag: the caller's override, else the guess.

    Inference from the name is a convenience and it is sometimes wrong, which is
    survivable only if the answer can be corrected without editing this table.
    `ComparisonConfig.directions` is that correction, and it is threaded through
    every entry point rather than existing only on the innermost function.
    """
    if config is not None and tag in config.directions:
        return bool(config.directions[tag])
    return infer_direction(tag)


def _finite_window(
    series: MetricSeries, config: ComparisonConfig, smooth: bool = True
) -> np.ndarray:
    """The final window, or `ComparisonError` if it is not all finite.

    Smoothed by default, because the comparison statistic is a mean over the
    smoothed window. `smooth=False` returns the same window unfiltered, which
    is what the autocorrelation estimate has to see.

    This is the second of the two barriers against a non finite value, and it
    exists because the first one can be absent. `MetricSeries.__post_init__`
    drops non finite points, so a series built through the parsers can never
    carry one; a series loaded from a database written before that filter
    existed, or handed in by a caller of this module's public functions, can.

    What the barrier buys is worth its cost. One NaN in a final window makes
    the statistic NaN, `np.abs(null) >= nan` False for every arrangement, and
    the exact p value `0 / total`, so a diverged run is reported as the most
    significant finding in the table at p = 0.0000. An infinity does the same
    to the effect and the relative effect and ranks the run first. Refusing to
    produce a number is the only honest option, and it costs one comparison
    rather than the whole verdict table.
    """
    values = (
        series.smoothed(config.smoothing_window) if smooth else series.values.astype(np.float64)
    )
    size = series.window_size(config.window_fraction, config.window_minimum)
    window = values[-size:]
    if not bool(np.isfinite(window).all()):
        bad = int(window.size - int(np.isfinite(window).sum()))
        raise ComparisonError(
            f"tag {series.tag!r} has {bad} non finite value(s) in its final window of "
            f"{window.size} points, so no statistic computed from it would be meaningful. "
            f"Reingest the run: parsing drops non finite points and reports the count"
        )
    return window


def window_statistic(series: MetricSeries, config: ComparisonConfig) -> float:
    """The comparison statistic: mean of the smoothed final window."""
    if series.is_empty:
        raise ComparisonError(f"tag {series.tag!r} has no points")
    return float(np.mean(_finite_window(series, config)))


def window_values(series: MetricSeries, config: ComparisonConfig) -> np.ndarray:
    """The smoothed final window itself, needed by the block mode."""
    return _finite_window(series, config)


def raw_window_values(series: MetricSeries, config: ComparisonConfig) -> np.ndarray:
    """The same final window, unsmoothed, for estimating autocorrelation."""
    return _finite_window(series, config, smooth=False)


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    """Estimate tau, the number of samples between effectively independent ones.

    Uses the initial positive sequence: sum the autocorrelation function until
    it first goes non positive, which is the standard truncation for a noisy
    empirical ACF. Returned clipped to [1, n/4], because an estimate longer than
    a quarter of the sample is not one the sample can support.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    centred = values - values.mean()
    if n < 8 or not np.any(centred):
        return 1.0
    padded = int(2 ** math.ceil(math.log2(2 * n)))
    spectrum = np.fft.rfft(centred, padded)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), padded)[:n].real
    if acf[0] <= 0:
        return 1.0
    acf /= acf[0]
    tau = 1.0
    for lag in range(1, n):
        if acf[lag] <= 0:
            break
        tau += 2.0 * acf[lag]
    return float(np.clip(tau, 1.0, n / 4.0))


def block_length(windows: list[np.ndarray]) -> int:
    """Block length for the window block mode, from the autocorrelation time.

    Pass the RAW windows. Estimating tau on the smoothed window measures the
    moving average rather than the data: a nine point filter leaves any series
    correlated over about nine points, so white noise came back with a tau near
    9 instead of near 1, the block tripled with it, and the mode then refused
    almost every run of a realistic length. Two 400 step runs hold a 40 point
    final window, which is one block of 27 and no test at all.

    Two further details were both forced by the calibration suite, and both cost
    a measured type I error of about 12 percent against a nominal 5 before they
    were fixed.

    First, tau is estimated on each window separately and the larger is taken,
    never on the two windows concatenated. Concatenating two runs puts a step
    change at the join, which the estimator reads as long range dependence and
    which inflates tau by roughly a factor of two.

    Second, the block is three times tau, not one. At one tau the block means
    are still visibly correlated with their neighbours, and a permutation that
    scatters neighbouring blocks across both groups then produces a null
    distribution narrower than the truth, which is precisely how a test becomes
    anticonservative. Three tau puts the measured type I back on nominal.
    """
    tau = max(integrated_autocorrelation_time(window) for window in windows)
    return max(2, math.ceil(BLOCK_TAU_MULTIPLIER * tau))


def permutation_p_value(
    observed: float,
    null_distribution: np.ndarray,
    exact: bool,
) -> float:
    """Two sided p value: how often does chance alone match this effect.

    The add one correction on both counts is not cosmetic. A sampled p value of
    exactly zero is a claim the resampling cannot support, and the corrected
    form `(1 + #{|null| >= |observed|}) / (1 + B)` is the unbiased estimator for
    a Monte Carlo permutation test (Phipson and Smyth, *Statistical Applications
    in Genetics and Molecular Biology* 9(1), 2010). For an exhaustive
    enumeration the observed arrangement is itself one of the permutations, so
    the same expression is the exact p value.

    A non finite observed effect is refused rather than counted. Every IEEE
    comparison against a NaN is False, so `np.abs(null) >= nan` counts zero
    arrangements as extreme and the exact branch returns `0 / total`, which is
    a p value of exactly zero on a run that diverged: the single most confident
    claim the tool can make, made from the absence of a number.
    """
    if not math.isfinite(observed):
        raise ComparisonError(
            f"the observed effect is {observed!r}, which no permutation count can rank; "
            f"a non finite effect means the underlying window was not finite"
        )
    # The tie tolerance is relative, because floating point error is. An
    # absolute 1e-15 is the right size for values around 1 and far too small for
    # values around 50, where the last bits of a sum are worth about 1e-14: on
    # such data the observed arrangement failed its own comparison and the exact
    # p value came out as 0.0, which is not a value this test can produce.
    magnitude = max(abs(observed), float(np.abs(null_distribution).max(initial=0.0)))
    tolerance = 1e-9 * magnitude
    at_least_as_extreme = int(
        np.count_nonzero(np.abs(null_distribution) >= abs(observed) - tolerance)
    )
    total = null_distribution.size
    if exact:
        return float(at_least_as_extreme / total)
    return float((1 + at_least_as_extreme) / (1 + total))


def _label_permutations(
    n_total: int,
    n_first: int,
    config: ComparisonConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Index matrix of group memberships, enumerated exactly when that is cheap.

    Returns a boolean matrix of shape (n_permutations, n_total) whose rows each
    select `n_first` positions for the first group, and a flag saying whether
    the enumeration was exhaustive.

    Row zero of an exhaustive enumeration is the identity arrangement, the first
    `n_first` positions, which is the observed grouping itself. Callers rely on
    that: taking the observed statistic from that row rather than computing it
    separately is what guarantees the observed arrangement is inside its own
    null distribution, bit for bit.
    """
    total_arrangements = comb(n_total, n_first)
    if total_arrangements <= config.exhaustive_limit:
        from itertools import combinations

        masks = np.zeros((total_arrangements, n_total), dtype=bool)
        for row, chosen in enumerate(combinations(range(n_total), n_first)):
            masks[row, list(chosen)] = True
        return masks, True

    draws = rng.random((config.n_permutations, n_total))
    order = np.argsort(draws, axis=1)
    masks = np.zeros((config.n_permutations, n_total), dtype=bool)
    np.put_along_axis(masks, order[:, :n_first], True, axis=1)
    return masks, False


def _identity_mask(n_total: int, n_first: int) -> np.ndarray:
    """The one row arrangement that is the data as it actually arrived."""
    mask = np.zeros((1, n_total), dtype=bool)
    mask[0, :n_first] = True
    return mask


def _studentized_null(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Welch t for every arrangement, second group minus first.

    This is the statistic the permutation is over, and permuting it rather than
    the raw mean difference is what keeps the test near nominal when the two
    conditions have both different spreads and different numbers of runs
    (Janssen, *Statistics and Probability Letters* 36(1), 1997). A raw mean
    difference is exact only under full exchangeability, which unequal variances
    break: measured at a nominal 5 percent, 7 runs against 3 with a 5x spread
    ratio rejected 17.9 percent of true nulls, and the more seeds sat on the
    stable baseline the worse it got, which is the wrong way round for the most
    common real design.

    Studentizing costs nothing where it is not needed. With equal group sizes
    the denominator is a decreasing function of the squared mean difference, so
    the arrangements rank in exactly the order the mean difference ranked them
    and every balanced p value is unchanged.

    Both group sums are formed directly rather than one being reconstructed as
    `total - first_sums`, and the values are centred before any of it. The
    subtraction looks free and is not: on values whose mean is far from zero it
    cancels away most of the significant digits, the null it produced disagreed
    with a directly computed observed statistic by more than the tie tolerance,
    and the observed arrangement was then counted out of its own null
    distribution, for an exact p value of 0.0 that no design can attain.
    Centring is the same argument applied to the sums of squares: a shift moves
    every group mean by the same amount and cancels out of the numerator and the
    denominator alike, so it costs nothing and it keeps the variances from being
    differences of two large numbers.
    """
    n_first = int(masks[0].sum())
    n_second = values.size - n_first
    centred = values - values.mean()
    squares = centred**2
    mean_first = (masks @ centred) / n_first
    mean_second = ((~masks) @ centred) / n_second
    # var = (sum of squares - n * mean^2) / (n - 1), clipped at zero because the
    # subtraction can land a hair below it on constant data.
    variance_first = np.maximum((masks @ squares) - n_first * mean_first**2, 0.0) / (n_first - 1)
    variance_second = np.maximum((~masks) @ squares - n_second * mean_second**2, 0.0) / (
        n_second - 1
    )
    difference = mean_second - mean_first
    spread = np.sqrt(variance_first / n_first + variance_second / n_second)

    # A zero denominator means both groups are constant. If they are constant at
    # the same value the arrangement is not extreme at all; if they are constant
    # at different values it is the most extreme there is. Neither is infinite,
    # and an infinity here would be refused later as a non finite statistic, so
    # the second case is given a magnitude just above every finite one, which
    # ranks it correctly and ties it with its mirror image.
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = np.where(spread > 0, difference / np.where(spread > 0, spread, 1.0), 0.0)
    degenerate = (spread <= 0) & (difference != 0)
    if degenerate.any():
        finite_maximum = float(np.abs(studentized[~degenerate]).max(initial=0.0))
        studentized = np.where(
            degenerate, np.sign(difference) * (finite_maximum + 1.0), studentized
        )
    return studentized


#: A design is warned about when the smaller condition is below this many runs
#: AND the two conditions' spreads differ by more than this ratio. Studentizing
#: makes the test asymptotically right under unequal variance; at few runs the
#: variance estimates that go into it are themselves poor, and the guarantee
#: becomes approximate rather than exact.
HETEROSCEDASTICITY_MIN_REPLICATES = 5
HETEROSCEDASTICITY_SD_RATIO = 2.0


def _heteroscedasticity_warnings(first: np.ndarray, second: np.ndarray) -> list[str]:
    """Say so when the design is in the regime the guarantee is weakest in."""
    smaller = min(first.size, second.size)
    if smaller < 2 or smaller >= HETEROSCEDASTICITY_MIN_REPLICATES:
        return []
    sd_first = float(np.std(first, ddof=1))
    sd_second = float(np.std(second, ddof=1))
    low, high = sorted((sd_first, sd_second))
    if high <= 0:
        return []
    ratio = high / low if low > 0 else math.inf
    if ratio <= HETEROSCEDASTICITY_SD_RATIO:
        return []
    spelled = "beyond measuring" if math.isinf(ratio) else f"{ratio:.1f}x"
    return [
        f"the smaller condition has {smaller} runs and the two conditions' spreads differ by "
        f"{spelled} ({sd_first:.4g} against {sd_second:.4g}); the studentized permutation test "
        f"is approximate rather than exact in this regime, so read the p value as indicative "
        f"and add seeds to the narrower condition before concluding"
    ]


def _min_attainable_p(n_first: int, n_second: int, exact: bool, config: ComparisonConfig) -> float:
    """The smallest p value this design can produce at all.

    Two sided p values come out of counting arrangements at least as extreme as
    the observed one, so the floor is set by how many of those the enumeration
    holds. When the groups are the same size, the complement of every
    arrangement is also enumerated and carries the same effect with the opposite
    sign, so the extreme count can never be lower than two and the floor is
    2/C. When they are not, the complement of an `n1` against `n2` split is an
    `n2` against `n1` split, which is not in the enumeration at all, and the
    floor is 1/C.

    Assuming the equal size case everywhere overstated the floor by a factor of
    two on every unbalanced design: at 2 against 5 it reported 0.0952 where
    0.0476 is attainable, so designs that can clear alpha were declared
    inconclusive before their p value was looked at.
    """
    if not exact:
        return 1.0 / (1 + config.n_permutations)
    arrangements = comb(n_first + n_second, n_first)
    return (2.0 if n_first == n_second else 1.0) / arrangements


def _observed_statistic(
    values: np.ndarray,
    n_first: int,
    null_distribution: np.ndarray,
    exact: bool,
) -> float:
    """The statistic for the arrangement the data actually came in.

    In exact mode this is literally row zero of the null distribution. In
    sampled mode the identity arrangement is not among the draws, so it is
    computed here through the same expression rather than a different one.
    """
    if exact:
        return float(null_distribution[0])
    identity = _identity_mask(values.size, n_first)
    return float(_studentized_null(values, identity)[0])


# ------------------------------------------------------------ mode one: seeds


def compare_seed_replicated(
    baseline_runs: list[Experiment],
    candidate_runs: list[Experiment],
    tag: str,
    config: ComparisonConfig | None = None,
    higher_is_better: bool | None = None,
    baseline_name: str | None = None,
    candidate_name: str | None = None,
) -> ComparisonResult:
    """Permutation test over seed replicates. The strong claim.

    Null hypothesis: the condition label is exchangeable across runs, so the
    final window means of the two conditions are draws from one distribution.

    Assumptions: runs within a condition are independent replicates differing
    only by seed; the final window is a fair summary of where each run arrived.
    No distributional form is assumed, and seed to seed variance is inside the
    null distribution rather than assumed away.
    """
    config = config or ComparisonConfig()
    baseline_name = baseline_name or (baseline_runs[0].variant_key if baseline_runs else "baseline")
    candidate_name = candidate_name or (
        candidate_runs[0].variant_key if candidate_runs else "candidate"
    )
    if higher_is_better is None:
        higher_is_better = direction_for(tag, config)

    baseline_stats = _statistics_for(baseline_runs, tag, config)
    candidate_stats = _statistics_for(candidate_runs, tag, config)
    n_baseline, n_candidate = baseline_stats.size, candidate_stats.size
    if n_baseline < 2 or n_candidate < 2:
        raise ComparisonError(
            f"seed replicated mode needs at least 2 runs per condition on {tag!r}; "
            f"got {n_baseline} baseline and {n_candidate} candidate"
        )

    warnings: list[str] = []
    if min(n_baseline, n_candidate) < config.min_replicates_for_seed_mode:
        warnings.append(
            f"only {min(n_baseline, n_candidate)} seeds in the smaller condition; "
            f"{config.min_replicates_for_seed_mode} is the recommended minimum"
        )

    warnings.extend(_heteroscedasticity_warnings(baseline_stats, candidate_stats))

    pooled = np.concatenate([baseline_stats, candidate_stats])
    rng = np.random.default_rng(config.seed)
    masks, exact = _label_permutations(pooled.size, n_baseline, config, rng)
    null_distribution = _studentized_null(pooled, masks)

    # The test statistic comes out of the same expression as every null value,
    # from the identity arrangement, so that in exact mode it IS
    # `null_distribution[0]` rather than a number that merely ought to equal it.
    # The effect below is the mean difference, which is what a reader wants
    # reported; the Welch t is what the arrangements are ranked by.
    observed = _observed_statistic(pooled, n_baseline, null_distribution, exact)
    p_value = permutation_p_value(observed, null_distribution, exact)
    effect = float(candidate_stats.mean() - baseline_stats.mean())

    # The smallest p value this design can produce at all. With three seeds per
    # condition that is 0.1, so no result can ever clear alpha 0.05, and saying
    # so is more useful than reporting a p value that was never able to fire.
    min_attainable = _min_attainable_p(n_baseline, n_candidate, exact, config)
    if min_attainable > config.alpha:
        warnings.append(
            f"with {n_baseline} and {n_candidate} seeds the smallest attainable p value is "
            f"{min_attainable:.3f}, above alpha {config.alpha:g}; add seeds before concluding"
        )

    spread = _pooled_standard_deviation(baseline_stats, candidate_stats)
    effect_size = effect / spread if spread > 0 else 0.0
    ci_low, ci_high = _welch_interval(baseline_stats, candidate_stats, config.confidence_level)

    return ComparisonResult(
        tag=tag,
        baseline=baseline_name,
        candidate=candidate_name,
        mode=MODE_SEED_REPLICATE,
        test_name="two sided studentized permutation test (Welch t) on the final window mean",
        baseline_statistic=float(baseline_stats.mean()),
        candidate_statistic=float(candidate_stats.mean()),
        effect=effect,
        relative_effect_pct=_relative(effect, float(baseline_stats.mean())),
        effect_size=float(effect_size),
        effect_size_name="Cohen's d over seed level statistics",
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="Welch t interval on the seed level statistics",
        ci_level=config.confidence_level,
        p_value=p_value,
        n_permutations=int(null_distribution.size),
        exact=exact,
        min_attainable_p=float(min_attainable),
        n_baseline=n_baseline,
        n_candidate=n_candidate,
        window_points=_window_points(baseline_runs + candidate_runs, tag, config),
        higher_is_better=higher_is_better,
        seed=config.seed,
        baseline_runs=tuple(run.run_id for run in baseline_runs),
        candidate_runs=tuple(run.run_id for run in candidate_runs),
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------ mode two: blocks


def compare_window_block(
    baseline_run: Experiment,
    candidate_run: Experiment,
    tag: str,
    config: ComparisonConfig | None = None,
    higher_is_better: bool | None = None,
) -> ComparisonResult:
    """Block permutation test over the two final windows. The weaker claim.

    Null hypothesis: the two final windows are blockwise exchangeable, that is,
    both are segments of the same stationary process.

    Assumptions: each window is stationary; dependence decays within one block
    length, taken as three times the integrated autocorrelation time estimated
    on the RAW window. The blocks themselves are cut from the smoothed window,
    because the smoothed window mean is the statistic being compared, and the
    ragged tail is dropped from the START so that the most recent points, the
    ones the final window exists to look at, are the ones that survive.

    Precondition: the final window must hold at least `MIN_BLOCKS_PER_RUN`
    such blocks. Where it does not, this raises `ComparisonError` instead of
    shortening the block to fit. A shortened block leaves the block means
    correlated, which the calibration suite measures running at better than
    twice the nominal error rate, and a silently miscalibrated p value is worse
    than no p value.

    Note the hypothesis this does *not* test. With one run per side there is no
    information about seed to seed variance, so a difference this test calls
    significant may be explained entirely by a different seed. The calibration
    suite measures that failure directly: against synthetic runs with realistic
    seed variance and a true effect of exactly zero, this mode reports a false
    positive rate above 70 percent. Every output carries the caveat, and the
    measured number is published rather than buried.
    """
    config = config or ComparisonConfig()
    if higher_is_better is None:
        higher_is_better = direction_for(tag, config)

    baseline_window = window_values(baseline_run.series(tag), config)
    candidate_window = window_values(candidate_run.series(tag), config)
    if baseline_window.size < 8 or candidate_window.size < 8:
        raise ComparisonError(
            f"window block mode needs at least 8 window points on {tag!r}; got "
            f"{baseline_window.size} and {candidate_window.size}"
        )

    warnings = ["single run per condition: this cannot separate a real effect from seed variance"]

    length = block_length(
        [
            raw_window_values(baseline_run.series(tag), config),
            raw_window_values(candidate_run.series(tag), config),
        ]
    )
    shortest = int(min(baseline_window.size, candidate_window.size))
    available = shortest // length
    if available < MIN_BLOCKS_PER_RUN:
        # Refusing beats reporting. Shortening the block to fit would leave the
        # block means correlated, and the calibration suite shows that exact
        # failure mode running at better than twice its nominal error rate.
        raise ComparisonError(
            f"window block mode cannot be calibrated on {tag!r}: the final window holds "
            f"{shortest} points, and at an autocorrelation time of about "
            f"{length / BLOCK_TAU_MULTIPLIER:.0f} steps that is only {available} independent "
            f"blocks, below the {MIN_BLOCKS_PER_RUN} this test requires. Log more steps, "
            f"widen the window with window_fraction, or supply seed replicates and use the "
            f"seed replicated mode instead."
        )

    baseline_blocks = _block_means(baseline_window, length)
    candidate_blocks = _block_means(candidate_window, length)
    n_baseline, n_candidate = baseline_blocks.size, candidate_blocks.size

    pooled_blocks = np.concatenate([baseline_blocks, candidate_blocks])
    rng = np.random.default_rng(config.seed)
    masks, exact = _label_permutations(pooled_blocks.size, n_baseline, config, rng)
    null_distribution = _studentized_null(pooled_blocks, masks)

    observed = _observed_statistic(pooled_blocks, n_baseline, null_distribution, exact)
    p_value = permutation_p_value(observed, null_distribution, exact)
    effect = float(candidate_blocks.mean() - baseline_blocks.mean())
    min_attainable = _min_attainable_p(n_baseline, n_candidate, exact, config)
    if min_attainable > config.alpha:
        warnings.append(
            f"the smallest attainable p value at this block count is {min_attainable:.3f}, "
            f"above alpha {config.alpha:g}"
        )

    spread = _pooled_standard_deviation(baseline_blocks, candidate_blocks)
    ci_low, ci_high = _block_bootstrap_interval(
        baseline_blocks, candidate_blocks, config, np.random.default_rng(config.seed + 1)
    )

    return ComparisonResult(
        tag=tag,
        baseline=baseline_run.run_id,
        candidate=candidate_run.run_id,
        mode=MODE_WINDOW_BLOCK,
        test_name="two sided studentized block permutation test (Welch t) on the final window mean",
        baseline_statistic=float(baseline_blocks.mean()),
        candidate_statistic=float(candidate_blocks.mean()),
        effect=effect,
        relative_effect_pct=_relative(effect, float(baseline_blocks.mean())),
        effect_size=float(effect / spread) if spread > 0 else 0.0,
        effect_size_name="Cohen's d over block means",
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="percentile bootstrap over blocks",
        ci_level=config.confidence_level,
        p_value=p_value,
        n_permutations=int(null_distribution.size),
        exact=exact,
        min_attainable_p=float(min_attainable),
        n_baseline=n_baseline,
        n_candidate=n_candidate,
        window_points=int(min(n_baseline, n_candidate) * length),
        higher_is_better=higher_is_better,
        seed=config.seed,
        baseline_runs=(baseline_run.run_id,),
        candidate_runs=(candidate_run.run_id,),
        warnings=tuple(warnings),
        block_length=int(length),
    )


# ---------------------------------------------------------- mode three: pairs


def paired_permutation(
    baseline: np.ndarray,
    candidate: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    clusters: Sequence[Hashable] | None = None,
    config: ComparisonConfig | None = None,
    higher_is_better: bool = True,
) -> ComparisonResult:
    """Paired permutation test on any statistic of a vector of outcomes.

    Null hypothesis: within each unit the two condition labels are exchangeable,
    so swapping a unit's baseline and candidate values is a draw from the same
    world. This is the right null when the two conditions were measured on the
    SAME units, which is the usual shape of an offline evaluation: the same
    fields, the same forms, the same examples, scored twice. Shuffling labels
    across units instead, as the seed replicated mode does, would throw away the
    pairing that makes such a comparison sensitive.

    Assumptions: the pairs line up positionally, and, when `clusters` is given,
    units sharing a cluster key are exchangeable only as a block. The statistic
    is any function of one vector to one number, because the quantity being
    compared is often not a mean of anything: a macro F1 over a split cannot be
    written as an average of per item numbers, and a test that assumed it could
    would be answering a different question.

    `clusters` is not optional detail. Fields on one form template share their
    markup, their locale and their author, so treating them as independent units
    counts evidence that is not there. When they are clustered, whole clusters
    swap together, and the smallest attainable p value falls out of the number
    of CLUSTERS: `2 / 2**n_clusters` under exhaustive enumeration. Twelve pairs
    in three templates can reach 0.25 and no lower, and reporting that plainly
    is more useful than a smaller number that was never earned.

    Enumeration is exhaustive when `2**n_clusters` is within
    `config.exhaustive_limit`, so the p value is exact; above that the swaps are
    sampled and the add one correction applies, as everywhere else here.

    Returns a `ComparisonResult` with mode `paired_cluster`. The direction is the
    caller's to state, because there is no tag name to infer it from.
    """
    config = config or ComparisonConfig()
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    if baseline.shape != candidate.shape:
        raise ComparisonError(
            f"paired permutation needs the two conditions to be the same length and shape, "
            f"pair by pair; got {baseline.shape} and {candidate.shape}"
        )
    n_pairs = int(baseline.shape[0]) if baseline.ndim else 0
    if n_pairs < 2:
        raise ComparisonError(f"paired permutation needs at least 2 pairs; got {n_pairs}")
    for name, values in (("baseline", baseline), ("candidate", candidate)):
        if np.issubdtype(values.dtype, np.inexact) and not bool(np.isfinite(values).all()):
            raise ComparisonError(
                f"the {name} array holds non finite values, and no permutation count can rank "
                f"a statistic computed from one; drop or repair those rows first"
            )

    cluster_index, n_clusters = _cluster_index(clusters, n_pairs)
    exhaustive = 2**n_clusters <= config.exhaustive_limit
    if exhaustive:
        # Row zero is the all false pattern, which is the data as it arrived, so
        # the observed statistic is inside its own null distribution by
        # construction rather than by a separate calculation that ought to agree.
        rows = np.arange(2**n_clusters, dtype=np.int64)
        patterns = ((rows[:, None] >> np.arange(n_clusters)) & 1).astype(bool)
    else:
        rng = np.random.default_rng(config.seed)
        patterns = rng.random((config.n_permutations, n_clusters)) < 0.5

    null_distribution = np.empty(patterns.shape[0], dtype=np.float64)
    for row in range(patterns.shape[0]):
        swapped_baseline, swapped_candidate = _swap_pairs(
            baseline, candidate, patterns[row][cluster_index]
        )
        null_distribution[row] = float(statistic(swapped_candidate)) - float(
            statistic(swapped_baseline)
        )

    baseline_statistic = float(statistic(baseline))
    candidate_statistic = float(statistic(candidate))
    observed = (
        float(null_distribution[0]) if exhaustive else candidate_statistic - baseline_statistic
    )
    p_value = permutation_p_value(observed, null_distribution, exhaustive)

    warnings: list[str] = []
    if n_clusters < n_pairs:
        warnings.append(
            f"{n_pairs} pairs fall into {n_clusters} clusters and whole clusters swap together, "
            f"so this p value rests on {n_clusters} independent units, not {n_pairs}"
        )
    min_attainable = 2.0 / 2**n_clusters if exhaustive else 1.0 / (1 + int(patterns.shape[0]))
    if min_attainable > config.alpha:
        warnings.append(
            f"with {n_clusters} clusters the smallest attainable p value is "
            f"{min_attainable:.3f}, above alpha {config.alpha:g}; add clusters before concluding"
        )

    null_spread = float(np.std(null_distribution, ddof=1)) if null_distribution.size > 1 else 0.0
    ci_low, ci_high = _cluster_bootstrap_interval(
        baseline, candidate, statistic, cluster_index, n_clusters, config
    )

    return ComparisonResult(
        tag="",
        baseline="baseline",
        candidate="candidate",
        mode=MODE_PAIRED_CLUSTER,
        test_name="two sided paired permutation test with clustered label swaps",
        baseline_statistic=baseline_statistic,
        candidate_statistic=candidate_statistic,
        effect=observed,
        relative_effect_pct=_relative(observed, baseline_statistic),
        effect_size=observed / null_spread if null_spread > 0 else 0.0,
        effect_size_name="effect in standard deviations of the permutation null",
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=(
            "percentile bootstrap over clusters"
            if clusters is not None
            else "percentile bootstrap over pairs"
        ),
        ci_level=config.confidence_level,
        p_value=p_value,
        n_permutations=int(null_distribution.size),
        exact=exhaustive,
        min_attainable_p=float(min_attainable),
        n_baseline=n_pairs,
        n_candidate=n_pairs,
        window_points=0,
        higher_is_better=higher_is_better,
        seed=config.seed,
        warnings=tuple(warnings),
    )


def _cluster_index(clusters: Sequence[Hashable] | None, n_pairs: int) -> tuple[np.ndarray, int]:
    """Map each pair to a cluster number, in order of first appearance.

    First appearance rather than sorted order, so a caller is never asked
    whether its keys are comparable. The p value does not depend on the
    numbering either way: the enumeration covers the same set of swaps.
    """
    if clusters is None:
        return np.arange(n_pairs, dtype=np.int64), n_pairs
    keys = list(clusters)
    if len(keys) != n_pairs:
        raise ComparisonError(
            f"paired permutation needs one cluster key per pair; got {len(keys)} keys "
            f"for {n_pairs} pairs"
        )
    numbering: dict[Hashable, int] = {}
    index = np.empty(n_pairs, dtype=np.int64)
    for position, key in enumerate(keys):
        index[position] = numbering.setdefault(key, len(numbering))
    return index, len(numbering)


def _swap_pairs(
    baseline: np.ndarray, candidate: np.ndarray, swap: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The two vectors with the marked pairs exchanged between them."""
    swapped_baseline = baseline.copy()
    swapped_candidate = candidate.copy()
    swapped_baseline[swap] = candidate[swap]
    swapped_candidate[swap] = baseline[swap]
    return swapped_baseline, swapped_candidate


def _cluster_bootstrap_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    cluster_index: np.ndarray,
    n_clusters: int,
    config: ComparisonConfig,
) -> tuple[float, float]:
    """Percentile interval from resampling whole clusters with replacement.

    Clusters rather than rows, for the same reason the permutation swaps them
    together: resampling rows inside a cluster would treat correlated
    observations as independent and return an interval narrower than the
    evidence supports.
    """
    rng = np.random.default_rng(config.seed + 1)
    members = [np.flatnonzero(cluster_index == number) for number in range(n_clusters)]
    draws = rng.integers(0, n_clusters, size=(config.n_bootstrap, n_clusters))
    differences = np.empty(config.n_bootstrap, dtype=np.float64)
    for row in range(config.n_bootstrap):
        index = np.concatenate([members[number] for number in draws[row]])
        differences[row] = float(statistic(candidate[index])) - float(statistic(baseline[index]))
    tail = (1.0 - config.confidence_level) / 2.0
    low, high = np.quantile(differences, [tail, 1.0 - tail])
    return float(low), float(high)


# ------------------------------------------------------------------ dispatcher


def group_by_variant(experiments: list[Experiment]) -> dict[str, list[Experiment]]:
    """Bucket runs into conditions, so seed replicates end up together."""
    grouped: dict[str, list[Experiment]] = defaultdict(list)
    for experiment in experiments:
        grouped[experiment.variant_key].append(experiment)
    return {key: sorted(runs, key=lambda r: r.run_id) for key, runs in sorted(grouped.items())}


def compare(
    baseline_runs: list[Experiment],
    candidate_runs: list[Experiment],
    tag: str,
    config: ComparisonConfig | None = None,
    higher_is_better: bool | None = None,
    baseline_name: str | None = None,
    candidate_name: str | None = None,
) -> ComparisonResult:
    """Compare two conditions, choosing the strongest mode the data supports.

    Two or more runs on each side means seed replicated mode. Anything less
    falls back to window block mode, which is labelled as the weaker claim.
    The choice is made by what is available, never by which produces the
    smaller p value.
    """
    config = config or ComparisonConfig()
    usable_baseline = [run for run in baseline_runs if run.has(tag)]
    usable_candidate = [run for run in candidate_runs if run.has(tag)]
    if not usable_baseline or not usable_candidate:
        raise ComparisonError(f"no runs carry tag {tag!r} on one or both sides")

    if len(usable_baseline) >= 2 and len(usable_candidate) >= 2:
        return compare_seed_replicated(
            usable_baseline,
            usable_candidate,
            tag,
            config,
            higher_is_better,
            baseline_name,
            candidate_name,
        )
    return compare_window_block(
        usable_baseline[0], usable_candidate[0], tag, config, higher_is_better
    )


def compare_all(
    experiments: list[Experiment],
    baseline: str,
    tags: list[str] | None = None,
    config: ComparisonConfig | None = None,
) -> list[ComparisonResult]:
    """Compare every condition against `baseline` on every shared tag.

    `baseline` may name a run or a variant key. Runs are grouped into variants
    first, so a sweep with five seeds per setting yields one strong comparison
    per setting rather than twenty five noisy pairwise ones.
    """
    config = config or ComparisonConfig()
    variants = group_by_variant(experiments)
    baseline_key = _resolve_baseline(baseline, variants, experiments)
    baseline_runs = variants[baseline_key]

    if tags is None:
        tags = sorted({tag for run in experiments for tag in run.tags})

    results: list[ComparisonResult] = []
    for variant_key, runs in variants.items():
        if variant_key == baseline_key:
            continue
        for tag in tags:
            if not any(run.has(tag) for run in baseline_runs) or not any(
                run.has(tag) for run in runs
            ):
                continue
            try:
                results.append(
                    compare(
                        baseline_runs,
                        runs,
                        tag,
                        config,
                        higher_is_better=direction_for(tag, config),
                        baseline_name=baseline_key,
                        candidate_name=variant_key,
                    )
                )
            except (ComparisonError, SeriesError):
                continue
    return results


def _resolve_baseline(
    baseline: str, variants: dict[str, list[Experiment]], experiments: list[Experiment]
) -> str:
    if baseline in variants:
        return baseline
    for run in experiments:
        if run.run_id == baseline:
            return run.variant_key
    known = ", ".join(sorted(run.run_id for run in experiments)[:8])
    raise ComparisonError(f"baseline {baseline!r} matches no run or variant; runs include: {known}")


# -------------------------------------------------------------- small helpers


def _statistics_for(runs: list[Experiment], tag: str, config: ComparisonConfig) -> np.ndarray:
    return np.array(
        [window_statistic(run.series(tag), config) for run in runs if run.has(tag)],
        dtype=np.float64,
    )


def _window_points(runs: list[Experiment], tag: str, config: ComparisonConfig) -> int:
    sizes = [
        run.series(tag).window_size(config.window_fraction, config.window_minimum)
        for run in runs
        if run.has(tag)
    ]
    return int(min(sizes)) if sizes else 0


def _relative(effect: float, baseline: float) -> float:
    if baseline == 0 or not math.isfinite(baseline):
        return 0.0
    return 100.0 * effect / abs(baseline)


def _pooled_standard_deviation(first: np.ndarray, second: np.ndarray) -> float:
    n1, n2 = first.size, second.size
    if n1 < 2 or n2 < 2:
        return float(np.std(np.concatenate([first, second]), ddof=0))
    pooled_variance = ((n1 - 1) * first.var(ddof=1) + (n2 - 1) * second.var(ddof=1)) / (n1 + n2 - 2)
    return float(math.sqrt(max(pooled_variance, 0.0)))


def _welch_interval(
    baseline: np.ndarray, candidate: np.ndarray, level: float
) -> tuple[float, float]:
    """Welch interval on the difference of the seed level means.

    The permutation test above assumes nothing about the shape of the seed level
    distribution, and the p value is the number that decides anything. This
    interval is reported alongside it for scale, and it does lean on a normal
    approximation at the seed level: each seed statistic is already a mean over
    tens of points, which is where that approximation is at its most defensible.
    The method is named in `ci_method` so nobody has to guess.
    """
    n1, n2 = baseline.size, candidate.size
    difference = float(candidate.mean() - baseline.mean())
    if n1 < 2 or n2 < 2:
        return difference, difference
    v1, v2 = baseline.var(ddof=1) / n1, candidate.var(ddof=1) / n2
    standard_error = math.sqrt(v1 + v2)
    if standard_error == 0:
        return difference, difference
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    critical = float(scipy_stats.t.ppf(0.5 + level / 2.0, df))
    return difference - critical * standard_error, difference + critical * standard_error


def _block_means(values: np.ndarray, length: int) -> np.ndarray:
    """Split a window into equal blocks, dropping the ragged tail from the start.

    Dropping rather than padding keeps every block the same weight, which the
    permutation over blocks assumes. Dropping from the START rather than the end
    is the part that matters to a reader: the remainder used to come off the
    most recent points, so a test about where a run ended up was run on
    everything except where it ended up, and the effect computed from the
    truncated blocks then disagreed with the statistics reported beside it by a
    measured 2.7 percent.
    """
    usable = (values.size // length) * length
    if usable == 0:
        return np.array([values.mean()], dtype=np.float64)
    return values[-usable:].reshape(-1, length).mean(axis=1)


def _block_bootstrap_interval(
    baseline_blocks: np.ndarray,
    candidate_blocks: np.ndarray,
    config: ComparisonConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile interval from resampling whole blocks with replacement."""
    draws_baseline = rng.integers(
        0, baseline_blocks.size, size=(config.n_bootstrap, baseline_blocks.size)
    )
    draws_candidate = rng.integers(
        0, candidate_blocks.size, size=(config.n_bootstrap, candidate_blocks.size)
    )
    differences = candidate_blocks[draws_candidate].mean(axis=1) - baseline_blocks[
        draws_baseline
    ].mean(axis=1)
    tail = (1.0 - config.confidence_level) / 2.0
    low, high = np.quantile(differences, [tail, 1.0 - tail])
    return float(low), float(high)


@dataclass
class ComparisonSet:
    """A batch of results plus the configuration that produced them."""

    results: list[ComparisonResult] = field(default_factory=list)
    config: ComparisonConfig = field(default_factory=ComparisonConfig)
    baseline: str = ""

    def tags(self) -> list[str]:
        return sorted({result.tag for result in self.results})

    def for_tag(self, tag: str) -> list[ComparisonResult]:
        return [result for result in self.results if result.tag == tag]
