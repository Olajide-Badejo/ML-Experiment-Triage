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

**Two modes, answering two different questions.**

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
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
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

# Block length as a multiple of the estimated autocorrelation time, and the
# fewest blocks per run the window block mode will accept. Both numbers were
# set by measurement rather than taste: see `block_length` and the calibration
# suite in `tests/statistics`.
BLOCK_TAU_MULTIPLIER = 3.0
MIN_BLOCKS_PER_RUN = 8

MODE_LABELS = {
    MODE_SEED_REPLICATE: "seed replicated permutation test (strong claim)",
    MODE_WINDOW_BLOCK: (
        "single run window block permutation test (WEAKER CLAIM: cannot see seed to seed variance)"
    ),
}


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
        return MODE_LABELS[self.mode]

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


def _finite_window(series: MetricSeries, config: ComparisonConfig) -> np.ndarray:
    """The smoothed final window, or `ComparisonError` if it is not all finite.

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
    smoothed = series.smoothed(config.smoothing_window)
    size = series.window_size(config.window_fraction, config.window_minimum)
    window = smoothed[-size:]
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

    Two details here were both forced by the calibration suite, and both cost a
    measured type I error of about 12 percent against a nominal 5 before they
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


def _difference_null(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Difference of group means for every arrangement, second group minus first.

    Both group sums are formed directly rather than one being reconstructed as
    `total - first_sums`. The subtraction looks free and is not: on values whose
    mean is far from zero it cancels away most of the significant digits, and
    the null it produced then disagreed with a directly computed observed
    statistic by more than the tie tolerance, so the observed arrangement was
    counted out of its own null distribution and the exact p value came back as
    an impossible 0.0.
    """
    n_first = int(masks[0].sum())
    n_second = values.size - n_first
    first_sums = masks @ values
    second_sums = (~masks) @ values
    return second_sums / n_second - first_sums / n_first


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
    return float(_difference_null(values, identity)[0])


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

    pooled = np.concatenate([baseline_stats, candidate_stats])
    rng = np.random.default_rng(config.seed)
    masks, exact = _label_permutations(pooled.size, n_baseline, config, rng)
    null_distribution = _difference_null(pooled, masks)

    # The observed statistic comes out of the same expression as every null
    # value, from the identity arrangement, so that in exact mode it IS
    # `null_distribution[0]` rather than a number that merely ought to equal it.
    observed = _observed_statistic(pooled, n_baseline, null_distribution, exact)
    p_value = permutation_p_value(observed, null_distribution, exact)

    # The smallest p value this design can produce at all. With three seeds per
    # condition that is 0.1, so no result can ever clear alpha 0.05, and saying
    # so is more useful than reporting a p value that was never able to fire.
    arrangements = comb(pooled.size, n_baseline)
    min_attainable = 2.0 / arrangements if exact else 1.0 / (1 + config.n_permutations)
    if min_attainable > config.alpha:
        warnings.append(
            f"with {n_baseline} and {n_candidate} seeds the smallest attainable p value is "
            f"{min_attainable:.3f}, above alpha {config.alpha:g}; add seeds before concluding"
        )

    spread = _pooled_standard_deviation(baseline_stats, candidate_stats)
    effect_size = observed / spread if spread > 0 else 0.0
    ci_low, ci_high = _welch_interval(baseline_stats, candidate_stats, config.confidence_level)

    return ComparisonResult(
        tag=tag,
        baseline=baseline_name,
        candidate=candidate_name,
        mode=MODE_SEED_REPLICATE,
        test_name="two sided permutation test on the final window mean",
        baseline_statistic=float(baseline_stats.mean()),
        candidate_statistic=float(candidate_stats.mean()),
        effect=observed,
        relative_effect_pct=_relative(observed, float(baseline_stats.mean())),
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
    length, taken as three times the estimated integrated autocorrelation time.

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

    length = block_length([baseline_window, candidate_window])
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
    null_distribution = _difference_null(pooled_blocks, masks)

    observed = _observed_statistic(pooled_blocks, n_baseline, null_distribution, exact)
    p_value = permutation_p_value(observed, null_distribution, exact)
    arrangements = comb(pooled_blocks.size, n_baseline)
    min_attainable = 2.0 / arrangements if exact else 1.0 / (1 + config.n_permutations)
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
        test_name="two sided block permutation test on the final window mean",
        baseline_statistic=float(baseline_window.mean()),
        candidate_statistic=float(candidate_window.mean()),
        effect=observed,
        relative_effect_pct=_relative(observed, float(baseline_window.mean())),
        effect_size=float(observed / spread) if spread > 0 else 0.0,
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
        window_points=int(min(baseline_window.size, candidate_window.size)),
        higher_is_better=higher_is_better,
        seed=config.seed,
        baseline_runs=(baseline_run.run_id,),
        candidate_runs=(candidate_run.run_id,),
        warnings=tuple(warnings),
        block_length=int(length),
    )


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
    """Split a window into equal blocks, dropping the ragged tail.

    Dropping rather than padding keeps every block the same weight, which the
    permutation over blocks assumes.
    """
    usable = (values.size // length) * length
    if usable == 0:
        return np.array([values.mean()], dtype=np.float64)
    return values[:usable].reshape(-1, length).mean(axis=1)


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
