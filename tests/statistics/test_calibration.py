"""Calibration: does this tool's 5 percent actually mean 5 percent.

This file is the reason to trust anything else in the repository. A permutation
test is easy to write and easy to write wrongly, and a wrong one fails silently
by producing confident p values that mean nothing. The only way to know is to
run it thousands of times against data whose truth you set yourself and count
how often it is wrong.

Measurements, all against the gates in the build specification:

* **Type I error.** Synthetic conditions with a true effect of exactly zero.
  Every rejection is a false positive. The gate is a measured rate inside
  [2, 8] percent at a nominal 5.
* **Power.** Synthetic conditions separated by a large real effect. Every
  failure to reject is a miss. The gate is above 90 percent.
* **Designs that break exchangeability.** Unequal spread, unequal seed counts,
  and heavy tails, measured cell by cell rather than assumed away. These are the
  arms that justify permuting a studentized statistic instead of a raw mean
  difference, and the cell where the guarantee is weakest carries a gate of its
  own and a stated reason.
* **The paired clustered mode.** Type I on data whose pair differences are
  correlated inside a cluster, measured both with the clustering declared and
  with it ignored.
* **The cost of the weak mode.** The same window block test, run on data that
  has realistic seed variance. This is not a gate, it is a measurement, and it
  is the number that justifies labelling the single run mode as weaker
  everywhere it appears.

Every case is driven from a fixed seed, so these tests are deterministic. They
are slow by design and marked `slow`; running them is the point of the suite.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from triage.analysis.comparison import (
    ComparisonConfig,
    ComparisonError,
    compare_seed_replicated,
    compare_window_block,
    paired_permutation,
)
from triage.synthetic import (
    CurveSpec,
    clustered_null_pair,
    effect_pair,
    null_pair,
    null_pair_designed,
)

ALPHA = 0.05
TYPE_ONE_GATE = (0.02, 0.08)
POWER_GATE = 0.90

# Seed mode is exhaustively enumerated at five seeds a side, so it is exact and
# cheap; the block mode samples, so it gets fewer cases and fewer resamples.
SEED_MODE_CASES = 3000
BLOCK_MODE_CASES = 2000
POWER_CASES = 800

# One cell of the design grid, at the sample size the defect register measured
# the broken statistic over, so the before and after numbers are comparable.
DESIGN_CASES = 1200
PAIRED_CASES = 1000
PAIRED_F1_CASES = 500

# The two spreads of the heteroscedastic cells: a stable baseline and a
# candidate that moves five times as much between seeds.
NARROW_SEED_SIGMA = 0.01
WIDE_SEED_SIGMA = 0.05

# Degrees of freedom for the heavy tailed arm. Three is the lowest integer with
# a finite variance and an infinite kurtosis, which is the hardest case a test
# built on means can still be asked to be right about.
HEAVY_TAIL_DF = 3.0

# The window block mode needs a final window long enough to hold at least eight
# effectively independent blocks. At 8000 steps the window is 800 points, which
# clears that for every autocorrelation level tested here.
BLOCK_MODE_STEPS = 8000

CALIBRATION_CONFIG = ComparisonConfig(n_permutations=2000)


def progress(case: int, total: int, every: int = 200) -> None:
    """Plain `[case k/n]` lines, readable in a terminal and in a CI log."""
    if case % every == 0 or case == total:
        print(f"[case {case}/{total}]", flush=True)


def rejection_rate(p_values: list[float]) -> float:
    return float((np.asarray(p_values) < ALPHA).mean())


def wilson_halfwidth(rate: float, n: int) -> float:
    """Rough half width of the 95 percent interval, for the printed summary."""
    return 1.96 * float(np.sqrt(max(rate * (1 - rate), 1e-9) / n))


# ------------------------------------------------------- seed replicated mode


@pytest.mark.slow
def test_seed_mode_type_one_error_is_near_nominal() -> None:
    """With no true effect, the strong mode must be wrong about 5 percent of the time."""
    spec = CurveSpec(seed_sigma=0.02)
    rng = np.random.default_rng(11)
    p_values: list[float] = []
    for case in range(1, SEED_MODE_CASES + 1):
        baseline, candidate = null_pair(spec, n_seeds=5, rng=rng)
        p_values.append(
            compare_seed_replicated(baseline, candidate, "val/loss", CALIBRATION_CONFIG).p_value
        )
        progress(case, SEED_MODE_CASES)

    rate = rejection_rate(p_values)
    print(
        f"\nseed replicated mode, type I error: {rate:.4f} "
        f"(+/- {wilson_halfwidth(rate, SEED_MODE_CASES):.4f}) "
        f"over {SEED_MODE_CASES} null cases at alpha {ALPHA}"
    )
    low, high = TYPE_ONE_GATE
    assert low <= rate <= high, f"type I error {rate:.4f} outside the gate {TYPE_ONE_GATE}"


@pytest.mark.slow
def test_seed_mode_type_one_error_holds_as_seed_variance_grows() -> None:
    """The strong mode stays calibrated when seed variance dominates the signal.

    This is the property the whole design exists for. The seed offsets are
    inside the null distribution, so widening them widens the null too, and the
    error rate does not move.
    """
    measured: dict[float, float] = {}
    for seed_sigma in (0.005, 0.02, 0.05):
        spec = CurveSpec(seed_sigma=seed_sigma)
        rng = np.random.default_rng(21)
        p_values = [
            compare_seed_replicated(
                *null_pair(spec, n_seeds=5, rng=rng), "val/loss", CALIBRATION_CONFIG
            ).p_value
            for _ in range(1000)
        ]
        measured[seed_sigma] = rejection_rate(p_values)
        print(f"seed sigma {seed_sigma:.3f}: type I {measured[seed_sigma]:.4f}", flush=True)

    low, high = TYPE_ONE_GATE
    for seed_sigma, rate in measured.items():
        assert low <= rate <= high, f"seed sigma {seed_sigma}: type I {rate:.4f} outside the gate"


@pytest.mark.slow
def test_seed_mode_power_on_a_large_effect() -> None:
    """A real effect roughly three seed deviations wide must be found reliably."""
    spec = CurveSpec(seed_sigma=0.02)
    rng = np.random.default_rng(12)
    p_values: list[float] = []
    for case in range(1, POWER_CASES + 1):
        baseline, candidate = effect_pair(spec, effect=-0.06, n_seeds=5, rng=rng)
        p_values.append(
            compare_seed_replicated(baseline, candidate, "val/loss", CALIBRATION_CONFIG).p_value
        )
        progress(case, POWER_CASES, every=100)

    power = rejection_rate(p_values)
    print(f"\nseed replicated mode, power: {power:.4f} over {POWER_CASES} cases")
    assert power > POWER_GATE, f"power {power:.4f} below the gate {POWER_GATE}"


def test_seed_mode_reports_when_the_design_cannot_reach_alpha() -> None:
    """Three seeds a side can never produce p below 0.05, and must say so.

    With three and three there are twenty distinct label assignments, so the
    smallest two sided p value the design can produce is 0.1. Reporting that
    plainly is more useful than a p value that was never able to fire.
    """
    spec = CurveSpec(seed_sigma=0.02)
    rng = np.random.default_rng(31)
    result = compare_seed_replicated(
        *effect_pair(spec, effect=-0.5, n_seeds=3, rng=rng), "val/loss", CALIBRATION_CONFIG
    )
    assert result.exact
    assert result.min_attainable_p == pytest.approx(0.1)
    assert result.p_value >= 0.1
    assert any("smallest attainable p value" in warning for warning in result.warnings)


# ------------------------------------ designs that break plain exchangeability


def seed_mode_type_one(
    baseline_spec: CurveSpec,
    candidate_spec: CurveSpec,
    n_baseline: int,
    n_candidate: int,
    seed: int,
    cases: int = DESIGN_CASES,
) -> float:
    """False positive rate of the seed mode on one cell of the design grid."""
    rng = np.random.default_rng(seed)
    p_values: list[float] = []
    for case in range(1, cases + 1):
        baseline, candidate = null_pair_designed(
            baseline_spec, candidate_spec, n_baseline, n_candidate, rng
        )
        p_values.append(
            compare_seed_replicated(baseline, candidate, "val/loss", CALIBRATION_CONFIG).p_value
        )
        progress(case, cases, every=400)
    return rejection_rate(p_values)


# Three gates, because the three regimes are genuinely different, and each one
# is set so that the defect it exists to catch would fail it. What none of them
# do is bend to fit: every cell's measured rate is published in
# `triage/calibration.py` exactly as it came out, gate or no gate.
#
#: Equal spreads, whatever the counts, and heavy tails: exchangeability actually
#: holds, so the test is exact and the ordinary gate applies.
DESIGN_GATE = (0.02, 0.08)

#: Unequal spreads with at least five runs on both sides. Studentizing is
#: asymptotic in the number of runs (Janssen 1997), so five a side is close to
#: but not on nominal: measured 7.8 percent where the raw mean difference
#: measured 8.3. Neither statistic is exact here and the gate does not pretend
#: one is; it bounds how far from nominal the cell is allowed to drift.
HETEROSCEDASTIC_GATE = (0.02, 0.10)

#: Unequal spreads with fewer than five runs on one side, which is where the
#: guarantee is weakest and where the tool emits a warning of its own. The
#: variance that studentizes the statistic is itself estimated from three
#: numbers, and no amount of resampling repairs that. The gate is set to catch
#: the defect rather than to assert an exactness the design cannot deliver:
#: permuting the raw mean difference measured 17.9 percent in the worst cell and
#: fails this gate, permuting the studentized statistic measures 12.9 and passes.
HETEROSCEDASTIC_SMALL_GATE = (0.01, 0.15)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_baseline", "n_candidate", "gate"),
    [
        (5, 5, HETEROSCEDASTIC_GATE),
        (3, 7, HETEROSCEDASTIC_SMALL_GATE),
        (7, 3, HETEROSCEDASTIC_SMALL_GATE),
    ],
)
def test_seed_mode_type_one_error_under_unequal_spread(
    n_baseline: int, n_candidate: int, gate: tuple[float, float]
) -> None:
    """Unequal spread plus unequal counts, the design that broke the old test.

    The baseline is the stable condition and the candidate is the wide one, five
    times its spread, which is the asymmetry a real sweep has: the setting that
    was already trusted has been run the most times and moves the least.

    Balanced cells are exact whatever the spreads, because with equal group
    sizes the studentized statistic ranks the arrangements exactly as the mean
    difference did. The unbalanced cells are the interesting ones, and they fail
    in opposite directions when the statistic is not studentized: seven narrow
    runs against three wide ones rejected 17.9 percent of true nulls, three
    narrow against seven wide only 1.1 percent.
    """
    rate = seed_mode_type_one(
        CurveSpec(seed_sigma=NARROW_SEED_SIGMA),
        CurveSpec(seed_sigma=WIDE_SEED_SIGMA),
        n_baseline,
        n_candidate,
        seed=71 + n_baseline,
    )
    print(
        f"\nseed replicated mode, {n_baseline} narrow against {n_candidate} wide "
        f"(sigma {NARROW_SEED_SIGMA} against {WIDE_SEED_SIGMA}): type I {rate:.4f} "
        f"(+/- {wilson_halfwidth(rate, DESIGN_CASES):.4f}) over {DESIGN_CASES} null cases"
    )
    low, high = gate
    assert low <= rate <= high, f"type I error {rate:.4f} outside the gate {gate}"


@pytest.mark.slow
def test_seed_mode_type_one_error_when_only_the_counts_are_unequal() -> None:
    """Seven against three at one spread: the count asymmetry on its own.

    Run beside the heteroscedastic cells so the two causes can be told apart. An
    unbalanced design with equal spreads is exactly exchangeable, so this cell
    should sit on nominal, and if it ever does not the fault is in the
    enumeration rather than in the statistic.
    """
    spec = CurveSpec(seed_sigma=0.02)
    rate = seed_mode_type_one(spec, spec, 7, 3, seed=81)
    print(
        f"\nseed replicated mode, 7 against 3 at one spread: type I {rate:.4f} "
        f"(+/- {wilson_halfwidth(rate, DESIGN_CASES):.4f}) over {DESIGN_CASES} null cases"
    )
    low, high = DESIGN_GATE
    assert low <= rate <= high, f"type I error {rate:.4f} outside the gate {DESIGN_GATE}"


@pytest.mark.slow
def test_seed_mode_type_one_error_under_heavy_tailed_noise() -> None:
    """Student t noise with three degrees of freedom, at the same spreads.

    Training metrics are not Gaussian, and the tail is where a test that assumed
    they were falls over: a t with three degrees of freedom has finite variance
    and infinite kurtosis, so single runs land far from the mean often enough to
    matter. A permutation test should not care, because it never assumed a shape
    in the first place, and this arm is what turns that argument into a number.
    """
    spec = CurveSpec(seed_sigma=0.02, tail_df=HEAVY_TAIL_DF)
    rate = seed_mode_type_one(spec, spec, 5, 5, seed=91)
    print(
        f"\nseed replicated mode, Student t noise at {HEAVY_TAIL_DF:.0f} degrees of freedom: "
        f"type I {rate:.4f} (+/- {wilson_halfwidth(rate, DESIGN_CASES):.4f}) "
        f"over {DESIGN_CASES} null cases"
    )
    low, high = DESIGN_GATE
    assert low <= rate <= high, f"type I error {rate:.4f} outside the gate {DESIGN_GATE}"


# ----------------------------------------------------------- window block mode


@pytest.mark.slow
def test_block_mode_type_one_error_is_near_nominal() -> None:
    """On its own terms, with no seed variance, the weak mode is calibrated too.

    The null this measures is the one the mode actually tests: two windows drawn
    from the same process. It is a real guarantee, and a narrow one.
    """
    spec = CurveSpec(seed_sigma=0.0, n_steps=BLOCK_MODE_STEPS)
    rng = np.random.default_rng(13)
    p_values: list[float] = []
    refused = 0
    for case in range(1, BLOCK_MODE_CASES + 1):
        baseline, candidate = null_pair(spec, n_seeds=1, rng=rng)
        try:
            p_values.append(
                compare_window_block(
                    baseline[0], candidate[0], "val/loss", CALIBRATION_CONFIG
                ).p_value
            )
        except ComparisonError:
            refused += 1
        progress(case, BLOCK_MODE_CASES, every=100)

    rate = rejection_rate(p_values)
    print(
        f"\nwindow block mode, type I error: {rate:.4f} "
        f"(+/- {wilson_halfwidth(rate, len(p_values)):.4f}) "
        f"over {len(p_values)} null cases, {refused} refused as too short"
    )
    assert refused / BLOCK_MODE_CASES < 0.10, "the block precondition is refusing too often here"
    low, high = TYPE_ONE_GATE
    assert low <= rate <= high, f"type I error {rate:.4f} outside the gate {TYPE_ONE_GATE}"


@pytest.mark.slow
def test_block_mode_power_on_a_large_effect() -> None:
    spec = CurveSpec(seed_sigma=0.0, n_steps=BLOCK_MODE_STEPS)
    rng = np.random.default_rng(14)
    p_values = []
    for case in range(1, 300 + 1):
        baseline, candidate = effect_pair(spec, effect=-0.06, n_seeds=1, rng=rng)
        try:
            p_values.append(
                compare_window_block(
                    baseline[0], candidate[0], "val/loss", CALIBRATION_CONFIG
                ).p_value
            )
        except ComparisonError:
            continue
        progress(case, 300, every=100)

    power = rejection_rate(p_values)
    print(f"\nwindow block mode, power: {power:.4f} over {len(p_values)} cases")
    assert power > POWER_GATE, f"power {power:.4f} below the gate {POWER_GATE}"


@pytest.mark.slow
def test_block_mode_is_anticonservative_when_seed_variance_is_present() -> None:
    """Measure the cost of the weaker claim, rather than asserting it away.

    Identical setup to the type I test above, except the runs now carry a per
    run offset, which every real training run does. The true effect is still
    exactly zero, so every rejection is still a false positive. The rate this
    records is the honest reason the single run mode is labelled weaker in every
    output this tool produces.
    """
    measured: dict[float, float] = {}
    for seed_sigma in (0.01, 0.02, 0.04):
        spec = CurveSpec(seed_sigma=seed_sigma, n_steps=BLOCK_MODE_STEPS)
        rng = np.random.default_rng(41)
        p_values = []
        for _ in range(800):
            baseline, candidate = null_pair(spec, n_seeds=1, rng=rng)
            try:
                p_values.append(
                    compare_window_block(
                        baseline[0], candidate[0], "val/loss", CALIBRATION_CONFIG
                    ).p_value
                )
            except ComparisonError:
                continue
        measured[seed_sigma] = rejection_rate(p_values)
        print(
            f"seed sigma {seed_sigma:.3f}: window block false positive rate "
            f"{measured[seed_sigma]:.4f}",
            flush=True,
        )

    # The claim is not a precise number, it is that the failure is severe and
    # grows with the seed variance the mode cannot see.
    assert measured[0.01] > 0.30, "expected the weak mode to fail badly under seed variance"
    assert measured[0.04] > measured[0.01], "the failure should worsen as seed variance grows"


def test_block_mode_refuses_a_window_it_cannot_calibrate() -> None:
    """Too few independent blocks is a refusal, never a quietly wrong p value."""
    spec = CurveSpec(seed_sigma=0.0, n_steps=600, rho=0.9)
    rng = np.random.default_rng(51)
    baseline, candidate = null_pair(spec, n_seeds=1, rng=rng)
    with pytest.raises(ComparisonError, match="cannot be calibrated"):
        compare_window_block(baseline[0], candidate[0], "val/loss", CALIBRATION_CONFIG)


# -------------------------------------------------- mode three: paired clusters

#: Ten clusters enumerate 1024 swap patterns, which is exact and still cheap,
#: and puts the smallest attainable p value at 2/1024, well below alpha. Four
#: pairs to a cluster is the shape the clustering matters in: forty scored
#: units, ten of them independent.
PAIRED_CLUSTERS = 10
PAIRED_CLUSTER_SIZE = 4

#: The interval is not what these arms measure, so the bootstrap is cut to a
#: tenth of the default; the p value is untouched by it.
PAIRED_CONFIG = ComparisonConfig(n_permutations=2000, n_bootstrap=200)


def macro_f1(predictions: np.ndarray, truth: np.ndarray) -> float:
    """Macro F1 over two classes: a statistic that is not a mean of anything.

    This is the consumer's statistic, and the reason `paired_permutation` takes
    a callable at all. It cannot be written as an average over the units, so a
    test that permuted per unit differences would be answering a different
    question from the one being asked.
    """
    scores = []
    for label in (0, 1):
        predicted = predictions == label
        actual = truth == label
        true_positives = float(np.count_nonzero(predicted & actual))
        precision_denominator = float(np.count_nonzero(predicted))
        recall_denominator = float(np.count_nonzero(actual))
        precision = true_positives / precision_denominator if precision_denominator else 0.0
        recall = true_positives / recall_denominator if recall_denominator else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


@pytest.mark.slow
def test_paired_mode_type_one_error_on_clustered_null_data() -> None:
    """Declaring the clusters keeps the paired mode at its nominal rate.

    The generator gives every cluster a shift of its own, drawn from a
    distribution centred on zero, and every pair in that cluster carries it.
    The true effect is still exactly zero, so every rejection is a false
    positive, but the pairs inside a cluster are no longer independent, which is
    what a form template, an annotator or a data slice does to the units under
    it.
    """
    rng = np.random.default_rng(101)
    p_values: list[float] = []
    for case in range(1, PAIRED_CASES + 1):
        baseline, candidate, clusters = clustered_null_pair(
            PAIRED_CLUSTERS, PAIRED_CLUSTER_SIZE, rng
        )
        p_values.append(
            paired_permutation(
                baseline, candidate, statistic=np.mean, clusters=clusters, config=PAIRED_CONFIG
            ).p_value
        )
        progress(case, PAIRED_CASES, every=200)

    rate = rejection_rate(p_values)
    print(
        f"\npaired clustered mode, type I error: {rate:.4f} "
        f"(+/- {wilson_halfwidth(rate, PAIRED_CASES):.4f}) over {PAIRED_CASES} null cases, "
        f"{PAIRED_CLUSTERS} clusters of {PAIRED_CLUSTER_SIZE} pairs"
    )
    low, high = TYPE_ONE_GATE
    assert low <= rate <= high, f"type I error {rate:.4f} outside the gate {TYPE_ONE_GATE}"


@pytest.mark.slow
def test_paired_mode_is_anticonservative_when_the_clustering_is_ignored() -> None:
    """The cost of calling clustered units independent, measured on the same data.

    Identical data to the arm above, with `clusters` left unset so every pair
    swaps on its own. The shared shift inside a cluster is then read as forty
    independent pieces of evidence rather than ten, and the rate this records is
    why the argument exists and why the reports name the cluster count beside
    the pair count.

    The generator's intracluster correlation is moderate on purpose, so this
    number is a floor on the damage rather than a dramatic instance of it: a
    real form template, annotator or data slice shares more than half a standard
    deviation across the units under it.
    """
    rng = np.random.default_rng(101)
    p_values: list[float] = []
    for case in range(1, PAIRED_CASES + 1):
        baseline, candidate, _clusters = clustered_null_pair(
            PAIRED_CLUSTERS, PAIRED_CLUSTER_SIZE, rng
        )
        p_values.append(
            paired_permutation(baseline, candidate, statistic=np.mean, config=PAIRED_CONFIG).p_value
        )
        progress(case, PAIRED_CASES, every=200)

    rate = rejection_rate(p_values)
    print(
        f"\npaired mode with the clustering ignored: false positive rate {rate:.4f} "
        f"over {PAIRED_CASES} null cases"
    )
    assert rate > 0.10, "ignoring the clustering should cost at least twice the nominal rate"


@pytest.mark.slow
def test_paired_mode_type_one_error_on_a_statistic_that_is_not_a_mean() -> None:
    """The same clustered null, scored by macro F1 rather than by a mean.

    Thresholding the paired scores turns them into two sets of predictions over
    one fixed truth, which is the consumer's shape exactly. The truth is taken
    from the SUM of the two scores, which is symmetric in them, so swapping a
    pair leaves it untouched and the swap invariance the test rests on survives
    the thresholding exactly. It also makes both sets of predictions genuinely
    informative, around 0.8 macro F1, rather than two coin flips whose statistic
    could not move.

    The gate here is one sided, and deliberately. Macro F1 over forty units
    takes a few dozen distinct values, so the two conditions frequently score
    exactly the same and the p value comes out at 1; a discrete statistic cannot
    produce a uniform p value and this test is therefore conservative rather
    than exact. Conservative is the safe direction and the measured rate is
    published as it stands. What must never happen is the other direction, so
    the arm gates on the upper bound alone.
    """
    rng = np.random.default_rng(102)
    p_values: list[float] = []
    for case in range(1, PAIRED_F1_CASES + 1):
        scores_baseline, scores_candidate, clusters = clustered_null_pair(
            PAIRED_CLUSTERS, PAIRED_CLUSTER_SIZE, rng
        )
        truth = ((scores_baseline + scores_candidate) > 0).astype(np.int64)
        baseline = (scores_baseline > 0).astype(np.int64)
        candidate = (scores_candidate > 0).astype(np.int64)
        p_values.append(
            paired_permutation(
                baseline,
                candidate,
                statistic=partial(macro_f1, truth=truth),
                clusters=clusters,
                config=PAIRED_CONFIG,
            ).p_value
        )
        progress(case, PAIRED_F1_CASES, every=100)

    rate = rejection_rate(p_values)
    print(
        f"\npaired clustered mode on macro F1: type I error {rate:.4f} "
        f"(+/- {wilson_halfwidth(rate, PAIRED_F1_CASES):.4f}) over {PAIRED_F1_CASES} null cases"
    )
    _low, high = TYPE_ONE_GATE
    assert rate <= high, f"type I error {rate:.4f} above the gate {high}"


# --------------------------------------------------------- properties of the p


@pytest.mark.slow
def test_null_p_values_are_uniform_not_merely_correct_at_alpha() -> None:
    """A calibrated test is right at every threshold, not just at 0.05.

    Checking one alpha can be passed by a test that is wrong in compensating
    directions. Under the null the p value should be uniform on [0, 1], so this
    checks the rejection rate at several thresholds at once.
    """
    spec = CurveSpec(seed_sigma=0.02)
    rng = np.random.default_rng(61)
    p_values = np.array(
        [
            compare_seed_replicated(
                *null_pair(spec, n_seeds=5, rng=rng), "val/loss", CALIBRATION_CONFIG
            ).p_value
            for _ in range(1500)
        ]
    )
    for threshold in (0.05, 0.10, 0.25, 0.50):
        rate = float((p_values < threshold).mean())
        tolerance = 3.0 * np.sqrt(threshold * (1 - threshold) / p_values.size)
        print(f"threshold {threshold:.2f}: rejection rate {rate:.4f}", flush=True)
        assert abs(rate - threshold) < max(tolerance, 0.02), (
            f"rejection rate {rate:.4f} at threshold {threshold} is not uniform"
        )
