"""Calibration: does this tool's 5 percent actually mean 5 percent.

This file is the reason to trust anything else in the repository. A permutation
test is easy to write and easy to write wrongly, and a wrong one fails silently
by producing confident p values that mean nothing. The only way to know is to
run it thousands of times against data whose truth you set yourself and count
how often it is wrong.

Three measurements, all against the gates in the build specification:

* **Type I error.** Synthetic conditions with a true effect of exactly zero.
  Every rejection is a false positive. The gate is a measured rate inside
  [2, 8] percent at a nominal 5.
* **Power.** Synthetic conditions separated by a large real effect. Every
  failure to reject is a miss. The gate is above 90 percent.
* **The cost of the weak mode.** The same window block test, run on data that
  has realistic seed variance. This is not a gate, it is a measurement, and it
  is the number that justifies labelling the single run mode as weaker
  everywhere it appears.

Every case is driven from a fixed seed, so these tests are deterministic. They
are slow by design and marked `slow`; running them is the point of the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.analysis.comparison import (
    ComparisonConfig,
    ComparisonError,
    compare_seed_replicated,
    compare_window_block,
)
from triage.synthetic import CurveSpec, effect_pair, null_pair

ALPHA = 0.05
TYPE_ONE_GATE = (0.02, 0.08)
POWER_GATE = 0.90

# Seed mode is exhaustively enumerated at five seeds a side, so it is exact and
# cheap; the block mode samples, so it gets fewer cases and fewer resamples.
SEED_MODE_CASES = 3000
BLOCK_MODE_CASES = 2000
POWER_CASES = 800

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
