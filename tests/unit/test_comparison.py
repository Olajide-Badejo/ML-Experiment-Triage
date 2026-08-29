"""Mechanics of the comparison layer, as distinct from its calibration.

The calibration suite proves the p values mean what they say. These tests cover
everything around them: which mode gets chosen, which direction counts as an
improvement, whether the same inputs give the same answer twice, and whether
every result carries the labelling the ground rules require.
"""

from __future__ import annotations

from dataclasses import replace
from math import comb

import numpy as np
import pytest

from triage.analysis.comparison import (
    MODE_SEED_REPLICATE,
    MODE_WINDOW_BLOCK,
    ComparisonConfig,
    ComparisonError,
    compare,
    compare_all,
    compare_seed_replicated,
    compare_window_block,
    direction_for,
    group_by_variant,
    infer_direction,
    integrated_autocorrelation_time,
    permutation_p_value,
    window_statistic,
    window_values,
)
from triage.core.experiment import Experiment, MetricSeries
from triage.synthetic import CurveSpec, effect_pair, generate_condition, null_pair

CONFIG = ComparisonConfig(n_permutations=1000)


def flat_run(run_id: str, level: float, n: int = 400, tag: str = "val/loss") -> Experiment:
    steps = np.arange(n, dtype=np.int64)
    return Experiment(
        run_id=run_id,
        source_path="",
        source_format="synthetic",
        config={"variant": run_id.rsplit("_seed", 1)[0], "seed": 0},
        metrics={
            tag: MetricSeries(tag=tag, steps=steps, values=np.full(n, level, dtype=np.float32))
        },
    )


# ------------------------------------------------------------------ direction


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("train/loss", False),
        ("val/loss", False),
        ("val/accuracy", True),
        ("top1_acc", True),
        ("test/f1", True),
        ("eval/perplexity", False),
        ("val/word_error_rate", False),
        ("val/auroc", True),
        ("something_unlabelled", False),
        # D17: `map` is a metric, `mape` and `smape` are errors, and substring
        # matching read the three character fragment inside both of them.
        ("val/mape", False),
        ("val/smape", False),
        ("val/map", True),
        ("val/mAP@50", True),
        ("val/mean_absolute_percentage_error", False),
        # Token boundaries, not substrings: `accuracy_loss` is not a thing, but
        # a tag can still carry a fragment inside a longer word.
        ("train/mapper_loss", False),
        ("val/reward", True),
    ],
)
def test_direction_is_inferred_from_the_tag(tag: str, expected: bool) -> None:
    assert infer_direction(tag) is expected


def test_a_direction_in_the_config_beats_the_inference() -> None:
    """D17: the documented override has to be reachable without editing code."""
    config = ComparisonConfig(n_permutations=1000, directions={"val/loss": True})
    assert direction_for("val/loss", config) is True
    assert direction_for("val/loss", ComparisonConfig()) is False


def test_compare_all_threads_the_configured_direction(  # D17
) -> None:
    rng = np.random.default_rng(101)
    runs = generate_condition("base", CurveSpec(), 3, rng) + generate_condition(
        "alt", CurveSpec(), 3, rng
    )
    config = ComparisonConfig(n_permutations=1000, directions={"val/loss": True})
    results = compare_all(runs, baseline="base", config=config)
    assert results
    assert all(result.higher_is_better for result in results)


def test_direction_can_be_overridden() -> None:
    """A tag the table gets wrong must still be correctable by the caller."""
    rng = np.random.default_rng(1)
    baseline, candidate = effect_pair(CurveSpec(), effect=-0.1, n_seeds=4, rng=rng)
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG, higher_is_better=True)
    assert result.higher_is_better
    assert not result.improved  # a fall is now a regression rather than a win


# ------------------------------------------------------------------ statistic


def test_window_statistic_is_the_mean_of_the_tail() -> None:
    values = np.concatenate([np.full(900, 1.0), np.full(100, 5.0)]).astype(np.float32)
    series = MetricSeries(tag="val/loss", steps=np.arange(1000, dtype=np.int64), values=values)
    # The window is the last 100 points, all of them 5.0, but the 9 point
    # average reaches back over the step: the four points at the start of the
    # window each borrow some of the 1.0 that precedes it. The deficit is
    # 4 * sum(4, 3, 2, 1) / 9 = 160 / 9, spread over the 100 point window.
    expected = 5.0 - (4.0 * (4 + 3 + 2 + 1) / 9.0) / 100.0
    assert window_statistic(series, ComparisonConfig()) == pytest.approx(expected, abs=1e-6)


def test_window_statistic_of_a_flat_run_is_the_level() -> None:
    """With nothing for the smoothing to bleed across, the statistic is exact."""
    series = MetricSeries(
        tag="val/loss",
        steps=np.arange(1000, dtype=np.int64),
        values=np.full(1000, 5.0, dtype=np.float32),
    )
    assert window_statistic(series, ComparisonConfig()) == pytest.approx(5.0, abs=1e-6)


def test_autocorrelation_time_recovers_a_known_ar1() -> None:
    """An AR(1) with rho 0.8 has an integrated autocorrelation time near 9."""
    rng = np.random.default_rng(5)
    rho, n = 0.8, 20000
    noise = np.empty(n)
    noise[0] = rng.normal()
    for index in range(1, n):
        noise[index] = rho * noise[index - 1] + rng.normal(0, np.sqrt(1 - rho**2))
    assert integrated_autocorrelation_time(noise) == pytest.approx(9.0, rel=0.25)


def test_autocorrelation_time_of_white_noise_is_one() -> None:
    rng = np.random.default_rng(6)
    assert integrated_autocorrelation_time(rng.normal(size=20000)) == pytest.approx(1.0, abs=0.5)


# ----------------------------------------------------------------- p mechanics


def test_sampled_p_value_never_returns_zero() -> None:
    """Zero is a claim resampling cannot support; the smallest is 1 / (1 + B)."""
    null = np.zeros(999)
    assert permutation_p_value(10.0, null, exact=False) == pytest.approx(1 / 1000)


def test_exact_p_value_counts_the_observed_arrangement() -> None:
    null = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert permutation_p_value(2.0, null, exact=True) == pytest.approx(2 / 5)


def test_a_tie_at_the_observed_value_is_counted_at_any_scale() -> None:
    """D10: the tie tolerance has to be relative, or it stops working at scale.

    A null value that differs from the observed one only by floating point error
    IS the observed arrangement, and must be counted as extreme. At values
    around 50 that error is about 1e-14, so an absolute tolerance of 1e-15 threw
    the observed arrangement out of its own null distribution.
    """
    observed = 50.0
    null = np.array([observed * (1 - 1e-14), 0.0, -observed * (1 - 1e-14)])
    assert permutation_p_value(observed, null, exact=True) == pytest.approx(2 / 3)


def test_the_attainable_floor_at_unequal_group_sizes_is_one_over_c() -> None:
    """D11: only an equal split enumerates the sign flipped arrangement as well.

    At 2 against 5 the enumeration holds C(7, 2) = 21 arrangements and the
    complement of a 2 against 5 split is a 5 against 2 split, which is not one
    of them. The floor is therefore 1/21 = 0.0476, not the 2/21 = 0.0952 that
    was reported, and the difference decides whether a design that can clear
    alpha is instead called inconclusive.
    """
    baseline = [flat_run(f"base_seed{i}", 1.0 + 0.001 * i) for i in range(2)]
    candidate = [flat_run(f"alt_seed{i}", 2.0 + 0.001 * i) for i in range(5)]
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)

    assert result.exact
    assert result.min_attainable_p == pytest.approx(1 / comb(7, 2))
    assert result.min_attainable_p < CONFIG.alpha
    assert result.p_value == pytest.approx(1 / comb(7, 2))
    assert not any("smallest attainable" in warning for warning in result.warnings)


def test_the_attainable_floor_at_equal_group_sizes_is_two_over_c() -> None:
    """The complement of an equal split IS enumerated, so the floor doubles."""
    baseline = [flat_run(f"base_seed{i}", 1.0 + 0.001 * i) for i in range(3)]
    candidate = [flat_run(f"alt_seed{i}", 2.0 + 0.001 * i) for i in range(3)]
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)
    assert result.min_attainable_p == pytest.approx(2 / comb(6, 3))


def test_an_exact_p_value_never_falls_below_its_own_floor() -> None:
    """D10: catastrophic cancellation produced an impossible exact p of 0.0.

    The null used to reconstruct the second group as `total - first_sums` while
    the observed statistic was computed directly, so on values around 50 the two
    disagreed in the last bits and the observed arrangement could fail its own
    `>=` test. The exact floor at 3 against 5 is 1 / C(8, 3); zero is not a p
    value this design can produce, whatever the data.
    """
    rng = np.random.default_rng(2024)
    floor = 1.0 / comb(8, 3)
    smallest = 1.0
    for trial in range(60):
        # A clear separation, so the observed arrangement is the extreme one in
        # essentially every trial: that is the case where losing it shows up.
        baseline = [flat_run(f"base{trial}_seed{i}", 50.0 + rng.normal(0, 0.001)) for i in range(3)]
        candidate = [
            flat_run(f"alt{trial}_seed{i}", 50.05 + rng.normal(0, 0.001)) for i in range(5)
        ]
        result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)
        assert result.exact
        smallest = min(smallest, result.p_value)
    assert smallest >= floor - 1e-12, f"smallest exact p was {smallest}, below the floor {floor}"


# ------------------------------------------- D12: studentizing the statistic


def raw_difference_p_value(pooled: np.ndarray, n_first: int) -> float:
    """The exact two sided p value of the RAW mean difference, computed here.

    Deliberately independent of the module under test: it enumerates the same
    arrangements and counts the same way, so it pins what the p value was before
    the statistic was studentized.
    """
    from itertools import combinations

    total = pooled.size
    differences = []
    for chosen in combinations(range(total), n_first):
        mask = np.zeros(total, dtype=bool)
        mask[list(chosen)] = True
        differences.append(pooled[~mask].mean() - pooled[mask].mean())
    values = np.asarray(differences)
    observed = abs(values[0])
    return float((np.abs(values) >= observed - 1e-9 * observed).mean())


def test_the_seed_mode_permutes_a_studentized_statistic() -> None:
    """D12: the raw mean difference is exact only under full exchangeability.

    Permuting the Welch t instead (Janssen 1997) keeps the test near nominal
    when the two conditions have different spreads and different seed counts,
    which is the common 7 against 3 design.
    """
    rng = np.random.default_rng(310)
    baseline, candidate = effect_pair(CurveSpec(), effect=-0.05, n_seeds=5, rng=rng)
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)
    assert "studentized" in result.test_name
    assert "Welch" in result.test_name


def test_a_balanced_design_gives_exactly_the_p_value_it_gave_before() -> None:
    """At equal seed counts studentizing must not move a single p value.

    With n1 == n2 the Welch denominator is a decreasing function of the squared
    mean difference, so ranking arrangements by |t| ranks them exactly as |mean
    difference| did. The guarantee is worth pinning: the change is meant to fix
    unbalanced designs and to leave balanced ones alone.
    """
    rng = np.random.default_rng(311)
    baseline, candidate = effect_pair(CurveSpec(), effect=-0.05, n_seeds=5, rng=rng)
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)

    pooled = np.array(
        [window_statistic(run.series("val/loss"), CONFIG) for run in baseline + candidate]
    )
    assert result.exact
    assert result.p_value == pytest.approx(raw_difference_p_value(pooled, len(baseline)))


def test_a_small_side_beside_a_very_different_spread_is_warned_about() -> None:
    """D12: the guarantee is weakest at few seeds and unequal variance."""
    baseline = [flat_run(f"base_seed{i}", 1.0 + 0.5 * i) for i in range(5)]
    candidate = [flat_run(f"alt_seed{i}", 1.0 + 0.01 * i) for i in range(3)]
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)
    assert any("spread" in warning for warning in result.warnings)


def test_an_even_design_with_similar_spreads_carries_no_such_warning() -> None:
    rng = np.random.default_rng(312)
    baseline, candidate = null_pair(CurveSpec(), n_seeds=5, rng=rng)
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)
    assert not any("spread" in warning for warning in result.warnings)


# ------------------------------------------------------ D1: the second barrier


def poisoned_series(values: list[float], tag: str = "val/loss") -> MetricSeries:
    """A series holding non finite values, built past the constructor's filter.

    `MetricSeries.__post_init__` is the first barrier and drops these points,
    so this reaches around it with `object.__setattr__` on purpose. The point
    of the test is that the comparison layer does not depend on that barrier
    having held: a series loaded from a database written by an older version,
    or built by a caller that mutated the arrays afterwards, must still stop
    here rather than fabricate a p value of exactly zero.
    """
    array = np.asarray(values, dtype=np.float32)
    series = MetricSeries(
        tag=tag,
        steps=np.arange(array.size, dtype=np.int64),
        values=np.zeros(array.size, dtype=np.float32),
    )
    object.__setattr__(series, "values", array)
    object.__setattr__(series, "steps", np.arange(array.size, dtype=np.int64))
    return series


@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_window_statistic_refuses_a_non_finite_window(poison: float) -> None:
    series = poisoned_series([1.0] * 39 + [float(poison)])
    with pytest.raises(ComparisonError, match="non finite"):
        window_statistic(series, CONFIG)


@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_window_values_refuses_a_non_finite_window(poison: float) -> None:
    series = poisoned_series([1.0] * 39 + [float(poison)])
    with pytest.raises(ComparisonError, match="non finite"):
        window_values(series, CONFIG)


def test_a_non_finite_point_smoothing_reaches_into_the_window_is_refused() -> None:
    """The barrier is applied after smoothing, which is what carries the poison in.

    The 9 point moving average spreads one NaN over 4 points either side, so a
    NaN just outside the final window still lands inside it once smoothed. The
    check therefore sits on the smoothed window rather than on the raw values,
    and this pins that: the poisoned point is at index 196 of 200, four steps
    before the 20 point window begins.
    """
    values = [1.0] * 200
    values[196] = float(np.nan)
    with pytest.raises(ComparisonError, match="non finite"):
        window_statistic(poisoned_series(values), CONFIG)


@pytest.mark.parametrize("observed", [np.nan, np.inf, -np.inf])
def test_permutation_p_value_refuses_a_non_finite_observed(observed: float) -> None:
    """A NaN observed made every comparison False and returned an exact p of 0."""
    null = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    with pytest.raises(ComparisonError, match="non finite"):
        permutation_p_value(float(observed), null, exact=True)


def test_identical_conditions_give_a_large_p_value() -> None:
    baseline = [flat_run(f"a_seed{i}", 1.0) for i in range(5)]
    candidate = [flat_run(f"b_seed{i}", 1.0) for i in range(5)]
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)
    assert result.p_value == pytest.approx(1.0)
    assert result.effect == pytest.approx(0.0)


def test_the_same_inputs_give_the_same_answer_twice() -> None:
    """Reproducibility is a deliverable, so the seed has to actually pin it."""
    rng = np.random.default_rng(9)
    baseline, candidate = effect_pair(CurveSpec(n_steps=8000), -0.05, 1, rng)
    first = compare_window_block(baseline[0], candidate[0], "val/loss", CONFIG)
    second = compare_window_block(baseline[0], candidate[0], "val/loss", CONFIG)
    assert first.p_value == second.p_value
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


# ------------------------------------------------ D16: the window block mode


def test_the_block_length_follows_the_raw_windows_autocorrelation() -> None:
    """D16: tau was estimated on the smoothed window, which is our own filter.

    A nine point moving average makes any series look autocorrelated over about
    nine points, so on white noise tau came out near 9 instead of near 1, the
    block tripled to about 27, and the mode refused every realistically sized
    run: two 400 step runs hold a 40 point window, which is one such block. The
    autocorrelation that matters belongs to the data, so it is measured on the
    raw window.
    """
    rng = np.random.default_rng(410)
    spec = CurveSpec(n_steps=400, rho=0.0, seed_sigma=0.0)
    baseline, candidate = null_pair(spec, n_seeds=1, rng=rng)
    result = compare_window_block(baseline[0], candidate[0], "val/loss", CONFIG)
    assert result.block_length <= 6, "white noise should not need a long block"
    assert result.n_baseline >= 8


def test_every_block_statistic_comes_from_the_points_actually_used() -> None:
    """D16: the ragged tail was dropped from the front, and only from the effect.

    Truncating from the start throws away the most recent points, which are the
    ones the final window exists to look at, while `baseline_statistic` and
    `candidate_statistic` were still means of the untruncated window: the two
    disagreed with the reported effect by a measured 2.7 percent. Every number
    now comes off the same block means, and the count reported is the count
    used.
    """
    rng = np.random.default_rng(411)
    # 797 window points is prime, so the tail is ragged at every block length.
    spec = CurveSpec(n_steps=7970, rho=0.6, seed_sigma=0.0)
    baseline, candidate = null_pair(spec, n_seeds=1, rng=rng)
    result = compare_window_block(baseline[0], candidate[0], "val/loss", CONFIG)

    window = window_values(baseline[0].series("val/loss"), CONFIG)
    usable = (window.size // result.block_length) * result.block_length
    assert 0 < usable < window.size, "this fixture is meant to have a ragged tail"
    assert result.baseline_statistic == pytest.approx(window[-usable:].mean())
    assert result.baseline_statistic != pytest.approx(window[:usable].mean())
    assert result.effect == pytest.approx(result.candidate_statistic - result.baseline_statistic)
    assert result.window_points == usable


# ---------------------------------------------------------------- mode choice


def test_two_runs_a_side_selects_the_strong_mode() -> None:
    rng = np.random.default_rng(2)
    baseline, candidate = null_pair(CurveSpec(), n_seeds=4, rng=rng)
    assert compare(baseline, candidate, "val/loss", CONFIG).mode == MODE_SEED_REPLICATE


def test_one_run_a_side_falls_back_to_the_weak_mode() -> None:
    rng = np.random.default_rng(3)
    baseline, candidate = null_pair(CurveSpec(n_steps=8000), n_seeds=1, rng=rng)
    result = compare(baseline, candidate, "val/loss", CONFIG)
    assert result.mode == MODE_WINDOW_BLOCK
    assert result.is_weak_mode
    assert "WEAKER CLAIM" in result.mode_label
    assert any("seed variance" in warning for warning in result.warnings)


def test_the_weak_mode_is_never_chosen_to_get_a_smaller_p() -> None:
    """Mode is decided by what data exists, never by which answer is nicer."""
    rng = np.random.default_rng(4)
    baseline, candidate = null_pair(CurveSpec(n_steps=8000), n_seeds=5, rng=rng)
    assert compare(baseline, candidate, "val/loss", CONFIG).mode == MODE_SEED_REPLICATE


def test_seed_mode_refuses_a_single_run() -> None:
    rng = np.random.default_rng(7)
    baseline, candidate = null_pair(CurveSpec(), n_seeds=1, rng=rng)
    with pytest.raises(ComparisonError, match="at least 2 runs"):
        compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)


# --------------------------------------------------------------- effect fields


def test_a_real_improvement_is_reported_as_one() -> None:
    rng = np.random.default_rng(8)
    baseline, candidate = effect_pair(CurveSpec(), effect=-0.10, n_seeds=5, rng=rng)
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)

    assert result.improved
    assert result.effect < 0
    assert result.signed_improvement_pct > 0
    assert result.relative_effect_pct < 0
    assert result.p_value < 0.05
    assert result.ci_high < 0, "the interval should exclude no effect"
    assert result.effect_size < -1.0


def test_every_result_labels_its_p_value_with_test_and_mode() -> None:
    """Ground rule: no p value ever appears without the test that produced it."""
    rng = np.random.default_rng(10)
    baseline, candidate = null_pair(CurveSpec(), n_seeds=5, rng=rng)
    label = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG).p_value_label()
    assert "p = " in label
    assert "permutation test" in label
    assert "strong claim" in label


def test_result_serialises_with_its_derived_fields() -> None:
    rng = np.random.default_rng(13)
    baseline, candidate = null_pair(CurveSpec(), n_seeds=4, rng=rng)
    data = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG).to_dict()
    for key in ("p_value", "mode_label", "improved", "signed_improvement_pct", "p_value_label"):
        assert key in data


# ------------------------------------------------------------------- grouping


def test_runs_group_into_variants_by_everything_but_the_seed() -> None:
    rng = np.random.default_rng(14)
    runs = generate_condition("lr001", CurveSpec(), 3, rng) + generate_condition(
        "lr010", CurveSpec(), 3, rng
    )
    grouped = group_by_variant(runs)
    assert sorted(grouped) == ["lr001", "lr010"]
    assert all(len(members) == 3 for members in grouped.values())


def test_compare_all_covers_every_variant_and_tag() -> None:
    rng = np.random.default_rng(15)
    runs = (
        generate_condition("base", CurveSpec(), 4, rng)
        + generate_condition("alt_a", CurveSpec().with_effect(-0.05), 4, rng)
        + generate_condition("alt_b", CurveSpec().with_effect(0.05), 4, rng)
    )
    results = compare_all(runs, baseline="base", config=CONFIG)
    assert {result.candidate for result in results} == {"alt_a", "alt_b"}
    assert all(result.baseline == "base" for result in results)
    assert all(result.mode == MODE_SEED_REPLICATE for result in results)


def test_compare_all_gives_every_result_a_stable_join_key() -> None:
    """E4b: a caller rejoining results needed something better than id().

    `tag` is not unique once one metric is compared across several conditions,
    so a consumer holding a list of results had no way to say which row a
    finding belonged to except the identity of the object. The key is
    `variant|tag`, and it is left empty and settable for library callers who
    build results themselves.
    """
    rng = np.random.default_rng(18)
    runs = (
        generate_condition("base", CurveSpec(), 3, rng)
        + generate_condition("alt_a", CurveSpec(), 3, rng)
        + generate_condition("alt_b", CurveSpec(), 3, rng)
    )
    results = compare_all(runs, baseline="base", config=CONFIG)

    assert results
    assert all(result.key == f"{result.candidate}|{result.tag}" for result in results)
    assert len({result.key for result in results}) == len(results)
    assert all("key" in result.to_dict() for result in results)


def test_a_result_built_directly_carries_no_key_until_one_is_set() -> None:
    rng = np.random.default_rng(19)
    baseline, candidate = null_pair(CurveSpec(), n_seeds=3, rng=rng)
    result = compare_seed_replicated(baseline, candidate, "val/loss", CONFIG)
    assert result.key == ""
    assert replace(result, key="mine").key == "mine"


def test_compare_all_accepts_a_run_id_as_the_baseline() -> None:
    rng = np.random.default_rng(16)
    runs = generate_condition("base", CurveSpec(), 3, rng) + generate_condition(
        "alt", CurveSpec(), 3, rng
    )
    results = compare_all(runs, baseline="base_seed1", config=CONFIG)
    assert [result.baseline for result in results] == ["base"]


def test_a_refused_comparison_is_recorded_rather_than_dropped() -> None:
    """D8: every refusal used to be swallowed by a bare `continue`.

    Three single run conditions too short for the block mode produced "0
    comparisons" and no hint that anything had been refused, while calling
    `compare()` directly on the same runs raised the full remedy message. The
    refusals are first class records now, one per condition and tag.
    """
    rng = np.random.default_rng(20)
    runs = generate_condition("base", CurveSpec(), 3, rng) + generate_condition(
        "too_short", CurveSpec(n_steps=120), 1, rng
    )
    results = compare_all(runs, baseline="base", config=CONFIG)

    assert not [result for result in results if result.candidate == "too_short"]
    assert len(results.refusals) == 1
    refusal = results.refusals[0]
    assert refusal.variant == "too_short"
    assert refusal.tag == "val/loss"
    assert "cannot be calibrated" in refusal.reason
    assert "too_short" in refusal.describe()


def test_a_comparison_that_succeeds_records_no_refusal() -> None:
    rng = np.random.default_rng(21)
    runs = generate_condition("base", CurveSpec(), 3, rng) + generate_condition(
        "alt", CurveSpec(), 3, rng
    )
    assert compare_all(runs, baseline="base", config=CONFIG).refusals == ()


def test_an_unknown_baseline_is_a_clear_error() -> None:
    rng = np.random.default_rng(17)
    runs = generate_condition("base", CurveSpec(), 2, rng)
    with pytest.raises(ComparisonError, match="matches no run or variant"):
        compare_all(runs, baseline="does_not_exist", config=CONFIG)
