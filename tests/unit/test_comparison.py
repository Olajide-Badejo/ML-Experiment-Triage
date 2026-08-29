"""Mechanics of the comparison layer, as distinct from its calibration.

The calibration suite proves the p values mean what they say. These tests cover
everything around them: which mode gets chosen, which direction counts as an
improvement, whether the same inputs give the same answer twice, and whether
every result carries the labelling the ground rules require.
"""

from __future__ import annotations

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


def test_compare_all_accepts_a_run_id_as_the_baseline() -> None:
    rng = np.random.default_rng(16)
    runs = generate_condition("base", CurveSpec(), 3, rng) + generate_condition(
        "alt", CurveSpec(), 3, rng
    )
    results = compare_all(runs, baseline="base_seed1", config=CONFIG)
    assert [result.baseline for result in results] == ["base"]


def test_an_unknown_baseline_is_a_clear_error() -> None:
    rng = np.random.default_rng(17)
    runs = generate_condition("base", CurveSpec(), 2, rng)
    with pytest.raises(ComparisonError, match="matches no run or variant"):
        compare_all(runs, baseline="does_not_exist", config=CONFIG)
