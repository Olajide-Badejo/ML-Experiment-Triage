"""The model itself: window sizing, smoothing, seed grouping, config typing.

These are small behaviours, but every one of them is load bearing for a
statistic further down. A window that silently returns three points, or a
variant key that puts two seeds of the same condition in different groups,
turns into a wrong p value rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.analysis.comparison import group_by_variant
from triage.core.experiment import Experiment, MetricSeries, SeriesError


def series(values: list[float] | np.ndarray, tag: str = "loss") -> MetricSeries:
    values = np.asarray(values, dtype=np.float32)
    return MetricSeries(tag=tag, steps=np.arange(values.size, dtype=np.int64), values=values)


def test_series_are_sorted_by_step() -> None:
    unsorted = MetricSeries(
        tag="loss",
        steps=np.array([5, 1, 3], dtype=np.int64),
        values=np.array([0.5, 0.1, 0.3], dtype=np.float32),
    )
    np.testing.assert_array_equal(unsorted.steps, [1, 3, 5])
    np.testing.assert_allclose(unsorted.values, [0.1, 0.3, 0.5], rtol=1e-6)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(SeriesError, match="steps but"):
        MetricSeries(tag="loss", steps=np.arange(3), values=np.arange(5))


def test_window_takes_the_larger_of_fraction_and_minimum() -> None:
    long_run = series(np.arange(1000, dtype=np.float32))
    assert long_run.window_size(fraction=0.1, minimum=20) == 100

    short_run = series(np.arange(50, dtype=np.float32))
    assert short_run.window_size(fraction=0.1, minimum=20) == 20


def test_window_is_clipped_to_a_short_run() -> None:
    """A 12 point run cannot give 20 points, and must not pretend otherwise."""
    tiny = series(np.arange(12, dtype=np.float32))
    assert tiny.window_size(fraction=0.1, minimum=20) == 12
    assert tiny.final_window(0.1, 20).size == 12


def test_final_window_is_the_tail() -> None:
    run = series(np.arange(100, dtype=np.float32))
    np.testing.assert_array_equal(run.final_window(fraction=0.1, minimum=20), np.arange(80, 100))


def test_smoothing_preserves_length_and_mean_level() -> None:
    rng = np.random.default_rng(0)
    noisy = series(rng.normal(5.0, 1.0, 500).astype(np.float32))
    smooth = noisy.smoothed(window=9)
    assert smooth.size == 500
    assert abs(smooth.mean() - noisy.values.mean()) < 0.05
    assert smooth.std() < noisy.values.std()


def test_smoothing_a_constant_series_is_the_constant() -> None:
    flat = series(np.full(100, 3.5, dtype=np.float32))
    np.testing.assert_allclose(flat.smoothed(9), 3.5, rtol=1e-6)


def test_empty_series_raise_rather_than_return_nan() -> None:
    empty = MetricSeries(tag="loss", steps=np.array([], np.int64), values=np.array([], np.float32))
    assert empty.is_empty
    with pytest.raises(SeriesError):
        empty.final_window()
    with pytest.raises(SeriesError):
        _ = empty.final_value


def experiment(config: dict, run_id: str = "r") -> Experiment:
    return Experiment(run_id=run_id, source_path="", source_format="csv", config=config, metrics={})


def test_seed_replicates_share_a_variant_key() -> None:
    a = experiment({"learning_rate": 0.01, "batch_size": 32, "seed": 0}, "a")
    b = experiment({"learning_rate": 0.01, "batch_size": 32, "seed": 1}, "b")
    c = experiment({"learning_rate": 0.02, "batch_size": 32, "seed": 0}, "c")
    assert a.variant_key == b.variant_key
    assert a.variant_key != c.variant_key
    assert (a.seed, b.seed) == (0, 1)


def test_int_and_float_spellings_of_one_value_are_one_condition() -> None:
    """D21h: a JSON log writing 32.0 and a YAML one writing 32 are one condition.

    `repr()` spelled them `32` and `32.0`, which split a five seed condition
    into singletons that silently fell back to the weaker single run mode.
    """
    integral = experiment({"batch_size": 32, "learning_rate": 0.001, "seed": 0}, "a")
    fractional = experiment({"batch_size": 32.0, "learning_rate": 0.001, "seed": 1}, "b")
    assert integral.variant_key == fractional.variant_key
    grouped = group_by_variant([integral, fractional])
    assert len(grouped) == 1
    assert len(next(iter(grouped.values()))) == 2


def test_booleans_do_not_collapse_into_the_numbers_they_wrap() -> None:
    """`bool` is an `int` subclass, so True must not canonicalize to 1."""
    truthy = experiment({"amp": True}, "a")
    one = experiment({"amp": 1}, "b")
    assert truthy.variant_key != one.variant_key


def test_a_non_finite_config_value_still_yields_a_stable_variant_key() -> None:
    a = experiment({"learning_rate": float("nan")}, "a")
    b = experiment({"learning_rate": float("nan")}, "b")
    assert a.variant_key == b.variant_key


def test_an_explicit_variant_wins_over_the_derived_key() -> None:
    a = experiment({"variant": "control", "learning_rate": 0.01, "seed": 0})
    b = experiment({"variant": "control", "learning_rate": 0.02, "seed": 1})
    assert a.variant_key == b.variant_key == "control"


def test_a_run_without_a_seed_reports_none() -> None:
    assert experiment({"learning_rate": 0.01}).seed is None
    assert experiment({"seed": "not-a-number"}).seed is None


def test_numeric_config_excludes_seeds_and_strings() -> None:
    run = experiment(
        {"learning_rate": 0.01, "batch_size": 32, "optimizer": "adam", "seed": 4, "amp": True}
    )
    assert run.numeric_config() == {"learning_rate": 0.01, "batch_size": 32.0, "amp": 1.0}


def test_missing_tag_names_the_available_tags() -> None:
    run = Experiment(
        run_id="r",
        source_path="",
        source_format="csv",
        metrics={"train/loss": series([1.0, 2.0], "train/loss")},
    )
    assert run.has("train/loss")
    assert not run.has("val/loss")
    with pytest.raises(KeyError, match="train/loss"):
        run.series("val/loss")
