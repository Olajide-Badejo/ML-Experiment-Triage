"""Temperature scaling, expected and maximum calibration error.

The project's identity is calibrated uncertainty, and until this module that
meant p values only. These tests are the same discipline applied to predicted
probabilities: a known calibrated input scores near zero, a known overconfident
one scores badly and is measurably improved, and the bins really do hold equal
mass rather than equal width.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.autofill.calibration import (
    DEFAULT_BINS,
    CalibrationSummary,
    calibrate,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_bins,
)
from triage.autofill.features import featurise
from triage.autofill.generator import GeneratorConfig, generate_fields
from triage.autofill.model import TrainConfig, softmax, train


def _sharpened(rng: np.random.Generator, n: int, n_classes: int, scale: float) -> tuple:
    """Logits whose truth is drawn from the distribution they imply.

    Drawing the label FROM the model's own distribution makes `scale == 1` a
    perfectly calibrated world by construction, so any deviation the estimator
    reports at that scale is estimator noise rather than miscalibration. Scaling
    the logits then makes the model overconfident by a known amount.
    """
    base = rng.normal(0.0, 1.5, (n, n_classes))
    truth = np.array([rng.choice(n_classes, p=row) for row in softmax(base)], dtype=np.int64)
    return base * scale, truth


def test_the_default_is_fifteen_bins() -> None:
    assert DEFAULT_BINS == 15


def test_equal_mass_bins_hold_equal_mass() -> None:
    """Equal MASS, not equal width: the specification is explicit and it matters.

    Equal width bins on a classifier whose confidences pile up above 0.9 put
    almost every row in one bin and estimate the rest from nothing.
    """
    rng = np.random.default_rng(0)
    confidence = rng.beta(8.0, 1.5, 1500)
    correct = rng.random(1500) < confidence
    bins = reliability_bins(confidence, correct, n_bins=15)
    counts = [b.count for b in bins]
    assert len(bins) == 15
    assert sum(counts) == 1500
    assert max(counts) - min(counts) <= 1


def test_a_calibrated_predictor_has_almost_no_calibration_error() -> None:
    rng = np.random.default_rng(1)
    confidence = rng.uniform(0.5, 1.0, 40000)
    correct = rng.random(40000) < confidence
    assert expected_calibration_error(confidence, correct) < 0.02


def test_a_confidently_wrong_predictor_has_the_worst_possible_error() -> None:
    confidence = np.full(500, 1.0)
    correct = np.zeros(500, dtype=bool)
    assert expected_calibration_error(confidence, correct) == pytest.approx(1.0)
    assert maximum_calibration_error(confidence, correct) == pytest.approx(1.0)


def test_the_maximum_is_at_least_the_expected_error() -> None:
    rng = np.random.default_rng(2)
    confidence = rng.beta(6.0, 2.0, 3000)
    correct = rng.random(3000) < confidence**1.5
    ece = expected_calibration_error(confidence, correct)
    mce = maximum_calibration_error(confidence, correct)
    assert mce >= ece > 0


def test_the_fitted_temperature_of_an_already_calibrated_head_is_about_one() -> None:
    rng = np.random.default_rng(3)
    logits, truth = _sharpened(rng, 6000, 8, scale=1.0)
    assert fit_temperature(logits, truth) == pytest.approx(1.0, abs=0.12)


def test_an_overconfident_head_is_cooled_and_a_diffident_one_is_sharpened() -> None:
    rng = np.random.default_rng(4)
    hot, truth = _sharpened(rng, 6000, 8, scale=2.5)
    assert fit_temperature(hot, truth) > 1.5
    cold, truth_cold = _sharpened(rng, 6000, 8, scale=0.4)
    assert fit_temperature(cold, truth_cold) < 0.8


def test_the_fitted_temperature_lowers_the_likelihood_it_was_fitted_on() -> None:
    rng = np.random.default_rng(5)
    logits, truth = _sharpened(rng, 4000, 8, scale=2.5)
    fitted = fit_temperature(logits, truth)
    assert negative_log_likelihood(logits, truth, fitted) < negative_log_likelihood(
        logits, truth, 1.0
    )


def test_temperature_scaling_reduces_the_calibration_error_of_a_real_head() -> None:
    """Acceptance criterion 3, on the model this repository actually trains."""
    records = generate_fields(GeneratorConfig(n_fields=3000), seed=0)
    trained = train(
        featurise([r for r in records if r.split == "train"]),
        featurise([r for r in records if r.split == "val"]),
        TrainConfig(epochs=6, seed=0),
    )
    val = featurise([r for r in records if r.split == "val"])
    summary = calibrate(trained.model.logits(val), val.classes)
    assert summary.ece_post < summary.ece_pre
    assert summary.temperature > 0


def test_the_summary_names_both_numbers_and_the_temperature() -> None:
    rng = np.random.default_rng(6)
    logits, truth = _sharpened(rng, 3000, 8, scale=2.2)
    summary = calibrate(logits, truth)
    assert summary.ece_pre > summary.ece_post
    assert summary.mce_pre >= summary.ece_pre
    assert summary.n_bins == DEFAULT_BINS
    assert summary.n_rows == 3000
    assert len(summary.bins_pre) == len(summary.bins_post) == DEFAULT_BINS


def test_the_summary_round_trips_through_plain_data() -> None:
    """The report reads this from a JSON file rather than from a live object."""
    rng = np.random.default_rng(7)
    logits, truth = _sharpened(rng, 1200, 8, scale=2.0)
    summary = calibrate(logits, truth)
    restored = CalibrationSummary.from_dict(summary.to_dict())
    assert restored == summary


def test_the_logged_metric_names_are_the_ones_the_specification_asks_for() -> None:
    rng = np.random.default_rng(8)
    logits, truth = _sharpened(rng, 800, 8, scale=2.0)
    logged = calibrate(logits, truth).as_metrics()
    assert set(logged) == {"val/ece_pre", "val/ece_post", "val/temperature"}


def test_calibrating_nothing_is_refused_rather_than_returning_a_number() -> None:
    with pytest.raises(ValueError, match="no rows"):
        fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=np.int64))


def test_a_mismatched_truth_length_is_refused() -> None:
    with pytest.raises(ValueError, match="one label per row"):
        fit_temperature(np.zeros((4, 3)), np.zeros(3, dtype=np.int64))


def test_more_bins_than_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="more bins than rows"):
        reliability_bins(np.array([0.5, 0.6]), np.array([True, False]), n_bins=15)
