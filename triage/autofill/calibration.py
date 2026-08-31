"""Calibrated probabilities: temperature scaling, ECE, MCE, reliability bins.

This project's whole claim is calibrated uncertainty, and until this module that
claim covered p values only. A predicted probability is the other half of it,
and it is the half a decision layer consumes: `policy.py` weighs an expected
reward against a penalty for a wrong fill, and an uncalibrated 0.9 makes that
arithmetic wrong in a way no amount of care downstream can repair. So the
dependency is stated rather than implied: a confidence is not usable until it
has been through here.

**Temperature scaling** (Guo et al., *On Calibration of Modern Neural
Networks*, ICML 2017) is one scalar `T` dividing the logits, fitted on the
validation split by minimising negative log likelihood. One parameter is the
point: it cannot overfit a held out split of a few hundred rows, and it cannot
change a single prediction, because dividing every logit by the same positive
number leaves the argmax where it was. What it changes is how much the top
probability is worth believing.

**Equal mass bins, not equal width.** The specification says fifteen equal mass
bins and that is the load bearing word. A trained classifier's confidences pile
up near one, so equal width bins put most of the data in the top bin and
estimate the other fourteen from a handful of rows each; the resulting ECE is
dominated by bins whose accuracy is measured on nothing. Equal mass gives every
bin the same standard error.

**ECE and MCE.** ECE is the mass weighted mean gap between a bin's mean
confidence and its accuracy: the error you expect to make believing this model's
probabilities. MCE is the largest single gap: the error in the worst
neighbourhood, which is what a threshold policy will find if it exists. Both are
reported before and after, because the number that matters is the pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Fifteen, per 5.2. Named rather than defaulted inline so the report and the
#: documentation quote the same constant the estimator used.
DEFAULT_BINS = 15

#: The search interval for the temperature, as a multiplicative range. Wide
#: enough for any head this workload produces (a fitted value near either end
#: means something is wrong with the head rather than with the search), and
#: bounded so the optimiser cannot wander into a region where the exponential
#: underflows and the likelihood goes flat.
TEMPERATURE_BOUNDS = (0.05, 20.0)


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram.

    `confidence` is the mean top probability of the rows in the bin, `accuracy`
    the fraction of them that were right, and `count` how many there were. A
    well calibrated bin has the first two equal, which is the diagonal the
    diagram draws.
    """

    confidence: float
    accuracy: float
    count: int

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.confidence)

    def to_dict(self) -> dict[str, float]:
        return {
            "confidence": self.confidence,
            "accuracy": self.accuracy,
            "count": float(self.count),
        }

    @classmethod
    def from_dict(cls, row: dict[str, float]) -> ReliabilityBin:
        return cls(
            confidence=float(row["confidence"]),
            accuracy=float(row["accuracy"]),
            count=int(row["count"]),
        )


@dataclass(frozen=True)
class CalibrationSummary:
    """Everything one calibration pass measured, as plain data.

    Plain data on purpose: the HTML report reads this from a JSON file written
    by `triage autofill evaluate`, so the reporting layer never imports this
    package and the numbers in the report are the numbers that were measured
    rather than a second computation of them.
    """

    temperature: float
    ece_pre: float
    ece_post: float
    mce_pre: float
    mce_post: float
    bins_pre: tuple[ReliabilityBin, ...]
    bins_post: tuple[ReliabilityBin, ...]
    n_bins: int = DEFAULT_BINS
    n_rows: int = 0

    def as_metrics(self) -> dict[str, float]:
        """The three series names 5.2 asks the evaluation to log."""
        return {
            "val/ece_pre": self.ece_pre,
            "val/ece_post": self.ece_post,
            "val/temperature": self.temperature,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "ece_pre": self.ece_pre,
            "ece_post": self.ece_post,
            "mce_pre": self.mce_pre,
            "mce_post": self.mce_post,
            "bins_pre": [item.to_dict() for item in self.bins_pre],
            "bins_post": [item.to_dict() for item in self.bins_post],
            "n_bins": self.n_bins,
            "n_rows": self.n_rows,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationSummary:
        return cls(
            temperature=float(data["temperature"]),
            ece_pre=float(data["ece_pre"]),
            ece_post=float(data["ece_post"]),
            mce_pre=float(data["mce_pre"]),
            mce_post=float(data["mce_post"]),
            bins_pre=tuple(ReliabilityBin.from_dict(row) for row in data["bins_pre"]),
            bins_post=tuple(ReliabilityBin.from_dict(row) for row in data["bins_post"]),
            n_bins=int(data["n_bins"]),
            n_rows=int(data["n_rows"]),
        )


def _check(logits: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(truth, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"logits must be (rows, classes); got shape {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("no rows to calibrate: a temperature fitted on nothing is not a number")
    if labels.shape[0] != values.shape[0]:
        raise ValueError(
            f"one label per row: got {values.shape[0]} rows of logits and {labels.shape[0]} labels"
        )
    return values, labels


def negative_log_likelihood(logits: np.ndarray, truth: np.ndarray, temperature: float) -> float:
    """Mean NLL of the true classes under `softmax(logits / temperature)`.

    Computed through the log sum exp rather than through a softmax and a log, so
    a confident wrong prediction contributes a large finite number instead of an
    infinity from a probability that underflowed to zero.
    """
    values, labels = _check(logits, truth)
    if not temperature > 0:
        raise ValueError(f"a temperature must be above zero; got {temperature}")
    scaled = values / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1)) + scaled.max(axis=1)
    return float(np.mean(log_partition - scaled[np.arange(labels.size), labels]))


def fit_temperature(logits: np.ndarray, truth: np.ndarray) -> float:
    """The one scalar that best explains this split's outcomes.

    Fitted with scipy's bounded scalar minimiser over log temperature. Log
    because the parameter is a scale: halving and doubling should be equally far
    from one, and a linear search spends most of its evaluations above it.
    """
    from scipy.optimize import minimize_scalar

    _check(logits, truth)
    low, high = (float(np.log(bound)) for bound in TEMPERATURE_BOUNDS)
    result = minimize_scalar(
        lambda log_temperature: negative_log_likelihood(
            logits, truth, float(np.exp(log_temperature))
        ),
        bounds=(low, high),
        method="bounded",
    )
    return float(np.exp(float(result.x)))


def top_probability(logits: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """`(confidence, predicted class)` under a temperature.

    The predicted class is returned alongside because it is free here and
    because a caller computing it separately at another temperature would be
    computing the same thing twice: scaling cannot move an argmax.
    """
    if not temperature > 0:
        raise ValueError(f"a temperature must be above zero; got {temperature}")
    values = np.asarray(logits, dtype=np.float64) / temperature
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    predicted = np.asarray(values.argmax(axis=1), dtype=np.int64)
    return probabilities[np.arange(predicted.size), predicted], predicted


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = DEFAULT_BINS
) -> tuple[ReliabilityBin, ...]:
    """`n_bins` bins of equal mass, ordered by confidence.

    Ties are not broken specially: rows of identical confidence may land in
    adjacent bins, which is the honest outcome of insisting on equal mass and is
    invisible in an ECE, since two adjacent bins with the same confidence
    contribute exactly what one merged bin would.
    """
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
    hits = np.asarray(correct).reshape(-1).astype(np.float64)
    if scores.size != hits.size:
        raise ValueError(f"one outcome per confidence: got {scores.size} and {hits.size}")
    if n_bins < 1:
        raise ValueError(f"a reliability diagram needs at least one bin; got {n_bins}")
    if scores.size < n_bins:
        raise ValueError(
            f"more bins than rows ({n_bins} bins, {scores.size} rows): every bin would hold "
            f"one row or none, and its accuracy would be zero or one rather than an estimate"
        )
    order = np.argsort(scores, kind="stable")
    return tuple(
        ReliabilityBin(
            confidence=float(scores[chunk].mean()),
            accuracy=float(hits[chunk].mean()),
            count=int(chunk.size),
        )
        for chunk in np.array_split(order, n_bins)
    )


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = DEFAULT_BINS
) -> float:
    """Mass weighted mean gap between confidence and accuracy."""
    bins = reliability_bins(confidence, correct, n_bins)
    total = sum(item.count for item in bins)
    return float(sum(item.count * item.gap for item in bins) / total)


def maximum_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = DEFAULT_BINS
) -> float:
    """The largest gap in any one bin: the worst neighbourhood, not the average."""
    return float(max(item.gap for item in reliability_bins(confidence, correct, n_bins)))


def calibrate(
    logits: np.ndarray, truth: np.ndarray, n_bins: int = DEFAULT_BINS
) -> CalibrationSummary:
    """Fit the temperature on these rows and measure the error either side of it.

    Fitted and measured on the SAME split, which is what post hoc temperature
    scaling is: the validation split is held out of training and is what the
    scalar is for. A single parameter fitted on several hundred rows does not
    meaningfully overfit them, and the alternative, a third split, costs training
    data to estimate one number.
    """
    values, labels = _check(logits, truth)
    temperature = fit_temperature(values, labels)

    before, predicted = top_probability(values, 1.0)
    after, _ = top_probability(values, temperature)
    correct = predicted == labels

    return CalibrationSummary(
        temperature=temperature,
        ece_pre=expected_calibration_error(before, correct, n_bins),
        ece_post=expected_calibration_error(after, correct, n_bins),
        mce_pre=maximum_calibration_error(before, correct, n_bins),
        mce_post=maximum_calibration_error(after, correct, n_bins),
        bins_pre=reliability_bins(before, correct, n_bins),
        bins_post=reliability_bins(after, correct, n_bins),
        n_bins=n_bins,
        n_rows=int(values.shape[0]),
    )
