"""The paired permutation test, and the open mode vocabulary it needs.

Two conditions measured on the SAME units are not two independent samples, and
shuffling their labels across units throws away the pairing that made the
comparison sensitive in the first place. The right null here is that within each
unit the two labels are exchangeable, which is a label swap rather than a
reshuffle.

The clustered form exists because the units are often not independent either.
Fields on one form template share almost everything, so a swap has to apply to
the whole template at once or the test counts evidence it does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.analysis.comparison import (
    MODE_PAIRED_CLUSTER,
    ComparisonConfig,
    ComparisonError,
    ComparisonResult,
    paired_permutation,
)

CONFIG = ComparisonConfig(n_permutations=999, n_bootstrap=200)


def mean_statistic(values: np.ndarray) -> float:
    return float(np.mean(values))


def macro_f1(correct: np.ndarray) -> float:
    """A statistic that is not the mean of anything, which is the point.

    Nothing here is a real macro F1; what matters is that it is a function of
    the whole vector rather than an average of per item numbers, because that is
    the shape the consumer's statistic has and the reason the API takes a
    callable at all.
    """
    values = np.asarray(correct, dtype=np.float64)
    hits = float(values.sum())
    return hits / (hits + 0.5 * (values.size - hits) + 1e-12)


# ------------------------------------------------------- the open vocabulary


def unknown_mode_result(mode: str = "template_clustered_paired") -> ComparisonResult:
    return ComparisonResult(
        tag="correct",
        baseline="rules",
        candidate="ngram",
        mode=mode,
        test_name="a test this version has never heard of",
        baseline_statistic=0.5,
        candidate_statistic=0.6,
        effect=0.1,
        relative_effect_pct=20.0,
        effect_size=1.0,
        effect_size_name="whatever the caller uses",
        ci_low=0.0,
        ci_high=0.2,
        ci_method="caller supplied",
        ci_level=0.95,
        p_value=0.01,
        n_permutations=1024,
        exact=True,
        min_attainable_p=0.002,
        n_baseline=10,
        n_candidate=10,
        window_points=0,
        higher_is_better=True,
        seed=1,
    )


def test_an_unknown_mode_still_has_a_label() -> None:
    """E3: the mode table was a closed vocabulary that raised KeyError.

    A caller writing a truthful mode string of its own could then call neither
    `mode_label` nor `to_dict`, which is every reporting path there is.
    """
    result = unknown_mode_result()
    assert "template_clustered_paired" in result.mode_label
    assert "template_clustered_paired" in result.p_value_label()
    assert result.to_dict()["mode_label"] == result.mode_label
    assert not result.is_weak_mode


def test_a_known_mode_keeps_its_registered_label() -> None:
    assert "clustered" in unknown_mode_result(MODE_PAIRED_CLUSTER).mode_label


# --------------------------------------------------------- the paired test


def test_a_paired_comparison_of_few_pairs_is_exhaustive() -> None:
    rng = np.random.default_rng(1)
    baseline = rng.normal(size=8)
    candidate = baseline + rng.normal(0.0, 0.01, size=8)
    result = paired_permutation(baseline, candidate, mean_statistic, config=CONFIG)

    assert result.mode == MODE_PAIRED_CLUSTER
    assert result.exact
    assert result.n_permutations == 2**8
    assert result.min_attainable_p == pytest.approx(2 / 2**8)
    assert result.n_baseline == result.n_candidate == 8


def test_a_consistent_paired_gain_reaches_the_floor() -> None:
    """Every pair moving the same way is the most extreme arrangement there is."""
    baseline = np.arange(10, dtype=np.float64)
    candidate = baseline + 1.0
    result = paired_permutation(baseline, candidate, mean_statistic, config=CONFIG)
    assert result.p_value == pytest.approx(2 / 2**10)
    assert result.effect == pytest.approx(1.0)
    assert result.improved


def test_a_paired_comparison_of_noise_is_not_significant() -> None:
    rng = np.random.default_rng(2)
    baseline = rng.normal(size=12)
    candidate = rng.normal(size=12)
    assert paired_permutation(baseline, candidate, mean_statistic, config=CONFIG).p_value > 0.05


def test_clusters_swap_together_and_set_the_attainable_floor() -> None:
    """E3: the unit of resampling is the cluster, and the floor says so.

    Twelve pairs in three templates carry three independent swaps, not twelve,
    so the smallest p value the design can reach is 2/8 and not 2/4096.
    Reporting the larger number is the whole point: it is the one that is true.
    """
    rng = np.random.default_rng(3)
    baseline = rng.normal(size=12)
    candidate = baseline + 0.5
    clusters = ["form_a"] * 4 + ["form_b"] * 4 + ["form_c"] * 4
    result = paired_permutation(baseline, candidate, mean_statistic, clusters, CONFIG)

    assert result.exact
    assert result.n_permutations == 2**3
    assert result.min_attainable_p == pytest.approx(2 / 8)
    assert result.p_value == pytest.approx(2 / 8)
    assert any("cluster" in warning for warning in result.warnings)


def test_the_p_value_does_not_depend_on_the_order_of_the_input() -> None:
    """Property: relabelling the rows cannot change the answer."""
    rng = np.random.default_rng(4)
    baseline = rng.normal(size=10)
    candidate = baseline + rng.normal(0.1, 0.3, size=10)
    clusters = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])

    straight = paired_permutation(baseline, candidate, mean_statistic, clusters, CONFIG)
    order = rng.permutation(10)
    shuffled = paired_permutation(
        baseline[order], candidate[order], mean_statistic, clusters[order], CONFIG
    )
    assert shuffled.p_value == pytest.approx(straight.p_value)
    assert shuffled.effect == pytest.approx(straight.effect)


def test_many_clusters_fall_back_to_sampling_with_the_add_one_correction() -> None:
    rng = np.random.default_rng(5)
    baseline = rng.normal(size=40)
    candidate = baseline + 1.0
    result = paired_permutation(baseline, candidate, mean_statistic, config=CONFIG)

    assert not result.exact
    assert result.n_permutations == CONFIG.n_permutations
    assert result.p_value == pytest.approx(1 / (1 + CONFIG.n_permutations))
    assert result.min_attainable_p == pytest.approx(1 / (1 + CONFIG.n_permutations))


def test_a_statistic_that_is_not_a_mean_is_permuted_correctly() -> None:
    """The consumer's statistic is macro F1 over a whole split, hence a callable."""
    rng = np.random.default_rng(6)
    baseline = (rng.random(16) < 0.5).astype(np.float64)
    candidate = np.minimum(baseline + (rng.random(16) < 0.6), 1.0)
    result = paired_permutation(baseline, candidate, macro_f1, config=CONFIG)

    assert result.baseline_statistic == pytest.approx(macro_f1(baseline))
    assert result.candidate_statistic == pytest.approx(macro_f1(candidate))
    assert result.effect == pytest.approx(macro_f1(candidate) - macro_f1(baseline))
    assert 0 < result.p_value <= 1


def test_the_direction_is_the_callers_to_state() -> None:
    baseline = np.arange(10, dtype=np.float64)
    candidate = baseline + 1.0
    result = paired_permutation(
        baseline, candidate, mean_statistic, config=CONFIG, higher_is_better=False
    )
    assert not result.improved


@pytest.mark.parametrize(
    ("baseline", "candidate", "clusters", "message"),
    [
        (np.arange(5.0), np.arange(4.0), None, "same length"),
        (np.arange(1.0), np.arange(1.0), None, "at least 2 pairs"),
        (np.arange(5.0), np.arange(5.0), ["a", "b"], "one cluster key per pair"),
    ],
)
def test_a_malformed_paired_input_is_refused_with_a_reason(
    baseline: np.ndarray, candidate: np.ndarray, clusters: list[str] | None, message: str
) -> None:
    with pytest.raises(ComparisonError, match=message):
        paired_permutation(baseline, candidate, mean_statistic, clusters, CONFIG)


def test_a_non_finite_input_is_refused_rather_than_ranked() -> None:
    baseline = np.array([1.0, 2.0, 3.0, np.nan])
    candidate = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ComparisonError, match="non finite"):
        paired_permutation(baseline, candidate, mean_statistic, config=CONFIG)
