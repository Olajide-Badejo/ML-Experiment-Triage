"""The generators the calibration arms measure against.

A calibration suite is only worth its data. If the synthetic world is not the
world the arm claims to be measuring, every number downstream of it is a
confident measurement of the wrong thing, so the generators get tests of their
own: the heavy tail keeps the spread it replaces, the Gaussian path is
untouched by the option existing, and the clustered null really does correlate
the pairs inside a cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.synthetic import CurveSpec, ar1_noise, clustered_null_pair, generate_curve, scaled_t


def test_the_heavy_tail_keeps_the_standard_deviation_it_replaces() -> None:
    """Only the shape of the tail changes; the spread is matched by construction.

    A heavy tailed arm that also widened the distribution would measure two
    things at once and settle neither, so this is the property that makes the
    arm interpretable.
    """
    rng = np.random.default_rng(3)
    draw = np.asarray(scaled_t(rng, df=3.0, sigma=0.05, size=200_000))
    assert float(np.std(draw)) == pytest.approx(0.05, rel=0.05)
    # Infinite kurtosis is the point: a Gaussian sample of this size sits near 3.
    kurtosis = float(np.mean((draw / np.std(draw)) ** 4))
    assert kurtosis > 6.0


def test_a_tail_of_two_degrees_of_freedom_is_refused() -> None:
    """At two degrees of freedom the variance does not exist to be matched."""
    with pytest.raises(ValueError, match="above 2"):
        scaled_t(np.random.default_rng(0), df=2.0, sigma=1.0)


def test_the_gaussian_path_is_untouched_by_the_option_existing() -> None:
    """The default curve must be bit identical to what it was before `tail_df`.

    Every committed verdict in the repository was generated through this path,
    and the reproducibility check compares them to four decimal places, so a
    changed draw order here would surface as a mystery three modules away.
    """
    spec = CurveSpec(n_steps=64, seed_sigma=0.02)
    curve = generate_curve(spec, np.random.default_rng(7))

    rng = np.random.default_rng(7)
    steps = np.arange(spec.n_steps, dtype=np.float64)
    trajectory = spec.floor + spec.amplitude * np.exp(-steps / spec.decay)
    offset = rng.normal(0.0, spec.seed_sigma)
    expected = trajectory + offset + ar1_noise(spec.n_steps, spec.rho, spec.noise_sigma, rng)
    assert np.array_equal(curve, expected)


def test_the_heavy_tailed_curve_differs_from_the_gaussian_one() -> None:
    """The option is not silently inert: setting it changes the draw."""
    spec = CurveSpec(n_steps=64, seed_sigma=0.02)
    gaussian = generate_curve(spec, np.random.default_rng(7))
    heavy = generate_curve(
        CurveSpec(n_steps=64, seed_sigma=0.02, tail_df=3.0), np.random.default_rng(7)
    )
    assert not np.allclose(gaussian, heavy)


def test_the_clustered_null_correlates_the_pairs_inside_a_cluster() -> None:
    """Pair differences share a shift within a cluster and nothing between them.

    Measured as the variance of the cluster mean differences against the
    variance of the differences themselves: independent pairs would put the
    first at roughly the second divided by the cluster size, and the shared
    shift puts it far above that.
    """
    rng = np.random.default_rng(11)
    baseline, candidate, keys = clustered_null_pair(60, 5, rng, cluster_sigma=0.5, item_sigma=1.0)
    assert baseline.shape == candidate.shape == (300,)
    assert len(keys) == 300 and len(set(keys)) == 60

    differences = (candidate - baseline).reshape(60, 5)
    within = float(np.var(differences))
    between = float(np.var(differences.mean(axis=1)))
    assert between > 2.0 * within / 5.0
    # The true effect is zero, so the mean difference is small beside its spread.
    assert abs(float(differences.mean())) < 0.2


def test_the_clustered_null_refuses_an_empty_design() -> None:
    with pytest.raises(ValueError, match="at least one cluster"):
        clustered_null_pair(0, 4, np.random.default_rng(0))
