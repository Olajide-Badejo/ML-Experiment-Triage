"""Controlled synthetic training curves, where the true effect is known.

Calibration is the whole credibility argument of this project, and calibration
is only possible against data whose ground truth you set yourself. That is what
this module is for. It is part of the library rather than the examples because
the statistical test suite depends on it directly.

The curve model has four parts, each standing in for something real:

    value(t) = floor + effect
             + amplitude * exp(-t / decay)      the learning curve itself
             + seed_offset                       run to run variance
             + ar1_noise(t)                      step to step measurement noise

`seed_offset` is drawn once per run and is the piece that matters most. Run to
run variance in deep learning is routinely larger than the effects people claim
from single runs, which is the finding this tool is built around (Bouthillier
et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021).
Setting `seed_sigma` to zero produces a world where single run comparison is
valid; setting it to a realistic value produces the world we actually live in,
and the difference between the two is measurable with the calibration suite.

The noise is AR(1) rather than independent because consecutive evaluations of a
training run are correlated, and that correlation is exactly what invalidates
the naive test this project replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from triage.core.experiment import Experiment, MetricSeries


@dataclass(frozen=True)
class CurveSpec:
    """One condition's curve, before a seed is drawn.

    Attributes:
        n_steps: number of logged points.
        floor: the value the curve converges towards.
        amplitude: how far above the floor the curve starts.
        decay: steps for the gap to the floor to fall by a factor of e.
        noise_sigma: standard deviation of the stationary AR(1) noise.
        rho: AR(1) coefficient; 0 is white noise, 0.8 is a typical training log.
        seed_sigma: standard deviation of the per run offset, the seed variance.
        effect: shift applied to the floor. This is the ground truth being
            recovered, so a null study sets it to zero on every condition.
        tail_df: degrees of freedom of a Student t, or `None` for a Gaussian.
            When set, both the seed offset and the AR(1) innovations are drawn
            from a t scaled to the same standard deviation as the Gaussian it
            replaces, so only the shape of the tail changes. Three degrees of
            freedom is heavy enough to have finite variance and infinite
            kurtosis, which is the regime a mean based statistic is worst in and
            the one a calibration arm should therefore measure.
    """

    n_steps: int = 1200
    floor: float = 0.35
    amplitude: float = 2.0
    decay: float = 90.0
    noise_sigma: float = 0.03
    rho: float = 0.8
    seed_sigma: float = 0.02
    effect: float = 0.0
    tail_df: float | None = None

    def with_effect(self, effect: float) -> CurveSpec:
        return replace(self, effect=effect)


def scaled_t(
    rng: np.random.Generator, df: float, sigma: float, size: int | None = None
) -> np.ndarray | float:
    """A Student t rescaled to standard deviation `sigma`.

    The variance of a standard t is `df / (df - 2)`, so dividing by the square
    root of that leaves a draw with the requested spread and a much heavier
    tail. Matching the spread is the whole point: a heavy tailed arm that also
    changed the scale would measure two things at once and settle neither.
    """
    if df <= 2:
        raise ValueError(f"tail_df must be above 2 for the variance to exist; got {df}")
    draw = rng.standard_t(df, size)
    return draw * (sigma / np.sqrt(df / (df - 2.0)))


def ar1_noise(
    n: int,
    rho: float,
    sigma: float,
    rng: np.random.Generator,
    tail_df: float | None = None,
) -> np.ndarray:
    """Stationary AR(1) noise with the requested marginal standard deviation.

    The innovation scale is sigma * sqrt(1 - rho^2) and the first sample is
    drawn from the stationary distribution, so the series has constant variance
    `sigma^2` from the first point rather than warming up into it. Without that
    the early points would be systematically quieter, which would show up as a
    trend in any window statistic.

    `tail_df` swaps the Gaussian innovations for a t of that many degrees of
    freedom, rescaled to the same standard deviation. The recursion and the
    variance are unchanged; only the shape of the tail moves.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if sigma <= 0:
        return np.zeros(n, dtype=np.float64)
    rho = float(np.clip(rho, -0.999, 0.999))
    innovation_sigma = sigma * float(np.sqrt(1.0 - rho**2))
    if tail_df is None:
        innovations = rng.normal(0.0, innovation_sigma, n)
        first = rng.normal(0.0, sigma)
    else:
        innovations = np.asarray(scaled_t(rng, tail_df, innovation_sigma, n))
        first = float(np.asarray(scaled_t(rng, tail_df, sigma)))
    noise = np.empty(n, dtype=np.float64)
    noise[0] = first
    for index in range(1, n):
        noise[index] = rho * noise[index - 1] + innovations[index]
    return noise


def generate_curve(spec: CurveSpec, rng: np.random.Generator) -> np.ndarray:
    """One run's curve: the shared trajectory, this run's offset, and noise."""
    steps = np.arange(spec.n_steps, dtype=np.float64)
    trajectory = spec.floor + spec.effect + spec.amplitude * np.exp(-steps / spec.decay)
    if spec.seed_sigma <= 0:
        offset: float = 0.0
    elif spec.tail_df is None:
        offset = float(rng.normal(0.0, spec.seed_sigma))
    else:
        offset = float(np.asarray(scaled_t(rng, spec.tail_df, spec.seed_sigma)))
    noise = ar1_noise(spec.n_steps, spec.rho, spec.noise_sigma, rng, spec.tail_df)
    return trajectory + offset + noise


def generate_run(
    run_id: str,
    spec: CurveSpec,
    rng: np.random.Generator,
    tag: str = "val/loss",
    config: dict[str, Any] | None = None,
    extra_tags: dict[str, CurveSpec] | None = None,
) -> Experiment:
    """Wrap generated curves in an `Experiment`, ready for the analysis layer."""
    steps = np.arange(spec.n_steps, dtype=np.int64)
    metrics = {
        tag: MetricSeries(tag=tag, steps=steps, values=generate_curve(spec, rng).astype(np.float32))
    }
    for other_tag, other_spec in (extra_tags or {}).items():
        metrics[other_tag] = MetricSeries(
            tag=other_tag,
            steps=np.arange(other_spec.n_steps, dtype=np.int64),
            values=generate_curve(other_spec, rng).astype(np.float32),
        )
    return Experiment(
        run_id=run_id,
        source_path=f"synthetic://{run_id}",
        source_format="synthetic",
        config=config or {},
        metrics=metrics,
        metadata={"generator": "triage.synthetic"},
    )


def generate_condition(
    name: str,
    spec: CurveSpec,
    n_seeds: int,
    rng: np.random.Generator,
    tag: str = "val/loss",
    config: dict[str, Any] | None = None,
) -> list[Experiment]:
    """`n_seeds` independent replicates of one condition.

    Each replicate gets its own seed offset, so the spread across this list is
    the seed variance the seed replicated test has to overcome.
    """
    runs = []
    for seed in range(n_seeds):
        run_config = dict(config or {})
        run_config.update({"variant": name, "seed": seed})
        runs.append(generate_run(f"{name}_seed{seed}", spec, rng, tag=tag, config=run_config))
    return runs


def null_pair(
    spec: CurveSpec, n_seeds: int, rng: np.random.Generator, tag: str = "val/loss"
) -> tuple[list[Experiment], list[Experiment]]:
    """Two conditions with a true effect of exactly zero.

    Any comparison that calls a difference here significant is a false positive,
    which is the definition the type I error measurement uses.
    """
    return null_pair_designed(spec, spec, n_seeds, n_seeds, rng, tag)


def null_pair_designed(
    baseline_spec: CurveSpec,
    candidate_spec: CurveSpec,
    n_baseline: int,
    n_candidate: int,
    rng: np.random.Generator,
    tag: str = "val/loss",
) -> tuple[list[Experiment], list[Experiment]]:
    """A null pair whose two sides may differ in spread, in count, or in both.

    The effect is still exactly zero on both sides, so every rejection is still
    a false positive. What changes is the design around it, and the design is
    what a permutation test's exactness actually depends on. A raw mean
    difference is exact only under full exchangeability, which two conditions of
    different spread do not satisfy; the failure is worst when the smaller
    condition is the wider one, and that is the shape of the commonest real
    sweep, where the stable baseline has been run the most times. Measuring it
    needs a generator that can build the asymmetry deliberately, which is this.
    """
    return (
        generate_condition("baseline", baseline_spec.with_effect(0.0), n_baseline, rng, tag),
        generate_condition("candidate", candidate_spec.with_effect(0.0), n_candidate, rng, tag),
    )


def clustered_null_pair(
    n_clusters: int,
    cluster_size: int,
    rng: np.random.Generator,
    cluster_sigma: float = 0.5,
    item_sigma: float = 1.0,
    level_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Paired scores whose differences are correlated inside a cluster.

    Returns `(baseline, candidate, cluster_keys)`: two vectors scored on the
    SAME units, and the key that says which unit belongs with which.

    The true effect is zero, but not in the easy way. Each cluster draws a shift
    of its own from a distribution centred on zero, and every pair in that
    cluster carries it, which is what a form template, an annotator or a data
    slice does to the fields inside it. A test that treats the pairs as
    independent sees that shared shift as evidence repeated many times and
    fires far too often; a test that swaps whole clusters together sees it once,
    which is what it is. Both rates are measured against this generator, and the
    difference between them is the reason `clusters` exists at all.

    Validity of the clustered null is exact rather than approximate: the cluster
    shift is symmetric about zero and independent of the unit level, so flipping
    the sign of a whole cluster leaves the joint distribution unchanged, which
    is precisely the invariance the clustered swap enumerates.
    """
    if n_clusters < 1 or cluster_size < 1:
        raise ValueError("a clustered null needs at least one cluster of at least one pair")
    n_pairs = n_clusters * cluster_size
    keys = [f"cluster{number}" for number in range(n_clusters) for _ in range(cluster_size)]
    index = np.repeat(np.arange(n_clusters), cluster_size)
    level = rng.normal(0.0, level_sigma, n_pairs)
    shift = rng.normal(0.0, cluster_sigma, n_clusters)[index]
    difference = shift + rng.normal(0.0, item_sigma, n_pairs)
    return level - difference / 2.0, level + difference / 2.0, keys


def effect_pair(
    spec: CurveSpec,
    effect: float,
    n_seeds: int,
    rng: np.random.Generator,
    tag: str = "val/loss",
) -> tuple[list[Experiment], list[Experiment]]:
    """Two conditions separated by a known, real effect, for measuring power."""
    return (
        generate_condition("baseline", spec.with_effect(0.0), n_seeds, rng, tag),
        generate_condition("candidate", spec.with_effect(effect), n_seeds, rng, tag),
    )
