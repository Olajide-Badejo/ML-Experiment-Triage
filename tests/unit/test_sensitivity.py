"""Rank correlation of hyperparameters against metrics, and what it cannot see."""

from __future__ import annotations

import numpy as np
import pytest

from triage.analysis.comparison import ComparisonConfig
from triage.analysis.sensitivity import SensitivityReport, analyse
from triage.core.experiment import Experiment, MetricSeries

CONFIG = ComparisonConfig()


def run(run_id: str, level: float, config: dict, n: int = 300) -> Experiment:
    steps = np.arange(n, dtype=np.int64)
    return Experiment(
        run_id=run_id,
        source_path="",
        source_format="synthetic",
        config=config,
        metrics={
            "val/loss": MetricSeries(
                tag="val/loss", steps=steps, values=np.full(n, level, dtype=np.float32)
            )
        },
    )


def sweep(pairs: list[tuple[float, float]], parameter: str = "learning_rate") -> list[Experiment]:
    return [
        run(f"run{index}", level, {parameter: value, "seed": index})
        for index, (value, level) in enumerate(pairs)
    ]


def replicated(
    levels: list[tuple[float, float]], n_seeds: int, parameter: str = "learning_rate"
) -> list[Experiment]:
    """One condition per level, each with `n_seeds` seed replicates of it."""
    return [
        run(
            f"lr{value}_seed{seed}",
            level + 0.001 * seed,
            {parameter: value, "seed": seed},
        )
        for value, level in levels
        for seed in range(n_seeds)
    ]


def test_seed_replicates_are_aggregated_to_the_variant_before_correlating() -> None:
    """D15a: the unit of analysis is the condition, not the run.

    Verified before this fix on exactly this shape: 5 learning rate levels with 8
    seeds each were fed in as 40 independent points and reported
    `rho = -0.594, p = 5.3e-05, n = 40`. Eight replicates of one condition are
    not eight independent draws of the learning rate, so the p value was about
    four orders of magnitude too small. The correct unit gives n = 5, whose exact
    two sided floor over 5! pairings is 2/120 = 0.0167: no p from five levels can
    be smaller than that, whatever the seeds say.
    """
    runs = replicated([(0.001, 1.0), (0.002, 0.9), (0.004, 0.8), (0.008, 0.7), (0.016, 0.6)], 8)
    result = analyse(runs, config=CONFIG)[0]

    assert result.n_runs == 40
    assert result.n_variants == 5
    assert result.p_value is not None
    assert result.p_value >= 2.0 / 120.0 - 1e-12
    assert result.p_value == pytest.approx(2.0 / 120.0)
    assert "n variants = 5" in result.p_value_label()


def test_four_variants_get_a_correlation_but_no_p_value() -> None:
    """D15b: `spearmanr`'s t approximation returned p = 0.0 at n = 4, verified.

    Four points cannot reach any useful significance under a permutation null
    either: the smallest two sided p over 4! pairings is 2/24 = 0.083. So the
    correlation is still reported, with its n, and the p value is declined rather
    than fabricated.
    """
    runs = sweep([(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)])
    result = analyse(runs, config=CONFIG)[0]

    assert result.correlation == pytest.approx(-1.0)
    assert result.n_variants == 4
    assert result.p_value is None
    assert result.adjusted_p is None
    assert not result.significant
    assert "not reported" in result.p_value_label()
    assert "5" in result.p_value_label()


def test_the_grid_is_corrected_for_multiplicity() -> None:
    """D15c: one p value per parameter per metric is a family, and was untreated."""
    runs = [
        run(
            f"run{index}",
            level,
            {"learning_rate": lr, "batch_size": bs, "momentum": mom, "seed": index},
        )
        for index, (lr, bs, mom, level) in enumerate(
            [
                (0.001, 16, 0.1, 1.0),
                (0.002, 64, 0.9, 0.9),
                (0.004, 32, 0.5, 0.8),
                (0.008, 128, 0.2, 0.7),
                (0.016, 48, 0.7, 0.6),
                (0.032, 96, 0.3, 0.5),
            ]
        )
    ]
    results = analyse(runs, config=CONFIG)
    assert len(results) == 3
    for result in results:
        assert result.p_value is not None
        assert result.adjusted_p is not None
        assert result.adjusted_p >= result.p_value - 1e-12
    assert any(result.adjusted_p > result.p_value for result in results)


def test_a_four_point_artefact_does_not_outrank_a_corrected_finding() -> None:
    """D15d: ranking by |rho| alone put the least evidenced row at the top.

    Both correlations here are a perfect -1 or +1. One is measured over six
    variants and survives the correction; the other is measured over four, which
    cannot reach any p at all. Sorted by magnitude the four point artefact leads
    on its name; sorted by evidence it does not.
    """
    runs = []
    for index, level in enumerate([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]):
        config: dict = {"learning_rate": 0.001 * (index + 1), "seed": index}
        if index < 4:
            config["dropout"] = 0.5 - 0.1 * index
        runs.append(run(f"run{index}", level, config))

    results = analyse(runs, config=CONFIG)
    assert [result.parameter for result in results] == ["learning_rate", "dropout"]
    assert abs(results[0].correlation) == pytest.approx(1.0)
    assert abs(results[1].correlation) == pytest.approx(1.0)
    assert results[0].significant
    assert results[0].n_variants == 6
    assert not results[1].significant
    assert results[1].n_variants == 4
    assert results[1].p_value is None


def test_the_permutation_p_value_is_deterministic() -> None:
    """Nine variants is past exhaustive enumeration, so the resampling is seeded."""
    runs = replicated([(float(index), 1.0 - 0.05 * index) for index in range(9)], 2)
    first = analyse(runs, config=CONFIG)
    second = analyse(runs, config=CONFIG)
    assert [result.p_value for result in first] == [result.p_value for result in second]
    assert first[0].n_variants == 9
    assert "resamples" in first[0].test_name


def test_a_perfect_monotone_relationship_gives_rho_one() -> None:
    runs = sweep([(0.001, 1.0), (0.002, 0.9), (0.004, 0.8), (0.008, 0.7), (0.016, 0.6)])
    result = analyse(runs, config=CONFIG)[0]
    assert result.parameter == "learning_rate"
    assert result.correlation == pytest.approx(-1.0)
    assert result.strength == "strong"
    assert result.n_runs == 5
    assert result.n_distinct_values == 5


def test_a_u_shaped_relationship_is_invisible_to_rank_correlation() -> None:
    """The documented limitation, isolated.

    The metric depends on the parameter completely and symmetrically, and the
    rank correlation is nonetheless near zero. A reader who takes a small rho
    as "no effect" is reading it wrong, which is why the reports say so.
    """
    runs = sweep([(1.0, 1.0), (2.0, 0.5), (3.0, 0.2), (4.0, 0.5), (5.0, 1.0)])
    result = analyse(runs, config=CONFIG)[0]
    assert abs(result.correlation) < 0.2
    assert result.strength in {"negligible", "weak"}


def test_the_reported_n_is_the_evidence_behind_the_correlation() -> None:
    """A correlation of 1.0 over four conditions must still say "four"."""
    runs = sweep([(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)])
    result = analyse(runs, config=CONFIG)[0]
    assert result.correlation == pytest.approx(-1.0)
    assert result.n_runs == 4
    assert result.n_variants == 4
    assert "4 variants" in result.p_value_label()


def test_a_constant_parameter_is_not_reported() -> None:
    runs = [
        run(f"run{i}", level, {"learning_rate": 0.001, "weight_decay": 0.01, "seed": i})
        for i, level in enumerate([1.0, 0.9, 0.8, 0.7, 0.6])
    ]
    assert analyse(runs, config=CONFIG) == []


def test_a_two_valued_parameter_is_not_reported() -> None:
    """Two levels is an A/B comparison, which the comparison layer already does."""
    runs = sweep([(1.0, 1.0), (1.0, 0.9), (2.0, 0.5), (2.0, 0.4)])
    assert analyse(runs, config=CONFIG) == []


def test_the_seed_is_never_correlated_against() -> None:
    """Correlating a metric with the seed tests the harness, not a hyperparameter."""
    runs = [
        run(f"run{i}", float(i), {"learning_rate": 0.001 * (i + 1), "seed": i}) for i in range(5)
    ]
    assert {result.parameter for result in analyse(runs, config=CONFIG)} == {"learning_rate"}


def test_too_few_runs_produce_nothing() -> None:
    assert analyse(sweep([(1.0, 1.0), (2.0, 0.5), (3.0, 0.2)]), config=CONFIG) == []


def test_results_are_sorted_by_absolute_correlation() -> None:
    runs = [
        run(
            f"run{i}",
            level,
            {"learning_rate": lr, "batch_size": bs, "seed": i},
        )
        for i, (lr, bs, level) in enumerate(
            [
                (0.001, 16, 1.0),
                (0.002, 64, 0.9),
                (0.004, 32, 0.8),
                (0.008, 128, 0.7),
                (0.016, 48, 0.6),
            ]
        )
    ]
    results = analyse(runs, config=CONFIG)
    magnitudes = [abs(result.correlation) for result in results]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert results[0].parameter == "learning_rate"


def test_direction_reads_in_terms_of_better_and_worse() -> None:
    runs = sweep([(0.001, 1.0), (0.002, 0.9), (0.004, 0.8), (0.008, 0.7)])
    result = analyse(runs, config=CONFIG)[0]
    assert not result.higher_is_better  # val/loss
    assert "lowers the metric" in result.direction
    assert "better" in result.direction


def test_the_report_caveat_names_the_smallest_sample() -> None:
    runs = sweep([(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)])
    report = SensitivityReport(results=analyse(runs, config=CONFIG))
    assert report.strongest is not None
    assert "4 runs" in report.caveat()
    assert "not controlled effects" in report.caveat()


def test_an_empty_report_says_so_rather_than_crashing() -> None:
    report = SensitivityReport(results=[])
    assert report.strongest is None
    assert "No hyperparameter" in report.caveat()
