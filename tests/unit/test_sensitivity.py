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


def test_the_reported_n_is_the_number_of_runs_behind_the_correlation() -> None:
    """A correlation of 1.0 over four runs must still say "four"."""
    runs = sweep([(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)])
    result = analyse(runs, config=CONFIG)[0]
    assert result.correlation == pytest.approx(-1.0)
    assert result.n_runs == 4
    assert "n = 4" in result.p_value_label()


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
