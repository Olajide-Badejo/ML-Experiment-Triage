"""The frozen import surface a real downstream project depends on.

`Autofill_audit` (dist `autofill-audit`) pins this package and imports it from
exactly one module, `evaluate/triage_bridge.py`, with an AST test on its own side
that fails if a second importer ever appears. It uses only the statistics layer:
no parsers of ours run in its pipeline except as a probe, no reports are
rendered, no CLI is invoked. That makes this list, and not the CLI, the surface
that breaking would break somebody.

The rule these tests enforce is narrow and absolute: **these names keep importing
from these exact module paths, and keep accepting the shapes the consumer builds
them with.** Behaviour underneath may change with a CHANGELOG entry the consumer
can act on; the import lines and the constructor calls may not. A test that
imported them from `triage` instead, or that reached for whatever path a refactor
had moved them to, would pass while the consumer's build failed, so every import
here is written out literally.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

# The contract, written as the consumer writes it. Any failure to import here is
# the failure the consumer would see, at the same line number they would see it.
from triage.analysis.comparison import ComparisonResult, permutation_p_value
from triage.analysis.regression import (
    VERDICT_BELOW_THRESHOLD,
    VERDICT_IMPROVEMENT,
    VERDICT_NO_CHANGE,
    VERDICT_REGRESSION,
    VERDICT_UNDERPOWERED,
    RegressionConfig,
    benjamini_hochberg,
    classify,
)
from triage.parsers import JsonlParser, ParseError

#: `module path -> names that must resolve from it`, exactly as filed.
CONTRACT = {
    "triage.analysis.comparison": ("ComparisonResult", "permutation_p_value"),
    "triage.analysis.regression": (
        "VERDICT_BELOW_THRESHOLD",
        "VERDICT_IMPROVEMENT",
        "VERDICT_NO_CHANGE",
        "VERDICT_REGRESSION",
        "VERDICT_UNDERPOWERED",
        "RegressionConfig",
        "benjamini_hochberg",
        "classify",
    ),
    "triage.parsers": ("JsonlParser", "ParseError"),
}


def a_result(
    tag: str = "macro_f1",
    p_value: float = 0.01,
    relative_pct: float = -10.0,
    mode: str = "template_clustered_paired",
    key: str = "",
) -> ComparisonResult:
    """A result shaped the way the consumer builds one: its own mode string."""
    return ComparisonResult(
        tag=tag,
        baseline="rules",
        candidate="ngram",
        mode=mode,
        test_name="paired permutation test clustered by form template",
        baseline_statistic=0.80,
        candidate_statistic=0.80 + relative_pct / 100.0,
        effect=relative_pct / 100.0,
        relative_effect_pct=relative_pct,
        effect_size=0.9,
        effect_size_name="Cohen's d",
        ci_low=-0.2,
        ci_high=-0.05,
        ci_method="cluster bootstrap",
        ci_level=0.95,
        p_value=p_value,
        n_permutations=1024,
        exact=True,
        min_attainable_p=0.002,
        n_baseline=10,
        n_candidate=10,
        window_points=0,
        higher_is_better=True,
        seed=7,
        key=key,
    )


# ------------------------------------------------------------- the import list


@pytest.mark.parametrize(
    ("module_path", "name"),
    [(path, name) for path, names in CONTRACT.items() for name in names],
)
def test_every_contracted_name_resolves_from_its_exact_module(module_path: str, name: str) -> None:
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f"{module_path}.{name} is part of the frozen contract"


def test_the_contract_is_the_size_it_was_filed_at() -> None:
    """A guard on the list itself, so a name cannot be quietly dropped from it."""
    assert sum(len(names) for names in CONTRACT.values()) == 12
    assert {ComparisonResult, RegressionConfig, JsonlParser, ParseError}
    assert callable(permutation_p_value)
    assert callable(benjamini_hochberg)
    assert callable(classify)


def test_the_verdict_strings_are_stable() -> None:
    """The consumer stores these in `analysis.json` and compares them as text."""
    assert VERDICT_REGRESSION == "regression"
    assert VERDICT_IMPROVEMENT == "improvement"
    assert VERDICT_NO_CHANGE == "no significant change"
    assert VERDICT_BELOW_THRESHOLD == "significant but below the practical threshold"
    assert VERDICT_UNDERPOWERED == "inconclusive: the design cannot reach alpha"


# ------------------------------------------------- the constructions they make


def test_the_default_construction_still_works() -> None:
    config = RegressionConfig()
    assert config.alpha == 0.05
    assert config.practical_threshold_pct == 2.0
    assert config.practical_threshold_absolute is None
    assert config.false_discovery_rate == 0.05


def test_the_keyword_construction_they_pinned_against_still_works() -> None:
    """v1.0.0's three keyword call, which must keep constructing the same gates."""
    config = RegressionConfig(alpha=0.01, practical_threshold_pct=1.0, false_discovery_rate=0.10)
    assert (config.alpha, config.practical_threshold_pct, config.false_discovery_rate) == (
        0.01,
        1.0,
        0.10,
    )
    assert "1 percent" in config.describe()


def test_the_absolute_threshold_construction_they_filed_for_works() -> None:
    """Their issue #3: a gate in the metric's own units, for latency_us and kin."""
    config = RegressionConfig(practical_threshold_pct=None, practical_threshold_absolute=50.0)
    assert config.practical_threshold_absolute == 50.0
    assert config.practical_threshold_pct is None
    assert "50" in config.describe()

    with pytest.raises(ValueError, match="exactly one"):
        RegressionConfig(practical_threshold_pct=2.0, practical_threshold_absolute=50.0)


def test_the_old_keyword_is_accepted_for_one_minor_version_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning):
        config = RegressionConfig(practical_threshold=4.0)
    assert config.practical_threshold_pct == 4.0

    # And the new spelling must not warn, or the consumer's `-W error` build
    # would fail on the fix rather than on the thing being deprecated.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RegressionConfig(practical_threshold_pct=4.0)


# --------------------------------------------------------- the calls they make


def test_classify_joins_positionally_to_the_results_it_was_given() -> None:
    """Their issue #4: the `id(finding.result)` rejoin can be retired."""
    results = [
        a_result(tag="macro_f1", p_value=0.001, relative_pct=+8.0, key="ngram|macro_f1"),
        a_result(tag="latency_us", p_value=0.4, relative_pct=+1.0, key="ngram|latency_us"),
    ]
    findings = classify(results, RegressionConfig())
    assert len(findings) == len(results)
    for finding, result in zip(findings, results, strict=True):
        assert finding.result is result
    assert [finding.result.key for finding in findings] == ["ngram|macro_f1", "ngram|latency_us"]


def test_a_result_built_with_the_consumers_own_mode_string_is_reportable() -> None:
    """Their issue #2: an unknown mode must not raise out of the reporting path."""
    result = a_result(mode="template_clustered_paired")
    assert "template_clustered_paired" in result.mode_label
    assert result.to_dict()["mode"] == "template_clustered_paired"
    assert classify([result], RegressionConfig())[0].family == "template_clustered_paired"


def test_benjamini_hochberg_still_takes_a_list_and_returns_an_array() -> None:
    adjusted = benjamini_hochberg([0.001, 0.02, 0.5], 0.05)
    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == (3,)
    assert np.all(np.diff(np.sort(adjusted)) >= 0)


def test_permutation_p_value_still_takes_observed_null_and_exact() -> None:
    null = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert permutation_p_value(2.0, null, True) == pytest.approx(1 / 5)
    assert 0.0 < permutation_p_value(0.5, null, False) <= 1.0


def test_the_jsonl_parser_and_its_error_are_still_the_probe_they_use() -> None:
    """Their bridge hands a step free outcomes file to this parser as a probe."""
    assert issubclass(ParseError, Exception)
    assert hasattr(JsonlParser(), "can_parse")
