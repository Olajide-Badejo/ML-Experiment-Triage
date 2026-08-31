"""The two gates, the false discovery correction, and the severity ranking.

The correction is checked against the worked example from Benjamini and
Hochberg's own paper rather than against whatever this implementation happens to
produce, which is the only way a multiplicity correction test is worth writing.
"""

from __future__ import annotations

import numpy as np
import pytest

from triage.analysis.comparison import (
    MODE_SEED_REPLICATE,
    MODE_WINDOW_BLOCK,
    ComparisonResult,
)
from triage.analysis.regression import (
    FAMILY_INADMISSIBLE,
    VERDICT_BELOW_THRESHOLD,
    VERDICT_IMPROVEMENT,
    VERDICT_NO_CHANGE,
    VERDICT_REGRESSION,
    VERDICT_UNDERPOWERED,
    RegressionConfig,
    TriageReport,
    benjamini_hochberg,
    classify,
    rank,
    severity_of,
)


def result(
    candidate: str = "candidate",
    tag: str = "val/loss",
    p_value: float = 0.01,
    relative_pct: float = -10.0,
    higher_is_better: bool = False,
    mode: str = MODE_SEED_REPLICATE,
    min_attainable_p: float = 0.002,
    baseline_statistic: float = 1.0,
    effect: float | None = None,
) -> ComparisonResult:
    effect = relative_pct / 100.0 if effect is None else effect
    return ComparisonResult(
        tag=tag,
        baseline="baseline",
        candidate=candidate,
        mode=mode,
        test_name="two sided permutation test on the final window mean",
        baseline_statistic=baseline_statistic,
        candidate_statistic=baseline_statistic + effect,
        effect=effect,
        relative_effect_pct=relative_pct,
        effect_size=1.5,
        effect_size_name="Cohen's d",
        ci_low=-0.2,
        ci_high=-0.05,
        ci_method="Welch t interval",
        ci_level=0.95,
        p_value=p_value,
        n_permutations=252,
        exact=True,
        min_attainable_p=min_attainable_p,
        n_baseline=5,
        n_candidate=5,
        window_points=600,
        higher_is_better=higher_is_better,
        seed=1,
    )


# ------------------------------------------------------- Benjamini Hochberg


def test_correction_matches_the_original_paper() -> None:
    """The 15 p values from Benjamini and Hochberg 1995, section 5.

    At a 5 percent false discovery rate the procedure rejects the four smallest
    of these, which is the published result. Anything that reproduces that has
    the step up direction and the rank scaling right.
    """
    p_values = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000,
    ]  # fmt: skip
    adjusted = benjamini_hochberg(p_values, false_discovery_rate=0.05)
    assert int((adjusted < 0.05).sum()) == 4

    # Bonferroni at the same level would reject only three, which is the power
    # this project trades away by not using it.
    assert int((np.asarray(p_values) * len(p_values) < 0.05).sum()) == 3


def test_adjusted_values_are_monotone_and_never_below_the_raw() -> None:
    p_values = [0.001, 0.02, 0.03, 0.5, 0.9]
    adjusted = benjamini_hochberg(p_values)
    assert np.all(adjusted >= np.asarray(p_values) - 1e-12)
    assert list(adjusted) == sorted(adjusted)
    assert np.all(adjusted <= 1.0)


def test_correction_of_a_single_test_changes_nothing() -> None:
    assert benjamini_hochberg([0.031]) == pytest.approx([0.031])


def test_correction_of_nothing_is_nothing() -> None:
    assert benjamini_hochberg([]).size == 0


def test_a_weak_mode_p_value_does_not_inflate_the_strong_family() -> None:
    """D13a: two claim strengths were pooled into one Benjamini Hochberg family.

    The weak mode fires on 53 to 87 percent of comparisons under seed variance
    alone, so its p values in a shared denominator destroy the FDR control of
    every strong result beside them. Corrected separately, the seed replicated
    finding here is a discovery; pooled, its adjusted p would be 0.02 * 5 = 0.10
    and it would not be.
    """
    results = [result(candidate="strong", tag="a", p_value=0.02, relative_pct=+12.0)] + [
        result(
            candidate=f"weak{index}",
            tag=f"w{index}",
            p_value=0.9,
            relative_pct=+12.0,
            mode=MODE_WINDOW_BLOCK,
        )
        for index in range(4)
    ]
    findings = classify(results, RegressionConfig())

    assert findings[0].family == MODE_SEED_REPLICATE
    assert findings[0].adjusted_p == pytest.approx(0.02)
    assert findings[0].verdict == VERDICT_REGRESSION
    assert {finding.family for finding in findings[1:]} == {MODE_WINDOW_BLOCK}


def test_an_inadmissible_comparison_is_excluded_from_the_denominator() -> None:
    """D13b: a design that can never reach alpha cannot be a discovery.

    Counting four such comparisons in the denominator only costs the admissible
    ones power. They are judged inconclusive either way, so they are held out and
    reported as their own family instead.
    """
    results = [result(candidate="real", tag="a", p_value=0.02, relative_pct=+12.0)] + [
        result(
            candidate=f"tiny{index}",
            tag=f"t{index}",
            p_value=0.5,
            relative_pct=+12.0,
            min_attainable_p=0.5,
        )
        for index in range(4)
    ]
    findings = classify(results, RegressionConfig())

    assert findings[0].adjusted_p == pytest.approx(0.02)
    assert findings[0].verdict == VERDICT_REGRESSION
    assert [finding.family for finding in findings[1:]] == [FAMILY_INADMISSIBLE] * 4
    assert all(finding.verdict == VERDICT_UNDERPOWERED for finding in findings[1:])
    assert not any(finding.passes_statistical_gate for finding in findings[1:])


def test_the_report_names_the_families_and_holds_the_inadmissible_apart() -> None:
    findings = classify(
        [
            result(candidate="strong", tag="a", p_value=0.02, relative_pct=+12.0),
            result(candidate="weak", tag="b", p_value=0.02, relative_pct=+12.0,
                   mode=MODE_WINDOW_BLOCK),
            result(candidate="tiny", tag="c", p_value=0.5, relative_pct=+12.0,
                   min_attainable_p=0.5),
        ],
        RegressionConfig(),
    )  # fmt: skip
    report = TriageReport(findings=findings, baseline="baseline")
    assert [finding.candidate for finding in report.inadmissible] == ["tiny"]
    note = report.family_note()
    assert MODE_SEED_REPLICATE in note
    assert MODE_WINDOW_BLOCK in note
    assert "1 excluded as inadmissible" in note


# ------------------------------------------------------------- the two gates


def test_both_gates_must_pass_for_a_regression() -> None:
    findings = classify([result(p_value=0.001, relative_pct=+12.0)], RegressionConfig())
    assert findings[0].verdict == VERDICT_REGRESSION
    assert findings[0].passes_statistical_gate
    assert findings[0].passes_practical_gate


def test_a_significant_but_tiny_effect_is_not_a_regression() -> None:
    """The whole reason for a practical gate: significance is not importance."""
    findings = classify([result(p_value=0.0001, relative_pct=+0.4)], RegressionConfig())
    assert findings[0].verdict == VERDICT_BELOW_THRESHOLD
    assert findings[0].passes_statistical_gate
    assert not findings[0].passes_practical_gate


def test_a_large_but_uncertain_effect_is_not_a_regression() -> None:
    findings = classify([result(p_value=0.4, relative_pct=+30.0)], RegressionConfig())
    assert findings[0].verdict == VERDICT_NO_CHANGE
    assert not findings[0].passes_statistical_gate


def test_the_false_discovery_rate_is_the_operative_gate() -> None:
    """D2: the field was inert. The gate compared the adjusted p against alpha.

    Verified before this fix: `--fdr 0.001`, `0.10` and `0.90` produced byte
    identical verdicts, because `false_discovery_rate` was passed to a procedure
    whose output does not depend on it and then never used again.
    """
    strict = classify(
        [result(p_value=0.03, relative_pct=+12.0)],
        RegressionConfig(false_discovery_rate=0.01),
    )
    lenient = classify(
        [result(p_value=0.03, relative_pct=+12.0)],
        RegressionConfig(false_discovery_rate=0.10),
    )
    assert strict[0].verdict == VERDICT_NO_CHANGE
    assert not strict[0].passes_statistical_gate
    assert lenient[0].verdict == VERDICT_REGRESSION
    assert lenient[0].passes_statistical_gate


def test_the_default_false_discovery_rate_is_five_percent() -> None:
    """The default was moved from 10 to 5 percent so the observed behaviour holds.

    Making the field operative at 10 percent would have loosened every existing
    verdict, including a consumer's. At 5 percent the decision at defaults is the
    same one the old alpha gate made, so nothing shifts under anybody.
    """
    config = RegressionConfig()
    assert config.false_discovery_rate == 0.05
    assert "5%" in config.describe()
    assert "--fdr" in config.describe()


def test_alpha_bounds_admissibility_and_nothing_else() -> None:
    """A tighter alpha no longer silences an admissible finding."""
    findings = classify(
        [result(p_value=0.03, relative_pct=+12.0, min_attainable_p=0.002)],
        RegressionConfig(alpha=0.01),
    )
    assert findings[0].verdict == VERDICT_REGRESSION
    assert "admissibility" in RegressionConfig().describe()


def test_the_practical_threshold_is_configurable() -> None:
    strict = classify([result(p_value=0.001, relative_pct=+3.0)], RegressionConfig())
    lenient = classify(
        [result(p_value=0.001, relative_pct=+3.0)],
        RegressionConfig(practical_threshold_pct=10.0),
    )
    assert strict[0].verdict == VERDICT_REGRESSION
    assert lenient[0].verdict == VERDICT_BELOW_THRESHOLD


# ------------------------------------------------- the practical gate, E2/D18


def test_exactly_one_practical_threshold_may_be_set() -> None:
    """E2, as the consumer filed it: one gate, stated once, never inferred."""
    with pytest.raises(ValueError, match="exactly one"):
        RegressionConfig(practical_threshold_pct=2.0, practical_threshold_absolute=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        RegressionConfig(practical_threshold_pct=None, practical_threshold_absolute=None)

    relative = RegressionConfig()
    absolute = RegressionConfig(practical_threshold_pct=None, practical_threshold_absolute=0.5)
    assert relative.practical_threshold_pct == 2.0
    assert absolute.practical_threshold_absolute == 0.5


def test_describe_prints_the_threshold_in_force() -> None:
    assert "2 percent" in RegressionConfig().describe()
    absolute = RegressionConfig(practical_threshold_pct=None, practical_threshold_absolute=0.5)
    assert "0.5" in absolute.describe()
    assert "percent" not in absolute.describe().split("relative")[-1]


def test_an_absolute_threshold_saves_a_regression_on_a_zero_baseline() -> None:
    """D18: a relative gate divides by the baseline, and a baseline can be zero.

    `_relative` returns 0.0 rather than an infinity when the baseline mean is
    zero or non finite, which is the right thing for the arithmetic and the wrong
    thing for the verdict: a real half unit regression on a metric that crosses
    zero was silently downgraded to "below the practical threshold". Stated in
    the metric's own units it is a regression, which is what it is.
    """
    zero_crossing = result(p_value=0.001, relative_pct=0.0, baseline_statistic=0.0, effect=+0.5)
    downgraded = classify([zero_crossing], RegressionConfig())
    assert downgraded[0].verdict == VERDICT_BELOW_THRESHOLD

    judged = classify(
        [zero_crossing],
        RegressionConfig(practical_threshold_pct=None, practical_threshold_absolute=0.1),
    )
    assert judged[0].verdict == VERDICT_REGRESSION
    assert judged[0].passes_practical_gate

    below = classify(
        [zero_crossing],
        RegressionConfig(practical_threshold_pct=None, practical_threshold_absolute=2.0),
    )
    assert below[0].verdict == VERDICT_BELOW_THRESHOLD


def test_the_old_practical_threshold_keyword_still_works_and_warns() -> None:
    """One minor version of grace for the keyword the CLI flag is spelled after."""
    with pytest.warns(DeprecationWarning, match="practical_threshold_pct"):
        config = RegressionConfig(practical_threshold=10.0)
    assert config.practical_threshold_pct == 10.0
    assert config.practical_threshold_absolute is None
    assert "10 percent" in config.describe()


def test_direction_decides_regression_from_improvement() -> None:
    worse_loss = classify([result(p_value=0.001, relative_pct=+12.0)], RegressionConfig())
    better_loss = classify([result(p_value=0.001, relative_pct=-12.0)], RegressionConfig())
    assert worse_loss[0].verdict == VERDICT_REGRESSION
    assert better_loss[0].verdict == VERDICT_IMPROVEMENT

    worse_accuracy = classify(
        [result(tag="val/accuracy", p_value=0.001, relative_pct=-12.0, higher_is_better=True)],
        RegressionConfig(),
    )
    assert worse_accuracy[0].verdict == VERDICT_REGRESSION


def test_an_underpowered_design_is_called_inconclusive_not_negative() -> None:
    """Three seeds a side cannot reach 0.05, and must not be read as "no change"."""
    findings = classify(
        [result(p_value=0.10, relative_pct=+25.0, min_attainable_p=0.10)], RegressionConfig()
    )
    assert findings[0].verdict == VERDICT_UNDERPOWERED


def test_the_explanation_names_both_gates() -> None:
    finding = classify([result(p_value=0.0001, relative_pct=+0.4)], RegressionConfig())[0]
    explanation = finding.explain()
    assert "statistical gate passed" in explanation
    assert "practical gate failed" in explanation


# -------------------------------------------------------------- the ranking


def test_classify_returns_findings_in_input_order() -> None:
    """E4a: the join a caller makes is positional, so the order has to be theirs.

    `classify` used to end `return rank(findings)`, which silently reordered the
    list against the results that went in. `Finding.tag` is not unique once one
    metric is compared across several conditions, so a consumer rejoining
    findings to results had no key left but `id(finding.result)`. Input order
    plus `ComparisonResult.key` retires that workaround.
    """
    results = [
        result(candidate="quiet", tag="a", p_value=0.8, relative_pct=+0.1),
        result(candidate="worse", tag="b", p_value=0.001, relative_pct=+20.0),
        result(candidate="better", tag="c", p_value=0.001, relative_pct=-15.0),
    ]
    findings = classify(results, RegressionConfig())
    assert [finding.candidate for finding in findings] == ["quiet", "worse", "better"]
    assert [finding.result for finding in findings] == results


def test_rank_is_how_a_caller_asks_for_severity_order() -> None:
    findings = classify(
        [
            result(candidate="quiet", p_value=0.8, relative_pct=+0.1),
            result(candidate="better", p_value=0.001, relative_pct=-15.0),
            result(candidate="worse", p_value=0.001, relative_pct=+20.0),
        ],
        RegressionConfig(),
    )
    assert [finding.candidate for finding in rank(findings)] == ["worse", "better", "quiet"]


def test_regressions_lead_then_improvements_then_the_rest() -> None:
    findings = rank(
        classify(
            [
                result(candidate="quiet", p_value=0.8, relative_pct=+0.1),
                result(candidate="better", p_value=0.001, relative_pct=-15.0),
                result(candidate="worse", p_value=0.001, relative_pct=+20.0),
            ],
            RegressionConfig(),
        )
    )
    assert [finding.candidate for finding in findings] == ["worse", "better", "quiet"]


def test_bigger_regressions_outrank_smaller_ones() -> None:
    findings = rank(
        classify(
            [
                result(candidate="small", tag="a", p_value=0.001, relative_pct=+5.0),
                result(candidate="large", tag="b", p_value=0.001, relative_pct=+40.0),
            ],
            RegressionConfig(),
        )
    )
    assert [finding.candidate for finding in findings] == ["large", "small"]


def test_a_weak_mode_finding_does_not_outrank_an_equal_strong_one() -> None:
    """A weaker claim about the world should not lead the table over a stronger one."""
    findings = rank(
        classify(
            [
                result(candidate="single_run", tag="a", p_value=0.001, relative_pct=+20.0,
                       mode=MODE_WINDOW_BLOCK),
                result(candidate="replicated", tag="b", p_value=0.001, relative_pct=+20.0),
            ],
            RegressionConfig(),
        )
    )  # fmt: skip
    assert [finding.candidate for finding in findings] == ["replicated", "single_run"]


def test_a_big_regression_at_a_modest_p_outranks_a_tiny_certain_one() -> None:
    """D19: the "tie breaker" was not a tie breaker, it was the ranking.

    The confidence factor spanned 1 to 13, so it could overwhelm the harm term it
    was meant to order within. Measured before the cap: a 3 percent regression at
    p = 1e-12 scored 39.00 against 23.98 for a 10 percent regression at p = 0.04,
    which is the docstring's stated intent exactly inverted. Both here clear the
    gate; the question is only which one a reader is pointed at first.
    """
    findings = rank(
        classify(
            [
                result(candidate="ten_percent", tag="a", p_value=0.04, relative_pct=+10.0),
                result(candidate="three_percent", tag="b", p_value=1e-12, relative_pct=+3.0),
            ],
            RegressionConfig(),
        )
    )
    assert [finding.verdict for finding in findings] == [VERDICT_REGRESSION] * 2
    assert [finding.candidate for finding in findings] == ["ten_percent", "three_percent"]


def test_the_confidence_factor_never_doubles_the_harm() -> None:
    """Certainty can break a tie between similar harms. It cannot outrank harm."""
    harmful = result(relative_pct=+10.0)
    assert severity_of(harmful, 1e-12) == pytest.approx(20.0)
    assert severity_of(harmful, 0.001) == pytest.approx(20.0)
    assert severity_of(harmful, 0.5) < 20.0


# ----------------------------------------------------------------- the report


def test_the_report_summarises_and_names_the_best_candidate() -> None:
    findings = classify(
        [
            result(candidate="good", tag="a", p_value=0.001, relative_pct=-8.0),
            result(candidate="best", tag="b", p_value=0.001, relative_pct=-22.0),
            result(candidate="bad", tag="c", p_value=0.001, relative_pct=+11.0),
        ],
        RegressionConfig(),
    )
    report = TriageReport(findings=findings, baseline="baseline")
    assert report.best_candidate == "best"
    assert len(report.regressions) == 1
    assert len(report.improvements) == 2
    assert "1 regressions, 2 improvements" in report.summary()


def test_the_best_candidate_prefers_a_seed_replicated_improvement() -> None:
    """D19: `best_candidate` could crown a claim the tool labels as weaker.

    A single run improvement is not comparable to a seed replicated one of any
    size, so the strong claims are considered first and the weaker ones only when
    there are no strong ones to name.
    """
    findings = classify(
        [
            result(candidate="weak_big", tag="a", p_value=0.001, relative_pct=-30.0,
                   mode=MODE_WINDOW_BLOCK),
            result(candidate="strong_small", tag="b", p_value=0.001, relative_pct=-10.0),
        ],
        RegressionConfig(),
    )  # fmt: skip
    report = TriageReport(findings=findings, baseline="baseline")
    assert report.best_candidate == "strong_small"
    assert report.best_finding is not None
    assert not report.best_finding.result.is_weak_mode


def test_the_best_candidate_falls_back_to_the_weaker_mode_and_stays_labelled() -> None:
    findings = classify(
        [
            result(candidate="only_weak", p_value=0.001, relative_pct=-30.0,
                   mode=MODE_WINDOW_BLOCK),
        ],
        RegressionConfig(),
    )  # fmt: skip
    report = TriageReport(findings=findings, baseline="baseline")
    assert report.best_candidate == "only_weak"
    assert report.best_finding is not None
    assert report.best_finding.result.is_weak_mode


def test_a_report_with_no_findings_names_no_best_candidate() -> None:
    assert TriageReport(findings=[], baseline="baseline").best_candidate is None
    assert classify([], RegressionConfig()) == []
