"""Which of these differences should anyone actually act on.

A p value alone is a bad trigger for a regression alert, for two reasons that
pull in opposite directions.

**Too many alerts.** With enough data every difference becomes statistically
significant, including a 0.05 percent change nobody would act on. So a finding
must clear a *practical* gate as well as a statistical one: a relative effect of
at least `practical_threshold_pct`, which is a visible, configurable constant
rather than a number buried in the code.

**Too many false alerts.** Testing eight metrics against a baseline at alpha
0.05 gives roughly a one in three chance of at least one false alarm even when
nothing changed. The correction here is Benjamini and Hochberg at a 10 percent
false discovery rate rather than Bonferroni, because training metrics are
strongly correlated with one another and Bonferroni, which assumes the worst
case dependence, would be needlessly conservative and cost real power
(Benjamini and Hochberg, *Journal of the Royal Statistical Society B* 57(1),
1995).

The two gates are independent and both must pass. Findings are then ranked by
severity, so the top of the table is the thing to look at first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from triage.analysis.comparison import ComparisonResult

VERDICT_REGRESSION = "regression"
VERDICT_IMPROVEMENT = "improvement"
VERDICT_NO_CHANGE = "no significant change"
VERDICT_BELOW_THRESHOLD = "significant but below the practical threshold"
VERDICT_UNDERPOWERED = "inconclusive: the design cannot reach alpha"


@dataclass(frozen=True)
class RegressionConfig:
    """The two gates and the correction, all visible and all configurable."""

    alpha: float = 0.05
    practical_threshold_pct: float = 2.0
    false_discovery_rate: float = 0.10
    correct_across: str = "metric"

    def describe(self) -> str:
        return (
            f"two gates: Benjamini Hochberg adjusted p below {self.alpha:g} at a "
            f"{self.false_discovery_rate:.0%} false discovery rate, and a relative effect of at "
            f"least {self.practical_threshold_pct:g} percent"
        )


@dataclass(frozen=True)
class Finding:
    """One comparison, judged. Everything needed to justify the verdict is here."""

    result: ComparisonResult
    adjusted_p: float
    verdict: str
    severity: float
    passes_statistical_gate: bool
    passes_practical_gate: bool

    @property
    def tag(self) -> str:
        return self.result.tag

    @property
    def candidate(self) -> str:
        return self.result.candidate

    @property
    def is_regression(self) -> bool:
        return self.verdict == VERDICT_REGRESSION

    @property
    def is_improvement(self) -> bool:
        return self.verdict == VERDICT_IMPROVEMENT

    def explain(self) -> str:
        """One line saying why this verdict, in the terms of the two gates."""
        statistical = "passed" if self.passes_statistical_gate else "failed"
        practical = "passed" if self.passes_practical_gate else "failed"
        return (
            f"{self.verdict}: statistical gate {statistical} "
            f"(adjusted p = {self.adjusted_p:.4f}), practical gate {practical} "
            f"(|{self.result.relative_effect_pct:+.2f} percent| against a threshold)"
        )


def benjamini_hochberg(p_values: list[float], false_discovery_rate: float = 0.10) -> np.ndarray:
    """Adjusted p values controlling the false discovery rate at `q`.

    Returns the step up adjusted values, which are monotone in the raw p values
    and directly comparable against alpha. Using adjusted values rather than a
    per test critical line keeps the reports honest, because the number printed
    beside a verdict is the number the verdict was made on.

    The `false_discovery_rate` argument does not change the adjusted values,
    which is a property of the procedure rather than an oversight: comparing the
    adjusted value against q is the same decision as the original step up rule.
    It is accepted here so the caller states the rate it intends, and so the
    reports can print it.
    """
    values = np.asarray(p_values, dtype=np.float64)
    n = values.size
    if n == 0:
        return values
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    scaled = ranked * n / np.arange(1, n + 1)
    # Step up: an adjusted value can never exceed one computed at a larger rank.
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def severity_of(result: ComparisonResult, adjusted_p: float) -> float:
    """How loudly this finding should shout, for ranking the verdict table.

    Size of the harm first, confidence second. The magnitude of the relative
    regression sets the scale, and a factor that grows as the adjusted p falls
    breaks ties between regressions of similar size. A result from the weaker
    single run mode is halved, because it is a weaker claim about the world and
    should not outrank a seed replicated finding of the same size.
    """
    harm = max(0.0, -result.signed_improvement_pct)
    confidence = float(-np.log10(max(adjusted_p, 1e-12)))
    severity = harm * (1.0 + confidence)
    return severity * 0.5 if result.is_weak_mode else severity


def classify(
    results: list[ComparisonResult], config: RegressionConfig | None = None
) -> list[Finding]:
    """Apply both gates with a false discovery correction, ranked by severity.

    The correction is applied across the metrics compared against one baseline,
    which is the family a reader actually looks at together. Correcting across
    every comparison in a large sweep at once would be defensible too, and would
    be far more conservative; the family chosen here is stated in the reports so
    nobody has to infer it.
    """
    config = config or RegressionConfig()
    if not results:
        return []

    adjusted = benjamini_hochberg(
        [result.p_value for result in results], config.false_discovery_rate
    )

    findings: list[Finding] = []
    for result, adjusted_p in zip(results, adjusted, strict=True):
        statistical = bool(adjusted_p < config.alpha)
        practical = bool(abs(result.relative_effect_pct) >= config.practical_threshold_pct)

        if result.min_attainable_p > config.alpha:
            verdict = VERDICT_UNDERPOWERED
        elif statistical and practical:
            verdict = VERDICT_IMPROVEMENT if result.improved else VERDICT_REGRESSION
        elif statistical:
            verdict = VERDICT_BELOW_THRESHOLD
        else:
            verdict = VERDICT_NO_CHANGE

        findings.append(
            Finding(
                result=result,
                adjusted_p=float(adjusted_p),
                verdict=verdict,
                severity=severity_of(result, float(adjusted_p))
                if verdict == VERDICT_REGRESSION
                else 0.0,
                passes_statistical_gate=statistical,
                passes_practical_gate=practical,
            )
        )

    return rank(findings)


def rank(findings: list[Finding]) -> list[Finding]:
    """Regressions first by severity, then improvements by size, then the rest."""

    def key(finding: Finding) -> tuple[int, float, float, str, str]:
        if finding.is_regression:
            group, magnitude = 0, -finding.severity
        elif finding.is_improvement:
            group, magnitude = 1, -finding.result.signed_improvement_pct
        else:
            group, magnitude = 2, finding.adjusted_p
        return (group, magnitude, finding.adjusted_p, finding.candidate, finding.tag)

    return sorted(findings, key=key)


@dataclass
class TriageReport:
    """Every finding for one baseline, plus the configuration that judged them."""

    findings: list[Finding] = field(default_factory=list)
    config: RegressionConfig = field(default_factory=RegressionConfig)
    baseline: str = ""

    @property
    def regressions(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.is_regression]

    @property
    def improvements(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.is_improvement]

    @property
    def best_candidate(self) -> str | None:
        """The run or variant with the largest significant improvement, if any."""
        improvements = self.improvements
        if not improvements:
            return None
        return max(improvements, key=lambda f: f.result.signed_improvement_pct).candidate

    def summary(self) -> str:
        return (
            f"{len(self.findings)} comparisons against {self.baseline}: "
            f"{len(self.regressions)} regressions, {len(self.improvements)} improvements"
        )
