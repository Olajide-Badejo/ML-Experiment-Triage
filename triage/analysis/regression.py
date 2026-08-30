"""Which of these differences should anyone actually act on.

A p value alone is a bad trigger for a regression alert, for two reasons that
pull in opposite directions.

**Too many alerts.** With enough data every difference becomes statistically
significant, including a 0.05 percent change nobody would act on. So a finding
must clear a *practical* gate as well as a statistical one, which is a visible,
configurable constant rather than a number buried in the code: a relative effect
of at least `practical_threshold_pct`, or, for a metric whose baseline can be
zero or whose units mean something on their own, an absolute effect of at least
`practical_threshold_absolute`. Exactly one of the two is ever in force.

**Too many false alerts.** Testing eight metrics against a baseline at alpha
0.05 gives roughly a one in three chance of at least one false alarm even when
nothing changed. The correction here is Benjamini and Hochberg at a 5 percent
false discovery rate by default, configurable with `--fdr`, rather than
Bonferroni, because training metrics are strongly correlated with one another
and Bonferroni, which assumes the worst case dependence, would be needlessly
conservative and cost real power (Benjamini and Hochberg, *Journal of the Royal
Statistical Society B* 57(1), 1995).

The two gates are independent and both must pass. Findings are then ranked by
severity, so the top of the table is the thing to look at first.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from triage.analysis.comparison import ComparisonResult

VERDICT_REGRESSION = "regression"
VERDICT_IMPROVEMENT = "improvement"
VERDICT_NO_CHANGE = "no significant change"
VERDICT_BELOW_THRESHOLD = "significant but below the practical threshold"
VERDICT_UNDERPOWERED = "inconclusive: the design cannot reach alpha"

#: The family a comparison that can never reach alpha is put in. It is not a
#: family in the Benjamini Hochberg sense at all: it is the set held out of every
#: denominator, because a comparison that cannot be a discovery can only cost the
#: admissible ones power by being counted beside them.
FAMILY_INADMISSIBLE = "inadmissible"


@dataclass(frozen=True)
class RegressionConfig:
    """The two gates and the correction, all visible and all configurable."""

    #: Admissibility only, and nothing else. A design whose smallest attainable p
    #: value exceeds `alpha` cannot produce a discovery at all, so its findings
    #: are reported as inconclusive rather than as "no change". This field does
    #: NOT gate significance: `false_discovery_rate` does. Through v1.0.0 the
    #: statistical gate read `adjusted_p < alpha` while `false_discovery_rate`
    #: was passed to a procedure that ignores it, so the rate the reports named
    #: was not the rate they applied. The field is kept, with its meaning
    #: narrowed, because a consumer constructs this dataclass and removing a
    #: field is a break.
    alpha: float = 0.05
    #: The practical gate as a percentage of the baseline. Set this or
    #: `practical_threshold_absolute`, never both and never neither.
    practical_threshold_pct: float | None = 2.0
    #: The practical gate in the metric's own units. A relative gate divides by
    #: the baseline, and a baseline can be zero: `_relative` then returns 0.0,
    #: which is right for the arithmetic and wrong for the verdict, silently
    #: downgrading real regressions on metrics that cross zero to "below the
    #: practical threshold". A metric that carries a unit (a latency in
    #: microseconds, a reward, a signed delta) belongs on this gate instead.
    practical_threshold_absolute: float | None = None
    #: The operative statistical gate: a finding is significant when its
    #: Benjamini Hochberg adjusted p value is at or below this rate. Five percent
    #: by default, which is the decision the old alpha gate happened to make, so
    #: making the field operative does not shift anybody's verdicts at defaults.
    false_discovery_rate: float = 0.05
    correct_across: str = "metric"
    #: DEPRECATED spelling of `practical_threshold_pct`, accepted for one minor
    #: version and removed in 1.2.0. It is the name the `--practical-threshold`
    #: flag is spelled after, so it is the one a caller is most likely to have
    #: written by hand. Passing it sets `practical_threshold_pct` and warns.
    practical_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.practical_threshold is not None:
            warnings.warn(
                "RegressionConfig(practical_threshold=...) is deprecated and will be "
                "removed in 1.2.0; pass practical_threshold_pct instead, or "
                "practical_threshold_absolute for a gate in the metric's own units",
                DeprecationWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "practical_threshold_pct", self.practical_threshold)
            object.__setattr__(self, "practical_threshold", None)

        relative = self.practical_threshold_pct is not None
        absolute = self.practical_threshold_absolute is not None
        if relative == absolute:
            both_or_neither = "both are set" if relative else "neither is set"
            raise ValueError(
                "exactly one of practical_threshold_pct and "
                f"practical_threshold_absolute must be set, and {both_or_neither}. "
                "A finding cannot clear two practical gates at once, and a report "
                "that named one while the code applied the other would be the "
                "defect this rule exists to prevent. To gate in the metric's own "
                "units, pass practical_threshold_pct=None alongside "
                "practical_threshold_absolute"
            )

    def describe_practical_gate(self) -> str:
        """The practical gate actually in force, in the terms it is applied in."""
        if self.practical_threshold_absolute is not None:
            return (
                f"an absolute effect of at least {self.practical_threshold_absolute:g} "
                f"in the units of the metric"
            )
        return f"a relative effect of at least {self.practical_threshold_pct:g} percent"

    def describe(self) -> str:
        return (
            f"two gates: Benjamini Hochberg at FDR {self.false_discovery_rate:.0%} "
            f"(adjusted p at or below {self.false_discovery_rate:g}, configurable with --fdr), "
            f"and {self.describe_practical_gate()}; "
            f"alpha {self.alpha:g} bounds admissibility only"
        )

    def clears_practical_gate(self, result: ComparisonResult) -> bool:
        """Whether this comparison's effect is large enough to be worth acting on."""
        if self.practical_threshold_absolute is not None:
            return bool(abs(result.effect) >= self.practical_threshold_absolute)
        return bool(abs(result.relative_effect_pct) >= self.practical_threshold_pct)


@dataclass(frozen=True)
class Finding:
    """One comparison, judged. Everything needed to justify the verdict is here."""

    result: ComparisonResult
    adjusted_p: float
    verdict: str
    severity: float
    passes_statistical_gate: bool
    passes_practical_gate: bool
    #: The multiplicity family this finding was corrected inside, which is the
    #: comparison mode for an admissible result and `FAMILY_INADMISSIBLE` for one
    #: whose design cannot reach alpha. An adjusted p value means nothing without
    #: the family it was adjusted within, so the family travels with it.
    family: str = ""

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
            f"({self.result.relative_effect_pct:+.2f} percent, "
            f"{self.result.effect:+.4g} absolute, against the threshold in force)"
        )


def benjamini_hochberg(p_values: list[float], false_discovery_rate: float = 0.05) -> np.ndarray:
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


#: The most the confidence term may multiply the harm by. A tie breaker that can
#: reorder the thing it is breaking ties within is not a tie breaker: uncapped,
#: `1 + -log10(p)` spans 1 to 13, and a 3 percent regression at p = 1e-12 scored
#: 39.00 against 23.98 for a 10 percent regression at p = 0.04. At 2.0 the
#: confidence term can at most double a harm, so it separates regressions of
#: similar size and never overturns a difference in size of more than two times.
MAX_CONFIDENCE_FACTOR = 2.0


def severity_of(result: ComparisonResult, adjusted_p: float) -> float:
    """How loudly this finding should shout, for ranking the verdict table.

    Size of the harm first, confidence second, and second means second. The
    magnitude of the relative regression sets the scale; a factor that grows as
    the adjusted p falls breaks ties between regressions of similar size, capped
    at `MAX_CONFIDENCE_FACTOR` so that it can never do more than that. A result
    from the weaker single run mode is halved, because it is a weaker claim about
    the world and should not outrank a seed replicated finding of the same size.
    """
    harm = max(0.0, -result.signed_improvement_pct)
    confidence = float(-np.log10(max(adjusted_p, 1e-12)))
    severity = harm * min(1.0 + confidence, MAX_CONFIDENCE_FACTOR)
    return severity * 0.5 if result.is_weak_mode else severity


def family_of(result: ComparisonResult, config: RegressionConfig | None = None) -> str:
    """The multiplicity family this comparison belongs to.

    The comparison mode is the family, because the mode is the claim: a seed
    replicated result and a single run window block result are not two tests of
    the same kind and must not share a denominator. Naming the family after the
    mode string rather than a fixed table keeps this open, so a caller that
    builds results in a mode of its own gets its own family instead of being
    quietly pooled with ours. A comparison that cannot reach alpha is held out of
    all of them.
    """
    config = config or RegressionConfig()
    if result.min_attainable_p > config.alpha:
        return FAMILY_INADMISSIBLE
    return result.mode


def classify(
    results: list[ComparisonResult], config: RegressionConfig | None = None
) -> list[Finding]:
    """Apply both gates with a false discovery correction, in input order.

    The correction is applied across the metrics compared against one baseline,
    which is the family a reader actually looks at together. Correcting across
    every comparison in a large sweep at once would be defensible too, and would
    be far more conservative; the family chosen here is stated in the reports so
    nobody has to infer it.

    **One family per comparison mode, and the inadmissible held out.** A single
    pooled family was wrong twice over. The single run window block mode fires on
    53 to 87 percent of comparisons under seed variance alone, measured, so its p
    values in a shared denominator destroy the false discovery control of every
    seed replicated result beside them: two different claims about the world do
    not belong to one family. And a comparison whose smallest attainable p value
    already exceeds alpha can never be a discovery, so counting it in the
    denominator can only cost the admissible comparisons power; it is held out
    and reported as `FAMILY_INADMISSIBLE` instead. Each finding records the
    family it was judged in, because an adjusted p value without its family is
    not interpretable.

    Findings come back **in the order the results went in**, one per result, so
    that `zip(results, classify(results))` is a valid join. This function used to
    end `return rank(findings)`, which reordered the list against its own input:
    since `Finding.tag` is not unique once one metric is compared across several
    conditions, a caller rejoining findings to results was left with object
    identity as its only key. Severity order is what `rank` is for, and the CLI
    and the report both call it.
    """
    config = config or RegressionConfig()
    if not results:
        return []

    families: dict[str, list[int]] = defaultdict(list)
    for index, result in enumerate(results):
        families[family_of(result, config)].append(index)

    adjusted = np.empty(len(results), dtype=np.float64)
    for family, indices in families.items():
        raw = [results[index].p_value for index in indices]
        if family == FAMILY_INADMISSIBLE:
            # Not corrected, because it was not tested: adjusting a p value
            # inside a family of one and calling it adjusted would be a
            # dressed up raw p value. The raw value is what is reported.
            values = np.asarray(raw, dtype=np.float64)
        else:
            values = benjamini_hochberg(raw, config.false_discovery_rate)
        for index, value in zip(indices, values, strict=True):
            adjusted[index] = value

    findings: list[Finding] = []
    for result, adjusted_p in zip(results, adjusted, strict=True):
        family = family_of(result, config)
        # The gate is the rate the report names. Comparing the step up adjusted
        # value against q is exactly the original Benjamini Hochberg decision,
        # which is why the adjusted value is what gets printed beside it. An
        # inadmissible comparison never faces the gate at all.
        statistical = bool(adjusted_p <= config.false_discovery_rate)
        statistical = statistical and family != FAMILY_INADMISSIBLE
        practical = config.clears_practical_gate(result)

        if family == FAMILY_INADMISSIBLE:
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
                family=family,
                severity=severity_of(result, float(adjusted_p))
                if verdict == VERDICT_REGRESSION
                else 0.0,
                passes_statistical_gate=statistical,
                passes_practical_gate=practical,
            )
        )

    return findings


def rank(findings: list[Finding]) -> list[Finding]:
    """Regressions first by severity, then improvements by size, then the rest.

    Public and stable: this is the ordering every output of this tool presents,
    and a caller that wants it asks for it here rather than getting it as a side
    effect of `classify`.
    """

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
    def inadmissible(self) -> list[Finding]:
        """The comparisons no correction was applied to, because none could help.

        Reported apart from the corrected families rather than mixed into them:
        these designs cannot reach alpha, so their p values were never candidates
        for discovery and were kept out of every denominator.
        """
        return [finding for finding in self.findings if finding.family == FAMILY_INADMISSIBLE]

    def family_note(self) -> str:
        """Which families the correction ran in, and how big each one was."""
        counts: dict[str, int] = defaultdict(int)
        for finding in self.findings:
            counts[finding.family] += 1
        corrected = ", ".join(
            f"{family} ({count})"
            for family, count in sorted(counts.items())
            if family != FAMILY_INADMISSIBLE
        )
        held_out = counts.get(FAMILY_INADMISSIBLE, 0)
        if not corrected:
            corrected = "no family large enough to correct"
        return (
            f"corrected separately per comparison mode: {corrected}"
            f"; {held_out} excluded as inadmissible"
        )

    @property
    def best_finding(self) -> Finding | None:
        """The largest significant improvement, preferring the stronger claim.

        A single run window block improvement is not comparable to a seed
        replicated one of any size: it cannot see seed to seed variance, which is
        exactly the variance that decides whether an improvement is real. So the
        seed replicated improvements are considered first, and a weaker mode
        finding is only ever crowned when there is no stronger one to crown. The
        finding is returned rather than the name so that a caller can see, and
        label, which mode the answer came from.
        """
        improvements = self.improvements
        if not improvements:
            return None
        strong = [finding for finding in improvements if not finding.result.is_weak_mode]
        return max(strong or improvements, key=lambda f: f.result.signed_improvement_pct)

    @property
    def best_candidate(self) -> str | None:
        """The run or variant with the largest significant improvement, if any."""
        best = self.best_finding
        return None if best is None else best.candidate

    def summary(self) -> str:
        return (
            f"{len(self.findings)} comparisons against {self.baseline}: "
            f"{len(self.regressions)} regressions, {len(self.improvements)} improvements"
        )
