"""Fill or skip: the sequential decision layer, as a contextual bandit.

Autofill's product truth is that a wrong fill costs the user more than no fill.
A classifier reports a distribution; a product has to choose an action, and the
two are not the same problem. This module is that second problem, modelled as a
CONTEXTUAL BANDIT: context is `(predicted type, calibrated confidence, locale)`,
the actions are `{fill, skip}`, and the reward is `+1` for a correct fill, `0`
for a skip, and a type dependent penalty for a wrong one. It is a bandit and not
reinforcement learning: there is no state that an action carries forward, no
episode and no discounting, and one field's decision does not change the next
field's context. Calling it RL would claim machinery that is not here and would
be a worse description of what it does.

**Why the confidence has to be the calibrated one.** The decision rule is an
expected value comparison: fill when `p * 1 + (1 - p) * (-penalty)` beats zero.
Every number in that expression except `p` is known exactly, so the whole layer
is only as good as `p` is honest. An uncalibrated 0.9 is not a usable 0.9, which
is why `calibration.py` exists and why this module reads what it produced.

**Why there is no importance sampling estimator here, and this is the honest
answer rather than the convenient one.** Off policy evaluation needs an
estimator (IPS, self normalised IPS, doubly robust) for one specific reason: the
logged data records the reward of the action that WAS taken and nothing about
the action that was not, so the value of a new policy has to be reconstructed
from a reweighted sample, at a variance cost that grows with how far the new
policy is from the logging one. That reason does not apply here. The evaluation
corpus is fully labelled, and the reward is a known function of `(prediction,
truth, action)`, so BOTH arms of every row are computable: the reward of filling
is `+1` or the penalty, the reward of skipping is `0`, and a policy's expected
reward is a sum over rows rather than an estimate with a confidence interval.
Fitting an IPS estimator on top of that would add variance to a quantity that is
already exact, and reporting its error bars would be theatre. What this module
cannot do, and does not claim, is estimate the reward of a policy that would
have shown the user a DIFFERENT set of fields; nothing here changes which fields
exist.

**The penalty is charged against the true type of the field, not the predicted
one.** A wrong fill leaves a bad value in a field whose type is what it is: the
user goes and repairs a payment field, an address field or a comment box, and
the work is the work regardless of what the model thought it was doing. That
choice also makes the penalty a property of the row rather than of the mistake,
which is what makes the offline evaluation above exact. It has a consequence
worth stating: the cost of an action is NOT observable at decision time, since
the policy only knows the predicted type. That gap is not a defect in the model,
it is the bandit problem, and it is why the learned thresholds are indexed by
what the policy can see.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triage.autofill.generator import FieldRecord
from triage.autofill.taxonomy import FIELD_TYPES, FieldType, index_of

#: A threshold of zero fills everything, because a confidence is never negative.
ALWAYS_FILL = 0.0

#: A threshold above one fills nothing, because a probability never exceeds one.
#: Two rather than infinity so the value survives JSON, which has no infinity.
NEVER_FILL = 2.0

#: The cost tiers, ordered from the most expensive mistake to the cheapest.
TIERS: tuple[str, ...] = ("payment", "identity", "address", "other")

#: The defaults 5.3 names: four for payment and identity tokens, two for
#: address, one for the rest. They are ratios rather than currency: filling a
#: card number wrongly is four correct fills worth of damage, and a policy only
#: ever compares them against each other.
DEFAULT_PENALTIES: dict[str, float] = {
    "payment": 4.0,
    "identity": 4.0,
    "address": 2.0,
    "other": 1.0,
}

#: Which tier each token belongs to. `organization` sits in `other` beside
#: `unknown` rather than in `address`: a company name is neither a person's
#: identity nor a locative component of an address, and a wrong one is a typo
#: rather than a leak. Every token in the taxonomy is here, because a token with
#: no tier would be a wrong fill that costs nothing.
TIER_OF: dict[FieldType, str] = {
    FieldType.CC_NAME: "payment",
    FieldType.CC_NUMBER: "payment",
    FieldType.CC_EXP_MONTH: "payment",
    FieldType.CC_EXP_YEAR: "payment",
    FieldType.CC_CSC: "payment",
    FieldType.GIVEN_NAME: "identity",
    FieldType.FAMILY_NAME: "identity",
    FieldType.NAME: "identity",
    FieldType.EMAIL: "identity",
    FieldType.TEL: "identity",
    FieldType.USERNAME: "identity",
    FieldType.ADDRESS_LINE1: "address",
    FieldType.ADDRESS_LINE2: "address",
    FieldType.ADDRESS_LEVEL2: "address",
    FieldType.ADDRESS_LEVEL1: "address",
    FieldType.POSTAL_CODE: "address",
    FieldType.COUNTRY_NAME: "address",
    FieldType.ORGANIZATION: "other",
    FieldType.UNKNOWN: "other",
}

#: Tier index per head index, for the bandit's context key.
_TIER_INDEX: np.ndarray = np.asarray(
    [TIERS.index(TIER_OF[member]) for member in FIELD_TYPES], dtype=np.int64
)

#: Rows of one predicted type below this are not evidence about that type, and
#: it inherits the global threshold instead. Twenty is the same floor the
#: evaluation uses for a bootstrap: fewer than that and a fitted cut is a fact
#: about a handful of rows.
DEFAULT_MIN_SUPPORT = 20

#: Points in the reported global sweep. The optimum is not found by this grid
#: (`best_threshold` finds it exactly); the grid is what the report quotes when
#: it says the threshold was swept.
DEFAULT_SWEEP_POINTS = 21

#: The order the comparison reports, from the trivial safe option upward.
POLICY_ORDER: tuple[str, ...] = (
    "never_fill",
    "always_fill",
    "global_threshold",
    "per_type_threshold",
)


@dataclass(frozen=True)
class RewardModel:
    """What each outcome is worth, in units of one correct fill.

    Configurable per 5.3, and configurable as TIERS rather than as nineteen
    numbers: the tier is the product judgement (how much does this class of
    mistake cost a user), and spreading it over individual tokens would invite
    tuning nineteen knobs against one evaluation split.
    """

    correct_fill: float = 1.0
    skip: float = 0.0
    penalties: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PENALTIES))

    def __post_init__(self) -> None:
        missing = [tier for tier in TIERS if tier not in self.penalties]
        if missing:
            raise ValueError(f"no penalty for tier(s) {', '.join(missing)}: every tier needs one")
        for tier, value in self.penalties.items():
            if tier not in TIERS:
                raise ValueError(f"{tier!r} is not a cost tier; the tiers are {', '.join(TIERS)}")
            if value < 0:
                raise ValueError(
                    f"a penalty is a magnitude, not a signed reward; got {value} for {tier!r}"
                )

    def penalty(self, field_type: FieldType) -> float:
        """What a wrong fill in a field of this type costs, as a magnitude."""
        return float(self.penalties[TIER_OF[field_type]])

    def penalty_by_index(self) -> np.ndarray:
        """The same, as a vector over head indices."""
        return np.asarray([self.penalty(member) for member in FIELD_TYPES], dtype=np.float64)

    def fill_rewards(self, stream: DecisionStream) -> np.ndarray:
        """The reward of FILLING each row, which the truth makes computable.

        The reward of skipping is `self.skip` for every row and needs no vector.
        """
        correct = stream.predicted == stream.truth
        penalties = self.penalty_by_index()[stream.truth]
        return np.where(correct, self.correct_fill, -penalties)

    @classmethod
    def parse(cls, spec: str) -> RewardModel:
        """`payment=8,address=3`, leaving the tiers it does not name alone.

        A partial specification on purpose: the defaults are documented and a
        caller who wants payment mistakes to cost twice as much should not have
        to restate the other three and risk changing one by accident.
        """
        penalties = dict(DEFAULT_PENALTIES)
        for chunk in (part.strip() for part in spec.split(",")):
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(
                    f"{chunk!r} is not a penalty: write them as tier=value, comma separated, "
                    f"over the tiers {', '.join(TIERS)}"
                )
            tier, _, value = chunk.partition("=")
            tier = tier.strip()
            if tier not in TIERS:
                raise ValueError(f"{tier!r} is not a cost tier; the tiers are {', '.join(TIERS)}")
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f"{value.strip()!r} is not a number, in {chunk!r}") from None
            if number < 0:
                raise ValueError(
                    f"a penalty is a magnitude, not a signed reward; got {number} for {tier!r}"
                )
            penalties[tier] = number
        return cls(penalties=penalties)

    def describe(self) -> str:
        """The reward function as one line, for a summary or a caption."""
        tiers = ", ".join(f"{tier} {-self.penalties[tier]:g}" for tier in TIERS)
        return (
            f"correct fill {self.correct_fill:g}, skip {self.skip:g}, wrong fill by tier: {tiers}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct_fill": self.correct_fill,
            "skip": self.skip,
            "penalties": {tier: float(self.penalties[tier]) for tier in TIERS},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RewardModel:
        return cls(
            correct_fill=float(data["correct_fill"]),
            skip=float(data["skip"]),
            penalties={tier: float(value) for tier, value in data["penalties"].items()},
        )


@dataclass(frozen=True)
class DecisionStream:
    """The rows a policy decides over, as parallel arrays.

    `predicted`, `confidence` and `locale_code` are the CONTEXT: everything the
    policy is allowed to look at. `truth` is not context, it is what makes the
    offline evaluation exact, and it is deliberately a separate field so that no
    policy can reach it by accident.
    """

    truth: np.ndarray
    predicted: np.ndarray
    confidence: np.ndarray
    locale_code: np.ndarray
    locales: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(self.truth.size)

    @classmethod
    def build(
        cls,
        truth: Sequence[int],
        predicted: Sequence[int],
        confidence: Sequence[float],
        locales: Sequence[str],
    ) -> DecisionStream:
        truth_array = np.asarray(truth, dtype=np.int64)
        predicted_array = np.asarray(predicted, dtype=np.int64)
        confidence_array = np.asarray(confidence, dtype=np.float64)
        sizes = {truth_array.size, predicted_array.size, confidence_array.size, len(locales)}
        if len(sizes) != 1:
            raise ValueError(
                f"a stream needs one truth, one prediction, one confidence and one locale per "
                f"row; got sizes {sorted(sizes)}"
            )
        if truth_array.size == 0:
            raise ValueError("no rows to decide over: an empty stream has no expected reward")
        names = tuple(sorted(set(locales)))
        codes = np.asarray([names.index(name) for name in locales], dtype=np.int64)
        return cls(
            truth=truth_array,
            predicted=predicted_array,
            confidence=confidence_array,
            locale_code=codes,
            locales=names,
        )


def stream_from_records(
    records: Sequence[FieldRecord], predicted: np.ndarray, confidence: np.ndarray
) -> DecisionStream:
    """A stream from the rows and one engine's calibrated predictions."""
    return DecisionStream.build(
        truth=[index_of(record.field_type) for record in records],
        predicted=[int(value) for value in predicted],
        confidence=[float(value) for value in confidence],
        locales=[record.locale for record in records],
    )


@dataclass(frozen=True)
class ThresholdPolicy:
    """Fill when the calibrated confidence reaches this type's threshold.

    All four policies compared here are one class: a vector of nineteen
    thresholds, one per predicted type. Always filling is the vector of zeros,
    never filling the vector of twos, a global threshold the constant vector,
    and the learned policy the interesting one. Writing them as one object is
    not a trick to save code; it is the statement that they are the same policy
    class evaluated at different points, which is what makes "the per type
    policy is at least as good as always filling, in sample" true BY
    CONSTRUCTION rather than by luck.
    """

    kind: str
    thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.thresholds) != len(FIELD_TYPES):
            raise ValueError(
                f"a threshold per predicted type: the taxonomy has {len(FIELD_TYPES)} tokens "
                f"and this policy carries {len(self.thresholds)}"
            )

    @property
    def name(self) -> str:
        return self.kind

    def vector(self) -> np.ndarray:
        return np.asarray(self.thresholds, dtype=np.float64)

    def fill_mask(self, stream: DecisionStream) -> np.ndarray:
        """`True` where this policy fills, over a whole stream at once."""
        mask: np.ndarray = stream.confidence >= self.vector()[stream.predicted]
        return mask

    def decide(self, field_type: FieldType, confidence: float) -> bool:
        """One decision, which is what the agentic demo calls per field."""
        return bool(confidence >= self.thresholds[index_of(field_type)])

    def threshold_for(self, field_type: FieldType) -> float:
        return self.thresholds[index_of(field_type)]

    def is_uniform(self) -> bool:
        return len(set(self.thresholds)) == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "thresholds": {
                member.value: float(value)
                for member, value in zip(FIELD_TYPES, self.thresholds, strict=True)
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ThresholdPolicy:
        thresholds = data["thresholds"]
        missing = [member.value for member in FIELD_TYPES if member.value not in thresholds]
        if missing:
            raise ValueError(
                f"the exported policy has no threshold for {', '.join(missing)}: it was written "
                f"against a different taxonomy and cannot be applied to this one"
            )
        return cls(
            kind=str(data["kind"]),
            thresholds=tuple(float(thresholds[member.value]) for member in FIELD_TYPES),
        )


def always_fill() -> ThresholdPolicy:
    """Fill every field, whatever the model says: the naive product."""
    return ThresholdPolicy(kind="always_fill", thresholds=(ALWAYS_FILL,) * len(FIELD_TYPES))


def never_fill() -> ThresholdPolicy:
    """Fill nothing: reward exactly zero, and the bar a policy has to clear."""
    return ThresholdPolicy(kind="never_fill", thresholds=(NEVER_FILL,) * len(FIELD_TYPES))


def global_threshold(threshold: float) -> ThresholdPolicy:
    """One confidence cut for every type."""
    return ThresholdPolicy(
        kind="global_threshold", thresholds=(float(threshold),) * len(FIELD_TYPES)
    )


def per_type_threshold(thresholds: Mapping[FieldType, float]) -> ThresholdPolicy:
    """A cut per predicted type, which is the context the policy can see."""
    return ThresholdPolicy(
        kind="per_type_threshold",
        thresholds=tuple(float(thresholds[member]) for member in FIELD_TYPES),
    )


@dataclass(frozen=True)
class PolicyOutcome:
    """One policy's exact performance on one stream.

    `expected_reward` is per FIELD rather than per fill, so that policies which
    fill different numbers of fields are comparable: a policy that fills one
    field perfectly is not better than one that fills a thousand well.
    """

    name: str
    kind: str
    n_rows: int
    n_filled: int
    fill_rate: float
    total_reward: float
    expected_reward: float
    correction_cost: float
    fill_accuracy: float | None
    policy: ThresholdPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "n_rows": self.n_rows,
            "n_filled": self.n_filled,
            "fill_rate": self.fill_rate,
            "total_reward": self.total_reward,
            "expected_reward": self.expected_reward,
            "correction_cost": self.correction_cost,
            "fill_accuracy": self.fill_accuracy,
            "threshold": self.policy.thresholds[0] if self.policy.is_uniform() else None,
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyOutcome:
        accuracy = data["fill_accuracy"]
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            n_rows=int(data["n_rows"]),
            n_filled=int(data["n_filled"]),
            fill_rate=float(data["fill_rate"]),
            total_reward=float(data["total_reward"]),
            expected_reward=float(data["expected_reward"]),
            correction_cost=float(data["correction_cost"]),
            fill_accuracy=None if accuracy is None else float(accuracy),
            policy=ThresholdPolicy.from_dict(data["policy"]),
        )


def evaluate_policy(
    policy: ThresholdPolicy, stream: DecisionStream, reward: RewardModel, name: str | None = None
) -> PolicyOutcome:
    """Exact offline evaluation: a sum over rows, not an estimate.

    Both arms of every row are known (see the module docstring), so this walks
    the stream once and adds up what actually happens. There is no estimator, no
    resampling and no interval, because there is nothing here to be uncertain
    about beyond the sample itself.
    """
    filled = policy.fill_mask(stream)
    fill_reward = reward.fill_rewards(stream)
    earned = np.where(filled, fill_reward, reward.skip)
    wrong = filled & (stream.predicted != stream.truth)
    penalties = reward.penalty_by_index()[stream.truth]
    n_filled = int(filled.sum())
    return PolicyOutcome(
        name=name or policy.kind,
        kind=policy.kind,
        n_rows=stream.n_rows,
        n_filled=n_filled,
        fill_rate=n_filled / stream.n_rows,
        total_reward=float(earned.sum()),
        expected_reward=float(earned.mean()),
        # What the user pays to undo the mistakes, per field of the form. It is
        # not the negative of the reward: a skip costs the user typing, which is
        # what they were doing anyway, and charging for it would make doing
        # nothing look expensive.
        correction_cost=float((penalties * wrong).sum() / stream.n_rows),
        fill_accuracy=(
            None
            if n_filled == 0
            else float((filled & (stream.predicted == stream.truth)).sum()) / n_filled
        ),
        policy=policy,
    )


def best_threshold(confidence: np.ndarray, rewards: np.ndarray) -> tuple[float, float]:
    """The exact best `fill iff confidence >= t`, and what it earns.

    Sorting by confidence turns "every threshold" into "every prefix", and the
    best prefix is one cumulative sum away. Candidate cuts are taken only at the
    ends of runs of EQUAL confidence, because a threshold cannot split two rows
    that look identical to it, and pretending otherwise would report a reward no
    policy can actually earn.

    A prefix has to beat zero strictly to be chosen, so a stream where filling
    never pays yields `NEVER_FILL` rather than a cut that happens to break even.
    """
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if scores.size != values.size:
        raise ValueError(f"one reward per confidence: got {scores.size} and {values.size}")
    if scores.size == 0:
        return NEVER_FILL, 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    cumulative = np.cumsum(values[order])
    boundary = np.ones(sorted_scores.size, dtype=bool)
    boundary[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    candidates = np.flatnonzero(boundary)
    totals = cumulative[candidates]
    best = int(np.argmax(totals))
    if totals[best] <= 0:
        return NEVER_FILL, 0.0
    return float(sorted_scores[candidates[best]]), float(totals[best])


def fit_global_threshold(stream: DecisionStream, reward: RewardModel) -> ThresholdPolicy:
    """The single best confidence cut over the whole fitting split."""
    threshold, _ = best_threshold(stream.confidence, reward.fill_rewards(stream))
    return global_threshold(threshold)


def fit_per_type_thresholds(
    stream: DecisionStream, reward: RewardModel, min_support: int = DEFAULT_MIN_SUPPORT
) -> ThresholdPolicy:
    """One cut per predicted type, fitted independently and therefore optimally.

    The expected reward is a sum over rows and the predicted type partitions the
    rows, so maximising each type's own block maximises the total: the greedy
    fit IS the optimum over this policy class, and no joint search is needed.

    A type with fewer than `min_support` rows in the fitting split inherits the
    global threshold rather than a cut of its own. That is shrinkage toward the
    only evidence there is, and it is what keeps the fitted policy from carrying
    nineteen thresholds of which several were decided by three rows.
    """
    fill_reward = reward.fill_rewards(stream)
    fallback = fit_global_threshold(stream, reward).thresholds[0]
    thresholds: dict[FieldType, float] = {}
    for position, member in enumerate(FIELD_TYPES):
        rows = stream.predicted == position
        support = int(rows.sum())
        if support < max(1, min_support):
            thresholds[member] = fallback
            continue
        cut, _ = best_threshold(stream.confidence[rows], fill_reward[rows])
        thresholds[member] = cut
    return per_type_threshold(thresholds)


def sweep_global_threshold(
    stream: DecisionStream, reward: RewardModel, n_points: int = DEFAULT_SWEEP_POINTS
) -> tuple[tuple[float, float], ...]:
    """Expected reward across an evenly spaced grid of global thresholds.

    The reported sweep, not the search: `best_threshold` already found the exact
    optimum, and a grid cannot beat it. This exists so the report can show the
    SHAPE of the tradeoff, which is the part a reader needs in order to see that
    the optimum is a plateau rather than a spike.
    """
    if n_points < 2:
        raise ValueError(f"a sweep needs at least two points; got {n_points}")
    grid = np.linspace(0.0, 1.0, n_points)
    return tuple(
        (
            float(value),
            evaluate_policy(global_threshold(float(value)), stream, reward).expected_reward,
        )
        for value in grid
    )


@dataclass(frozen=True)
class ThompsonConfig:
    """The knobs of the online simulation, all of them seeded.

    `noise_sd` is the assumed spread of a single fill's reward around its cell
    mean. One is close to the truth here (rewards live in `[-4, 1]`) and it is a
    prior width rather than a fitted quantity: too small and the bandit stops
    exploring on two observations, too large and it never stops.
    """

    seed: int = 0
    n_buckets: int = 5
    prior_reward: float = 0.0
    prior_strength: float = 1.0
    noise_sd: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_buckets": self.n_buckets,
            "prior_reward": self.prior_reward,
            "prior_strength": self.prior_strength,
            "noise_sd": self.noise_sd,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ThompsonConfig:
        return cls(
            seed=int(data["seed"]),
            n_buckets=int(data["n_buckets"]),
            prior_reward=float(data["prior_reward"]),
            prior_strength=float(data["prior_strength"]),
            noise_sd=float(data["noise_sd"]),
        )


@dataclass(frozen=True)
class ThompsonResult:
    """What one seeded pass of the bandit over the stream actually earned.

    `actions` and `order` are arrays and are deliberately not serialised: the
    artifact records what the run measured, and a caller who wants the trace
    reruns the simulation, which is free and reproduces it exactly.
    """

    n_rows: int
    n_filled: int
    total_reward: float
    expected_reward: float
    correction_cost: float
    cumulative_regret: float
    regret_per_decision: float
    best_fixed: str
    best_fixed_reward: float
    config: ThompsonConfig
    n_cells: int
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), repr=False)
    order: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_filled": self.n_filled,
            "fill_rate": self.n_filled / self.n_rows if self.n_rows else 0.0,
            "total_reward": self.total_reward,
            "expected_reward": self.expected_reward,
            "correction_cost": self.correction_cost,
            "cumulative_regret": self.cumulative_regret,
            "regret_per_decision": self.regret_per_decision,
            "best_fixed": self.best_fixed,
            "best_fixed_reward": self.best_fixed_reward,
            "n_cells": self.n_cells,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ThompsonResult:
        return cls(
            n_rows=int(data["n_rows"]),
            n_filled=int(data["n_filled"]),
            total_reward=float(data["total_reward"]),
            expected_reward=float(data["expected_reward"]),
            correction_cost=float(data["correction_cost"]),
            cumulative_regret=float(data["cumulative_regret"]),
            regret_per_decision=float(data["regret_per_decision"]),
            best_fixed=str(data["best_fixed"]),
            best_fixed_reward=float(data["best_fixed_reward"]),
            config=ThompsonConfig.from_dict(data["config"]),
            n_cells=int(data["n_cells"]),
        )


def confidence_bucket_edges(confidence: np.ndarray, n_buckets: int) -> np.ndarray:
    """Interior bucket edges of EQUAL MASS, the same choice `calibration.py` made.

    A trained head's confidences pile up near one, so equal width buckets would
    put almost every decision in the top one and leave the bandit with a single
    arm that matters. Equal mass gives every cell a comparable number of
    observations to learn from.
    """
    if n_buckets < 1:
        raise ValueError(f"a bandit needs at least one confidence bucket; got {n_buckets}")
    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    return np.asarray(np.quantile(np.asarray(confidence, dtype=np.float64), quantiles))


def _cell_index(stream: DecisionStream, edges: np.ndarray, n_buckets: int) -> np.ndarray:
    """The context key: `(locale, cost tier of the predicted type, bucket)`.

    Pooling the nineteen predicted types down to their four cost tiers is
    deliberate. The arms have to be learnable from one pass over a few hundred
    decisions, and `2 x 19 x 5` arms would leave most of them with three
    observations and turn the simulation into a demonstration of the prior. The
    fitted threshold policy uses all nineteen types because it is fitted offline
    on the whole split at once, where support is not the binding constraint.

    Computed arithmetically from integer codes rather than from a dictionary of
    tuples, so the arm numbering is a fact about the taxonomy rather than about
    this process's string hashing.
    """
    bucket = np.searchsorted(edges, stream.confidence, side="right")
    tier = _TIER_INDEX[stream.predicted]
    index: np.ndarray = (stream.locale_code * len(TIERS) + tier) * n_buckets + bucket
    return index


def simulate_thompson(
    stream: DecisionStream,
    reward: RewardModel,
    benchmark: np.ndarray,
    config: ThompsonConfig | None = None,
    edges: np.ndarray | None = None,
    best_fixed: str = "best fixed policy",
) -> ThompsonResult:
    """One seeded pass of Thompson sampling over the evaluation stream.

    The point of this simulation is the part the offline comparison cannot show:
    what it COSTS to learn the policy rather than to be handed it. The fitted
    thresholds above saw the whole split at once. A bandit sees one field at a
    time, and only learns the reward of the arm it pulled, which is why the
    posterior is only updated when the action is `fill`: skipping teaches it
    nothing, and a simulation that updated on skips would be reporting numbers
    from full feedback while calling itself a bandit.

    **The posterior.** One Gaussian per context cell over the mean reward of
    filling there, conjugate under a Normal prior with known noise: after `n`
    observations the mean is the prior pulled toward the sample mean and the
    spread is `noise_sd / sqrt(prior_strength + n)`. Gaussian rather than the
    textbook Beta, because the reward is not a coin flip: it lives on `[-4, 1]`
    and its scale is exactly what the decision turns on. Sampling one draw and
    filling when it beats the reward of skipping is Thompson sampling; the
    exploration is the width of the posterior and nothing else.

    **Determinism.** Every draw comes from one `SeedSequence`, and the stream
    order is a seeded permutation of the rows. Nothing here consults `hash()`,
    the wall clock, or dictionary ordering.
    """
    settings = config or ThompsonConfig()
    rng = np.random.default_rng(np.random.SeedSequence(entropy=settings.seed))
    bucket_edges = (
        confidence_bucket_edges(stream.confidence, settings.n_buckets) if edges is None else edges
    )
    cells = _cell_index(stream, bucket_edges, settings.n_buckets)
    n_cells = len(stream.locales) * len(TIERS) * settings.n_buckets

    fill_reward = reward.fill_rewards(stream)
    penalties = reward.penalty_by_index()[stream.truth]
    wrong = stream.predicted != stream.truth

    counts = np.zeros(n_cells, dtype=np.float64)
    totals = np.zeros(n_cells, dtype=np.float64)
    order = rng.permutation(stream.n_rows)
    actions = np.zeros(stream.n_rows, dtype=bool)

    earned = 0.0
    cost = 0.0
    for position, row in enumerate(order):
        cell = int(cells[row])
        strength = settings.prior_strength + counts[cell]
        posterior_mean = (settings.prior_strength * settings.prior_reward + totals[cell]) / strength
        draw = float(rng.normal(posterior_mean, settings.noise_sd / np.sqrt(strength)))
        if draw <= reward.skip:
            continue
        actions[position] = True
        observed = float(fill_reward[row])
        counts[cell] += 1.0
        totals[cell] += observed
        earned += observed
        if wrong[row]:
            cost += float(penalties[row])

    target = np.asarray(benchmark, dtype=np.float64).reshape(-1)
    if target.size != stream.n_rows:
        raise ValueError(
            f"the benchmark needs one reward per row: got {target.size} for {stream.n_rows} rows"
        )
    best_reward = float(target.sum())
    n_filled = int(actions.sum())
    return ThompsonResult(
        n_rows=stream.n_rows,
        n_filled=n_filled,
        total_reward=earned,
        expected_reward=earned / stream.n_rows,
        correction_cost=cost / stream.n_rows,
        # Regret against the best fixed policy IN HINDSIGHT, computed exactly on
        # the same rows. It can come out negative, which is not a bug: the fixed
        # class is restricted, and a bandit that adapts within a stream can beat
        # every member of it on that stream.
        cumulative_regret=best_reward - earned,
        regret_per_decision=(best_reward - earned) / stream.n_rows,
        best_fixed=best_fixed,
        best_fixed_reward=best_reward / stream.n_rows,
        config=settings,
        n_cells=n_cells,
        actions=actions,
        order=order,
    )


@dataclass(frozen=True)
class PolicyComparison:
    """Every policy on one stream, the chosen one, and what learning it cost.

    This is the artifact the report renders and the agentic demo loads, so it is
    plain data with a round trip: the numbers on the page are the numbers that
    were measured, never a second computation that could disagree with them.
    """

    reward: RewardModel
    outcomes: tuple[PolicyOutcome, ...]
    chosen: str
    thompson: ThompsonResult
    global_sweep: tuple[tuple[float, float], ...]
    fit_split: str
    eval_split: str
    locale: str
    in_sample: bool
    min_support: int = DEFAULT_MIN_SUPPORT

    def outcome(self, name: str) -> PolicyOutcome:
        for item in self.outcomes:
            if item.name == name:
                return item
        raise KeyError(f"no policy named {name!r}; this comparison holds {self.names()}")

    def names(self) -> str:
        return ", ".join(item.name for item in self.outcomes)

    def best_fixed(self) -> PolicyOutcome:
        """The best of the fixed policies by exact expected reward."""
        return max(self.outcomes, key=lambda item: item.expected_reward)

    def chosen_policy(self) -> ThresholdPolicy:
        return self.outcome(self.chosen).policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward.to_dict(),
            # The reward function as one line, written here rather than rebuilt
            # by the reporting layer, which imports nothing from this package.
            "reward_note": self.reward.describe(),
            "policies": [item.to_dict() for item in self.outcomes],
            "chosen": self.chosen,
            "best_fixed": self.best_fixed().name,
            "thompson": self.thompson.to_dict(),
            "global_sweep": [[point, value] for point, value in self.global_sweep],
            "fit_split": self.fit_split,
            "eval_split": self.eval_split,
            "locale": self.locale,
            "in_sample": self.in_sample,
            "min_support": self.min_support,
            "context": ["predicted_type", "calibrated_confidence", "locale"],
            "method": (
                "exact offline policy evaluation over fully labelled rows; no importance "
                "sampling estimator is needed because both arms of every row are known"
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyComparison:
        return cls(
            reward=RewardModel.from_dict(data["reward"]),
            outcomes=tuple(PolicyOutcome.from_dict(row) for row in data["policies"]),
            chosen=str(data["chosen"]),
            thompson=ThompsonResult.from_dict(data["thompson"]),
            global_sweep=tuple((float(a), float(b)) for a, b in data["global_sweep"]),
            fit_split=str(data["fit_split"]),
            eval_split=str(data["eval_split"]),
            locale=str(data["locale"]),
            in_sample=bool(data["in_sample"]),
            min_support=int(data["min_support"]),
        )


def compare_policies(
    fit: DecisionStream,
    evaluation: DecisionStream,
    reward: RewardModel,
    *,
    fit_split: str = "val",
    eval_split: str = "val",
    locale: str = "all",
    chosen: str = "per_type_threshold",
    min_support: int = DEFAULT_MIN_SUPPORT,
    thompson: ThompsonConfig | None = None,
    n_sweep_points: int = DEFAULT_SWEEP_POINTS,
) -> PolicyComparison:
    """Fit on one stream, score every policy exactly on the other.

    The two streams are usually the validation split and whatever split is being
    evaluated, and they are usually the SAME split, which is honest but in
    sample: thresholds fitted on the rows they are then scored on cannot lose.
    That is recorded as `in_sample` rather than hidden, and the out of sample
    number is one flag away (`--split test`), which is the version of the claim
    that costs something.

    `in_sample` is decided by identity, not by equality: the caller either
    handed the same stream twice or it did not.
    """
    if chosen not in POLICY_ORDER:
        raise ValueError(f"{chosen!r} is not a policy; the choices are {', '.join(POLICY_ORDER)}")
    policies = {
        "never_fill": never_fill(),
        "always_fill": always_fill(),
        "global_threshold": fit_global_threshold(fit, reward),
        "per_type_threshold": fit_per_type_thresholds(fit, reward, min_support=min_support),
    }
    outcomes = tuple(
        evaluate_policy(policies[name], evaluation, reward, name=name) for name in POLICY_ORDER
    )
    best = max(outcomes, key=lambda item: item.expected_reward)
    simulation = simulate_thompson(
        evaluation,
        reward,
        benchmark=reward.fill_rewards(evaluation) * best.policy.fill_mask(evaluation),
        config=thompson,
        edges=confidence_bucket_edges(fit.confidence, (thompson or ThompsonConfig()).n_buckets),
        best_fixed=best.name,
    )
    return PolicyComparison(
        reward=reward,
        outcomes=outcomes,
        chosen=chosen,
        thompson=simulation,
        global_sweep=sweep_global_threshold(evaluation, reward, n_points=n_sweep_points),
        fit_split=fit_split,
        eval_split=eval_split,
        locale=locale,
        in_sample=fit is evaluation,
        min_support=min_support,
    )


def write_policy(path: Path | str, comparison: PolicyComparison) -> Path:
    """Export the comparison, which is how the agentic demo gets its policy."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def load_policy(path: Path | str) -> PolicyComparison:
    """Read an exported comparison back, thresholds and all."""
    return PolicyComparison.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
