"""The fill or skip decision layer: a contextual bandit over the same rows.

Two kinds of test live here. The small ones build a stream by hand and check the
arithmetic against a Python loop, because the whole claim of this module is that
a policy's expected reward is COMPUTED rather than estimated and an exact number
can be checked exactly. The large ones train the real head on the real corpus
and assert the acceptance criterion of 5.6: the per type threshold policy beats
always filling on expected reward.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage.autofill.evaluate import evaluate_engines, write_evaluation
from triage.autofill.features import featurise
from triage.autofill.generator import GeneratorConfig, generate_fields
from triage.autofill.model import TrainConfig, train
from triage.autofill.policy import (
    ALWAYS_FILL,
    DEFAULT_MIN_SUPPORT,
    DEFAULT_PENALTIES,
    NEVER_FILL,
    TIER_OF,
    TIERS,
    DecisionStream,
    PolicyComparison,
    RewardModel,
    ThompsonConfig,
    ThresholdPolicy,
    always_fill,
    best_threshold,
    compare_policies,
    evaluate_policy,
    fit_global_threshold,
    fit_per_type_thresholds,
    global_threshold,
    load_policy,
    never_fill,
    simulate_thompson,
    stream_from_records,
    sweep_global_threshold,
    write_policy,
)
from triage.autofill.taxonomy import FIELD_TYPES, FieldType, index_of


def _stream(
    truth: list[FieldType],
    predicted: list[FieldType],
    confidence: list[float],
    locales: list[str] | None = None,
) -> DecisionStream:
    """A hand written stream, so the arithmetic can be checked by hand."""
    return DecisionStream.build(
        truth=[index_of(item) for item in truth],
        predicted=[index_of(item) for item in predicted],
        confidence=confidence,
        locales=locales or ["en_US"] * len(truth),
    )


# --------------------------------------------------------------------------
# The reward model
# --------------------------------------------------------------------------


def test_the_penalty_tiers_are_the_ones_the_specification_names() -> None:
    assert DEFAULT_PENALTIES == {"payment": 4.0, "identity": 4.0, "address": 2.0, "other": 1.0}
    assert set(TIERS) == set(DEFAULT_PENALTIES)
    assert TIER_OF[FieldType.CC_NUMBER] == "payment"
    assert TIER_OF[FieldType.CC_CSC] == "payment"
    assert TIER_OF[FieldType.GIVEN_NAME] == "identity"
    assert TIER_OF[FieldType.EMAIL] == "identity"
    assert TIER_OF[FieldType.POSTAL_CODE] == "address"
    assert TIER_OF[FieldType.ADDRESS_LINE2] == "address"
    assert TIER_OF[FieldType.UNKNOWN] == "other"


def test_every_token_in_the_taxonomy_has_a_tier() -> None:
    """A token with no tier would be a wrong fill that costs nothing."""
    assert set(TIER_OF) == set(FIELD_TYPES)


def test_a_wrong_fill_costs_the_tier_of_the_field_it_landed_in() -> None:
    reward = RewardModel()
    assert reward.penalty(FieldType.CC_NUMBER) == 4.0
    assert reward.penalty(FieldType.ADDRESS_LINE1) == 2.0
    assert reward.penalty(FieldType.UNKNOWN) == 1.0
    vector = reward.penalty_by_index()
    assert vector.shape == (len(FIELD_TYPES),)
    assert vector[index_of(FieldType.CC_CSC)] == 4.0


def test_the_penalties_are_configurable() -> None:
    reward = RewardModel.parse("payment=8,address=3")
    assert reward.penalty(FieldType.CC_NUMBER) == 8.0
    assert reward.penalty(FieldType.POSTAL_CODE) == 3.0
    # Tiers the string left alone keep their documented defaults.
    assert reward.penalty(FieldType.EMAIL) == 4.0


def test_an_unknown_tier_in_a_penalty_string_is_refused() -> None:
    with pytest.raises(ValueError, match="payments"):
        RewardModel.parse("payments=8")


def test_an_unparsable_penalty_string_names_what_it_wanted() -> None:
    with pytest.raises(ValueError, match="tier=value"):
        RewardModel.parse("payment")


def test_a_negative_penalty_is_refused_because_the_sign_is_the_module_s() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        RewardModel.parse("payment=-4")


def test_a_correct_fill_earns_one_a_wrong_fill_is_charged_and_a_skip_earns_nothing() -> None:
    stream = _stream(
        truth=[FieldType.EMAIL, FieldType.CC_NUMBER, FieldType.POSTAL_CODE],
        predicted=[FieldType.EMAIL, FieldType.EMAIL, FieldType.ADDRESS_LINE1],
        confidence=[0.9, 0.8, 0.7],
    )
    rewards = RewardModel().fill_rewards(stream)
    assert rewards.tolist() == [1.0, -4.0, -2.0]


def test_a_wrong_fill_is_charged_against_the_true_type_not_the_predicted_one() -> None:
    """The user pays to repair the field they are standing in."""
    stream = _stream(
        truth=[FieldType.CC_NUMBER, FieldType.UNKNOWN],
        predicted=[FieldType.UNKNOWN, FieldType.CC_NUMBER],
        confidence=[0.9, 0.9],
    )
    assert RewardModel().fill_rewards(stream).tolist() == [-4.0, -1.0]


# --------------------------------------------------------------------------
# Exact offline evaluation
# --------------------------------------------------------------------------


def test_expected_reward_is_the_exact_mean_and_not_an_estimate() -> None:
    stream = _stream(
        truth=[FieldType.EMAIL, FieldType.CC_NUMBER, FieldType.POSTAL_CODE, FieldType.TEL],
        predicted=[FieldType.EMAIL, FieldType.EMAIL, FieldType.ADDRESS_LINE1, FieldType.TEL],
        confidence=[0.95, 0.85, 0.60, 0.99],
    )
    reward = RewardModel()
    outcome = evaluate_policy(global_threshold(0.9), stream, reward)
    # By hand: rows 0 and 3 are filled and correct, rows 1 and 2 are skipped.
    assert outcome.n_filled == 2
    assert outcome.total_reward == pytest.approx(2.0)
    assert outcome.expected_reward == pytest.approx(0.5)
    assert outcome.fill_rate == pytest.approx(0.5)
    assert outcome.correction_cost == pytest.approx(0.0)
    assert outcome.fill_accuracy == pytest.approx(1.0)


def test_the_correction_cost_is_what_the_user_pays_to_undo_the_wrong_fills() -> None:
    stream = _stream(
        truth=[FieldType.EMAIL, FieldType.CC_NUMBER, FieldType.POSTAL_CODE, FieldType.TEL],
        predicted=[FieldType.EMAIL, FieldType.EMAIL, FieldType.ADDRESS_LINE1, FieldType.TEL],
        confidence=[0.95, 0.85, 0.60, 0.99],
    )
    outcome = evaluate_policy(always_fill(), stream, RewardModel())
    assert outcome.correction_cost == pytest.approx(6.0 / 4)
    assert outcome.expected_reward == pytest.approx((1.0 - 4.0 - 2.0 + 1.0) / 4)


def test_an_exact_expected_reward_matches_a_loop_over_the_rows() -> None:
    """The estimator this module does not need, checked against the truth."""
    rng = np.random.default_rng(7)
    n = 400
    truth = rng.integers(0, len(FIELD_TYPES), n)
    predicted = np.where(rng.random(n) < 0.8, truth, rng.integers(0, len(FIELD_TYPES), n))
    stream = DecisionStream.build(
        truth=truth.tolist(),
        predicted=predicted.tolist(),
        confidence=rng.random(n).tolist(),
        locales=["de_DE" if value else "en_US" for value in rng.integers(0, 2, n)],
    )
    reward = RewardModel()
    policy = global_threshold(0.4)

    total = 0.0
    for position in range(n):
        if stream.confidence[position] < 0.4:
            continue
        true_type = FIELD_TYPES[int(stream.truth[position])]
        total += 1.0 if truth[position] == predicted[position] else -reward.penalty(true_type)

    outcome = evaluate_policy(policy, stream, reward)
    assert outcome.total_reward == pytest.approx(total)
    assert outcome.expected_reward == pytest.approx(total / n)


def test_never_filling_earns_exactly_nothing() -> None:
    stream = _stream(
        truth=[FieldType.EMAIL, FieldType.CC_NUMBER],
        predicted=[FieldType.EMAIL, FieldType.EMAIL],
        confidence=[1.0, 1.0],
    )
    outcome = evaluate_policy(never_fill(), stream, RewardModel())
    assert outcome.n_filled == 0
    assert outcome.expected_reward == 0.0
    assert outcome.correction_cost == 0.0
    assert outcome.fill_accuracy is None


def test_the_two_degenerate_policies_are_threshold_vectors_like_the_others() -> None:
    """One code path, four policies: the kinds differ only in their vector."""
    assert set(always_fill().thresholds) == {ALWAYS_FILL}
    assert set(never_fill().thresholds) == {NEVER_FILL}
    assert never_fill().decide(FieldType.EMAIL, 1.0) is False
    assert always_fill().decide(FieldType.EMAIL, 0.0) is True


# --------------------------------------------------------------------------
# Fitting thresholds
# --------------------------------------------------------------------------


def test_the_best_threshold_is_the_exact_maximum_over_every_cut() -> None:
    rng = np.random.default_rng(11)
    confidence = np.round(rng.random(60), 2)
    rewards = rng.choice([1.0, -1.0, -4.0], size=60)

    threshold, total = best_threshold(confidence, rewards)

    brute = 0.0
    for candidate in sorted({*confidence.tolist(), NEVER_FILL}):
        brute = max(brute, float(rewards[confidence >= candidate].sum()))
    assert total == pytest.approx(brute)
    assert float(rewards[confidence >= threshold].sum()) == pytest.approx(total)


def test_a_threshold_never_splits_a_run_of_equal_confidences() -> None:
    """Equal contexts get equal treatment, which is what a threshold means."""
    confidence = np.array([0.5, 0.5, 0.5, 0.9])
    rewards = np.array([1.0, -4.0, 1.0, 1.0])
    threshold, total = best_threshold(confidence, rewards)
    filled = confidence >= threshold
    assert filled.tolist() == [False, False, False, True]
    assert total == pytest.approx(1.0)


def test_the_best_threshold_refuses_to_fill_when_no_prefix_pays() -> None:
    confidence = np.array([0.9, 0.8, 0.7])
    rewards = np.array([-4.0, -4.0, -1.0])
    threshold, total = best_threshold(confidence, rewards)
    assert threshold == NEVER_FILL
    assert total == 0.0


def test_the_global_threshold_is_fitted_by_sweeping_the_split() -> None:
    stream = _stream(
        truth=[FieldType.EMAIL] * 4 + [FieldType.CC_NUMBER] * 4,
        predicted=[FieldType.EMAIL] * 8,
        confidence=[0.99, 0.98, 0.97, 0.96, 0.50, 0.40, 0.30, 0.20],
    )
    policy = fit_global_threshold(stream, RewardModel())
    assert policy.kind == "global_threshold"
    assert len(set(policy.thresholds)) == 1
    assert 0.50 < policy.thresholds[0] <= 0.96


def test_the_swept_grid_is_reported_with_the_reward_it_measured() -> None:
    stream = _stream(
        truth=[FieldType.EMAIL] * 4 + [FieldType.CC_NUMBER] * 4,
        predicted=[FieldType.EMAIL] * 8,
        confidence=[0.99, 0.98, 0.97, 0.96, 0.50, 0.40, 0.30, 0.20],
    )
    sweep = sweep_global_threshold(stream, RewardModel(), n_points=11)
    assert len(sweep) == 11
    assert sweep[0][0] == 0.0
    assert sweep[-1][0] == pytest.approx(1.0)
    assert sweep[0][1] == pytest.approx(
        evaluate_policy(always_fill(), stream, RewardModel()).expected_reward
    )


def test_per_type_thresholds_split_the_two_types_a_single_threshold_cannot() -> None:
    """The case the whole per type policy exists for.

    One type is right whenever it is confident and the other is wrong at the
    same confidences, so no single global cut separates them and two do.
    """
    stream = _stream(
        truth=[FieldType.EMAIL] * 30 + [FieldType.CC_NUMBER] * 30,
        predicted=[FieldType.EMAIL] * 30 + [FieldType.CC_NUMBER] * 15 + [FieldType.EMAIL] * 15,
        confidence=[0.8] * 30 + [0.8] * 30,
    )
    reward = RewardModel()
    per_type = fit_per_type_thresholds(stream, reward, min_support=1)
    best_global = fit_global_threshold(stream, reward)
    assert (
        evaluate_policy(per_type, stream, reward).expected_reward
        > evaluate_policy(best_global, stream, reward).expected_reward
    )


def test_per_type_thresholds_are_at_least_as_good_as_always_filling_in_sample() -> None:
    """Always filling is the threshold vector of zeros, so it is in the class."""
    rng = np.random.default_rng(3)
    n = 600
    truth = rng.integers(0, len(FIELD_TYPES), n)
    predicted = np.where(rng.random(n) < 0.7, truth, rng.integers(0, len(FIELD_TYPES), n))
    stream = DecisionStream.build(
        truth=truth.tolist(),
        predicted=predicted.tolist(),
        confidence=rng.random(n).tolist(),
        locales=["en_US"] * n,
    )
    reward = RewardModel()
    fitted = fit_per_type_thresholds(stream, reward, min_support=1)
    assert (
        evaluate_policy(fitted, stream, reward).expected_reward
        >= evaluate_policy(always_fill(), stream, reward).expected_reward
    )


def test_a_type_the_fitting_split_never_predicted_inherits_the_global_threshold() -> None:
    stream = _stream(
        truth=[FieldType.EMAIL] * 4 + [FieldType.CC_NUMBER] * 4,
        predicted=[FieldType.EMAIL] * 8,
        confidence=[0.99, 0.98, 0.97, 0.96, 0.50, 0.40, 0.30, 0.20],
    )
    reward = RewardModel()
    fitted = fit_per_type_thresholds(stream, reward, min_support=1)
    inherited = fitted.thresholds[index_of(FieldType.USERNAME)]
    assert inherited == pytest.approx(fit_global_threshold(stream, reward).thresholds[0])


def test_a_type_below_the_support_floor_inherits_the_global_threshold() -> None:
    """Three rows of a class are not evidence about that class."""
    stream = _stream(
        truth=[FieldType.EMAIL] * 40 + [FieldType.CC_NUMBER] * 3,
        predicted=[FieldType.EMAIL] * 40 + [FieldType.CC_NAME] * 3,
        confidence=[0.9] * 43,
    )
    reward = RewardModel()
    fitted = fit_per_type_thresholds(stream, reward, min_support=DEFAULT_MIN_SUPPORT)
    assert fitted.thresholds[index_of(FieldType.CC_NAME)] == pytest.approx(
        fit_global_threshold(stream, reward).thresholds[0]
    )


# --------------------------------------------------------------------------
# The Thompson sampling simulation
# --------------------------------------------------------------------------


def _learnable_stream(n: int = 800, seed: int = 5) -> DecisionStream:
    """A stream where confidence really does separate right from wrong."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, len(FIELD_TYPES), n)
    confidence = rng.random(n)
    right = rng.random(n) < confidence
    predicted = np.where(right, truth, (truth + 1) % len(FIELD_TYPES))
    return DecisionStream.build(
        truth=truth.tolist(),
        predicted=predicted.tolist(),
        confidence=confidence.tolist(),
        locales=["de_DE" if value else "en_US" for value in rng.integers(0, 2, n)],
    )


def test_the_thompson_simulation_is_deterministic_under_a_seed() -> None:
    stream = _learnable_stream()
    reward = RewardModel()
    benchmark = reward.fill_rewards(stream) * 0
    first = simulate_thompson(stream, reward, benchmark, ThompsonConfig(seed=1))
    second = simulate_thompson(stream, reward, benchmark, ThompsonConfig(seed=1))
    assert first.actions.tolist() == second.actions.tolist()
    assert first.total_reward == second.total_reward
    assert first.to_dict() == second.to_dict()


def test_a_different_seed_gives_a_different_run() -> None:
    stream = _learnable_stream()
    reward = RewardModel()
    benchmark = reward.fill_rewards(stream) * 0
    first = simulate_thompson(stream, reward, benchmark, ThompsonConfig(seed=1))
    second = simulate_thompson(stream, reward, benchmark, ThompsonConfig(seed=2))
    assert first.actions.tolist() != second.actions.tolist()


def test_the_bandit_learns_to_stop_filling_the_cell_that_never_pays() -> None:
    """Online learning, measured rather than asserted: the fill rate falls."""
    stream = _learnable_stream()
    reward = RewardModel()
    result = simulate_thompson(stream, reward, reward.fill_rewards(stream) * 0, ThompsonConfig())
    low = stream.confidence[result.order] < 0.2
    early = result.actions[low][: low.sum() // 2]
    late = result.actions[low][low.sum() // 2 :]
    assert early.mean() > late.mean()
    assert late.mean() < 0.35


def test_regret_is_measured_against_the_best_fixed_policy_row_by_row() -> None:
    stream = _learnable_stream()
    reward = RewardModel()
    best = fit_per_type_thresholds(stream, reward, min_support=1)
    benchmark = reward.fill_rewards(stream) * best.fill_mask(stream)
    result = simulate_thompson(stream, reward, benchmark, ThompsonConfig(seed=0))
    assert result.cumulative_regret == pytest.approx(
        float(benchmark.sum()) - result.total_reward, abs=1e-9
    )
    assert result.regret_per_decision == pytest.approx(result.cumulative_regret / stream.n_rows)


def test_the_bandit_pays_for_what_it_has_to_learn() -> None:
    """A bandit that never explored would have nothing to be sorry about."""
    stream = _learnable_stream()
    reward = RewardModel()
    best = fit_per_type_thresholds(stream, reward, min_support=1)
    benchmark = reward.fill_rewards(stream) * best.fill_mask(stream)
    result = simulate_thompson(stream, reward, benchmark, ThompsonConfig(seed=0))
    assert result.cumulative_regret > 0
    assert 0 < result.n_filled < stream.n_rows


# --------------------------------------------------------------------------
# The comparison, its artifact, and the acceptance criterion
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> list:
    return generate_fields(GeneratorConfig(n_fields=2400), seed=0)


@pytest.fixture(scope="module")
def streams(corpus: list) -> tuple[DecisionStream, DecisionStream]:
    """`(val, test)` streams from the real head, calibrated as evaluate does."""
    trained = train(
        featurise([r for r in corpus if r.split == "train"]),
        featurise([r for r in corpus if r.split == "val"]),
        TrainConfig(epochs=4, seed=0),
    ).model
    built = []
    for split in ("val", "test"):
        rows = [r for r in corpus if r.split == split]
        scored = evaluate_engines(rows, trained)["ngram"]
        built.append(stream_from_records(rows, scored.predicted, scored.confidence))
    return built[0], built[1]


def test_the_per_type_policy_beats_always_filling_on_expected_reward(
    streams: tuple[DecisionStream, DecisionStream],
) -> None:
    """Acceptance criterion 3, the policy half."""
    validation, _ = streams
    reward = RewardModel()
    fitted = fit_per_type_thresholds(validation, reward)
    assert (
        evaluate_policy(fitted, validation, reward).expected_reward
        > evaluate_policy(always_fill(), validation, reward).expected_reward
    )


def test_the_per_type_policy_still_beats_always_filling_out_of_sample(
    streams: tuple[DecisionStream, DecisionStream],
) -> None:
    """Thresholds fitted on val, scored on test: the claim that is not free."""
    validation, held_out = streams
    reward = RewardModel()
    fitted = fit_per_type_thresholds(validation, reward)
    assert (
        evaluate_policy(fitted, held_out, reward).expected_reward
        > evaluate_policy(always_fill(), held_out, reward).expected_reward
    )


def test_the_comparison_ranks_every_policy_and_names_the_chosen_one(
    streams: tuple[DecisionStream, DecisionStream],
) -> None:
    validation, held_out = streams
    comparison = compare_policies(validation, held_out, RewardModel(), fit_split="val")
    names = [outcome.name for outcome in comparison.outcomes]
    assert names == ["never_fill", "always_fill", "global_threshold", "per_type_threshold"]
    assert comparison.chosen == "per_type_threshold"
    assert comparison.chosen_policy().kind == "per_type_threshold"
    assert comparison.best_fixed().expected_reward >= max(
        outcome.expected_reward for outcome in comparison.outcomes
    )
    assert comparison.in_sample is False


def test_a_comparison_fitted_on_the_split_it_scores_says_so(
    streams: tuple[DecisionStream, DecisionStream],
) -> None:
    validation, _ = streams
    comparison = compare_policies(validation, validation, RewardModel(), fit_split="val")
    assert comparison.in_sample is True


def test_the_comparison_round_trips_through_json(
    streams: tuple[DecisionStream, DecisionStream],
) -> None:
    validation, held_out = streams
    comparison = compare_policies(validation, held_out, RewardModel(), fit_split="val")
    restored = PolicyComparison.from_dict(json.loads(json.dumps(comparison.to_dict())))
    assert restored.to_dict() == comparison.to_dict()
    assert restored.chosen_policy() == comparison.chosen_policy()


def test_the_exported_policy_decides_the_same_way_after_a_round_trip(
    streams: tuple[DecisionStream, DecisionStream], tmp_path: Path
) -> None:
    """What part 12 consumes: a policy that is the same on every run."""
    validation, held_out = streams
    comparison = compare_policies(validation, held_out, RewardModel(), fit_split="val")
    path = write_policy(tmp_path / "policy.json", comparison)
    loaded = load_policy(path).chosen_policy()
    original = comparison.chosen_policy()
    for field_type in FIELD_TYPES:
        for confidence in (0.0, 0.5, 0.9, 0.99, 1.0):
            assert loaded.decide(field_type, confidence) == original.decide(field_type, confidence)


def test_a_policy_artifact_is_written_beside_the_evaluation(tmp_path: Path, corpus: list) -> None:
    trained = train(
        featurise([r for r in corpus if r.split == "train"]),
        featurise([r for r in corpus if r.split == "val"]),
        TrainConfig(epochs=3, seed=0),
    ).model
    paths = write_evaluation(
        tmp_path,
        records=[r for r in corpus if r.split == "val"],
        model=trained,
        split="val",
        locale="all",
        bootstrap=2,
        seed=0,
    )
    assert paths.policy is not None
    payload = json.loads(paths.policy.read_text(encoding="utf-8"))
    assert payload["chosen"] == "per_type_threshold"
    assert payload["in_sample"] is True
    names = [row["name"] for row in payload["policies"]]
    assert "always_fill" in names and "per_type_threshold" in names
    assert payload["thompson"]["best_fixed"] in names


def test_the_heuristic_evaluation_exports_no_policy(tmp_path: Path, corpus: list) -> None:
    """A decision layer over uncalibrated tiers would be a threshold over six values."""
    paths = write_evaluation(
        tmp_path,
        records=[r for r in corpus if r.split == "val"],
        model=None,
        split="val",
        locale="all",
        bootstrap=2,
        seed=0,
    )
    assert paths.policy is None
    assert not (tmp_path / "policy.json").exists()


def test_a_threshold_policy_is_what_it_serialises_to() -> None:
    policy = global_threshold(0.75)
    assert ThresholdPolicy.from_dict(policy.to_dict()) == policy
