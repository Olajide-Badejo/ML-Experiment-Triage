"""The pure numpy classifier, and the run directory its trainer writes.

Two families of test. The first is that the optimiser optimises: the loss falls,
the seed determines the result exactly, and the regularisation and the learning
rate are knobs rather than decoration. The second is the one acceptance
criterion 1 rests on: the trainer writes a directory `triage ingest` reads with
no autofill specific code anywhere in the ingest layer, so the assertion is made
against the real parser rather than against a description of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage.autofill.features import FEATURE_DIM, featurise
from triage.autofill.generator import GeneratorConfig, generate_fields
from triage.autofill.model import (
    METRIC_TAGS,
    LogisticModel,
    TrainConfig,
    accuracy,
    confusion_matrix,
    macro_f1,
    train,
    write_run_dir,
)
from triage.autofill.taxonomy import FIELD_TYPES
from triage.parsers.json_parser import JsonlParser


@pytest.fixture(scope="module")
def corpus() -> tuple[object, object]:
    records = generate_fields(GeneratorConfig(n_fields=1500), seed=0)
    train_rows = [r for r in records if r.split == "train"]
    val_rows = [r for r in records if r.split == "val"]
    return featurise(train_rows), featurise(val_rows)


@pytest.fixture(scope="module")
def trained(
    corpus: tuple,
) -> object:
    train_matrix, val_matrix = corpus
    return train(train_matrix, val_matrix, TrainConfig(epochs=4, seed=0))


def test_the_metric_tags_are_the_five_the_specification_names() -> None:
    assert METRIC_TAGS == ("train/loss", "val/loss", "val/accuracy", "val/macro_f1")


def test_training_lowers_the_training_loss(trained) -> None:  # type: ignore[no-untyped-def]
    losses = [point["train/loss"] for point in trained.history]
    assert losses[-1] < losses[0] * 0.6


def test_training_learns_something_better_than_the_majority_class(trained) -> None:  # type: ignore[no-untyped-def]
    assert trained.history[-1]["val/accuracy"] > 0.5
    assert trained.history[-1]["val/macro_f1"] > 0.4


def test_the_history_is_long_enough_for_the_default_comparison_window() -> None:
    """`ComparisonConfig.window_minimum` is 20, so a shorter run is untestable."""
    assert TrainConfig().eval_points >= 20


def test_the_same_seed_gives_the_same_weights(corpus: tuple) -> None:
    train_matrix, val_matrix = corpus
    first = train(train_matrix, val_matrix, TrainConfig(epochs=2, seed=5))
    second = train(train_matrix, val_matrix, TrainConfig(epochs=2, seed=5))
    np.testing.assert_array_equal(first.model.weights, second.model.weights)
    assert first.history == second.history


def test_a_different_seed_gives_different_weights(corpus: tuple) -> None:
    train_matrix, val_matrix = corpus
    first = train(train_matrix, val_matrix, TrainConfig(epochs=2, seed=5))
    second = train(train_matrix, val_matrix, TrainConfig(epochs=2, seed=6))
    assert not np.array_equal(first.model.weights, second.model.weights)


def test_l2_pulls_the_weights_towards_zero(corpus: tuple) -> None:
    train_matrix, val_matrix = corpus
    loose = train(train_matrix, val_matrix, TrainConfig(epochs=3, l2=0.0, seed=0))
    tight = train(train_matrix, val_matrix, TrainConfig(epochs=3, l2=0.05, seed=0))
    assert float(np.abs(tight.model.weights).sum()) < float(np.abs(loose.model.weights).sum())


def test_the_weights_have_the_shape_of_the_feature_space_and_the_head(trained) -> None:  # type: ignore[no-untyped-def]
    assert trained.model.weights.shape == (FEATURE_DIM, len(FIELD_TYPES))
    assert trained.model.weights.dtype == np.float32


def test_probabilities_are_a_distribution_over_the_head(trained, corpus: tuple) -> None:  # type: ignore[no-untyped-def]
    _, val_matrix = corpus
    probabilities = trained.model.probabilities(val_matrix)
    assert probabilities.shape == (len(val_matrix), len(FIELD_TYPES))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert probabilities.min() >= 0.0


def test_the_temperature_flattens_the_probabilities_without_moving_the_argmax(
    trained,  # type: ignore[no-untyped-def]
    corpus: tuple,
) -> None:
    _, val_matrix = corpus
    sharp = trained.model.probabilities(val_matrix, temperature=1.0)
    flat = trained.model.probabilities(val_matrix, temperature=3.0)
    np.testing.assert_array_equal(sharp.argmax(axis=1), flat.argmax(axis=1))
    assert flat.max(axis=1).mean() < sharp.max(axis=1).mean()


def test_a_temperature_of_zero_is_refused(trained) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="above zero"):
        trained.model.probabilities_from_logits(np.zeros((2, len(FIELD_TYPES))), temperature=0.0)


def test_saving_and_loading_a_model_preserves_every_prediction(
    trained,  # type: ignore[no-untyped-def]
    corpus: tuple,
    tmp_path: Path,
) -> None:
    _, val_matrix = corpus
    path = tmp_path / "weights.npz"
    trained.model.save(path)
    reloaded = LogisticModel.load(path)
    np.testing.assert_array_equal(trained.model.predict(val_matrix), reloaded.predict(val_matrix))
    np.testing.assert_array_equal(trained.model.weights, reloaded.weights)
    assert reloaded.temperature == trained.model.temperature


def test_a_weights_file_from_another_taxonomy_is_refused(
    trained,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """A silently reindexed head would report an accuracy that means nothing."""
    path = tmp_path / "weights.npz"
    np.savez_compressed(
        path,
        weights=trained.model.weights,
        tokens=np.asarray(["given-name", "made-up"], dtype=object),
        temperature=np.float64(1.0),
    )
    with pytest.raises(ValueError, match="taxonomy"):
        LogisticModel.load(path)


def test_accuracy_and_macro_f1_agree_with_hand_computed_values() -> None:
    truth = np.array([0, 0, 1, 1, 2])
    predicted = np.array([0, 1, 1, 1, 0])
    assert accuracy(truth, predicted) == pytest.approx(3 / 5)
    # class 0: precision 1/2, recall 1/2, f1 1/2
    # class 1: precision 2/3, recall 1, f1 4/5
    # class 2: precision 0, recall 0, f1 0
    # macro over the three classes PRESENT in the data.
    assert macro_f1(truth, predicted, n_classes=3) == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_macro_f1_ignores_a_class_that_is_neither_true_nor_predicted() -> None:
    """A head of 19 classes on a split holding 8 must not be divided by 19."""
    truth = np.array([0, 0, 1])
    predicted = np.array([0, 0, 1])
    assert macro_f1(truth, predicted, n_classes=19) == pytest.approx(1.0)


def test_the_confusion_matrix_counts_every_row_once() -> None:
    truth = np.array([0, 0, 1, 2])
    predicted = np.array([0, 1, 1, 0])
    matrix = confusion_matrix(truth, predicted, n_classes=3)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 4
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1 and matrix[2, 0] == 1


def test_the_run_directory_is_exactly_what_triage_ingest_reads(
    trained,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    """Acceptance criterion 1, asserted against the real parser.

    Nothing in `triage.parsers` knows this package exists, and this is the test
    that would fail first if that ever stopped being true.
    """
    run = tmp_path / "lr0.1_l20.0001_seed0"
    write_run_dir(run, trained, TrainConfig(seed=0), locale_mix="en_US:0.50,de_DE:0.50")

    parser = JsonlParser()
    assert parser.can_parse(run)
    experiment = parser.parse(run, root=tmp_path)
    assert set(METRIC_TAGS) <= set(experiment.tags)
    assert len(experiment.series("val/macro_f1")) == TrainConfig().eval_points
    assert experiment.config["seed"] == 0
    assert experiment.variant_key == TrainConfig().variant == "lr0.1_l20.0001"


def test_the_config_holds_every_key_the_specification_names(
    trained,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    write_run_dir(run, trained, TrainConfig(), locale_mix="en_US:0.50,de_DE:0.50")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for key in ("learning_rate", "batch_size", "l2", "n_features", "locale_mix", "seed"):
        assert key in config, key
    assert config["n_features"] == FEATURE_DIM


def test_the_run_directory_is_byte_stable_for_one_seed(corpus: tuple, tmp_path: Path) -> None:
    train_matrix, val_matrix = corpus
    config = TrainConfig(epochs=2, seed=3)
    written = []
    for name in ("a", "b"):
        result = train(train_matrix, val_matrix, config)
        run = tmp_path / name
        write_run_dir(run, result, config, locale_mix="en_US:0.50,de_DE:0.50")
        written.append(run)
    for name in ("config.json", "metrics.jsonl"):
        assert (written[0] / name).read_bytes() == (written[1] / name).read_bytes(), name


def test_the_weights_are_written_beside_the_metrics(
    trained,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    write_run_dir(run, trained, TrainConfig(), locale_mix="en_US:1.00")
    assert (run / "weights.npz").exists()
    assert LogisticModel.load(run / "weights.npz").weights.shape == trained.model.weights.shape


def test_an_empty_training_split_is_refused(corpus: tuple) -> None:
    _, val_matrix = corpus
    empty = featurise([])
    with pytest.raises(ValueError, match="no training rows"):
        train(empty, val_matrix, TrainConfig())
