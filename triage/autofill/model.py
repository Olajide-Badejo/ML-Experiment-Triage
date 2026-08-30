"""Multinomial logistic regression in pure numpy, and the run directory it writes.

**Why the smallest model that could work.** This is a demo workload and a CI
fixture, not an entry in a leaderboard. A linear model over hashed character n
grams trains in about a second on a laptop CPU, has two hyperparameters worth
sweeping, produces probabilities that are worth calibrating, and is wrong often
enough to be interesting. Anything larger would make the sweep too slow to run
on every change, which is the only property that actually matters here.

**The optimiser.** Minibatch stochastic gradient descent with momentum, L2, and
a seeded reshuffle each epoch. Two details are worth stating because they are
approximations rather than the textbook update:

*Sparse updates.* A field switches on a few hundred of 65,536 slots, so a dense
L2 decay and a dense momentum step would cost more per step than the gradient
does, by a factor of about fifty. Both are therefore applied to the rows the
batch touches. The consequence is that a weight's momentum decays in ITS OWN
step count rather than in the global one, and its L2 pull is applied on the steps
it participates in. This is the update Vowpal Wabbit and every other sparse
linear trainer uses, it is what makes the sweep fit in two minutes, and it is
written down here rather than left to be discovered from the numbers.

*Binary features.* Every active feature has value one, so the forward pass is a
gather and a segment sum rather than a matrix product, and the gradient is a
scatter add. `np.add.reduceat` does the first; a single flattened `np.bincount`
over the batch's distinct slots does the second, which is a great deal faster
than `np.add.at` on the full weight matrix.

**The run directory is the contract with the rest of this tool** (acceptance
criterion 1). `config.json` plus `metrics.jsonl` is what `triage ingest` reads,
and this module writes exactly that and nothing bespoke, so the sweep lands in
the comparison layer without a line of autofill specific code anywhere in it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triage.autofill.features import FEATURE_DIM, FeatureMatrix
from triage.autofill.taxonomy import FIELD_TYPES

#: The series a training run logs, per 5.2. `step` is the index rather than a
#: tag, which is what the JSONL parser expects.
METRIC_TAGS: tuple[str, ...] = ("train/loss", "val/loss", "val/accuracy", "val/macro_f1")

#: Rows scored at once when the whole split is evaluated. Large enough that the
#: per call overhead disappears, small enough that the gathered block stays in
#: cache rather than becoming a hundred megabyte temporary.
EVAL_CHUNK = 1024


@dataclass(frozen=True)
class TrainConfig:
    """Every knob of one training run.

    `eval_points` is the length of the logged series and is not a detail: the
    comparison layer's default window is the last ten percent of a run subject to
    a floor of twenty points, so a run logging fewer than twenty is a run no
    calibrated test can be applied to.
    """

    learning_rate: float = 0.1
    batch_size: int = 64
    l2: float = 1e-4
    momentum: float = 0.9
    epochs: int = 8
    seed: int = 0
    eval_points: int = 40

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be above zero; got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}")
        if self.l2 < 0:
            raise ValueError(f"l2 must not be negative; got {self.l2}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1); got {self.momentum}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1; got {self.epochs}")
        if self.eval_points < 2:
            raise ValueError(f"a run needs at least 2 logged points; got {self.eval_points}")
        if self.seed < 0:
            raise ValueError(f"seed must not be negative; got {self.seed}")

    @property
    def variant(self) -> str:
        """The condition key, which is everything but the seed."""
        return f"lr{self.learning_rate:g}_l2{self.l2:g}"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row wise softmax, shifted by the row maximum so it cannot overflow."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    total: np.ndarray = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return total


def confusion_matrix(truth: np.ndarray, predicted: np.ndarray, n_classes: int) -> np.ndarray:
    """`matrix[true, predicted]`, counted with one pass of `bincount`."""
    flat = np.asarray(truth, dtype=np.int64) * n_classes + np.asarray(predicted, dtype=np.int64)
    counts = np.bincount(flat, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes)


def accuracy(truth: np.ndarray, predicted: np.ndarray) -> float:
    if truth.size == 0:
        return float("nan")
    return float(np.mean(truth == predicted))


def per_class_scores(
    truth: np.ndarray, predicted: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per class precision, recall and F1, with an empty class scored zero."""
    matrix = confusion_matrix(truth, predicted, n_classes)
    correct = np.diag(matrix).astype(np.float64)
    predicted_totals = matrix.sum(axis=0).astype(np.float64)
    true_totals = matrix.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted_totals > 0, correct / predicted_totals, 0.0)
        recall = np.where(true_totals > 0, correct / true_totals, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)
    return precision, recall, f1


def macro_f1(truth: np.ndarray, predicted: np.ndarray, n_classes: int = len(FIELD_TYPES)) -> float:
    """Unweighted mean F1 over the classes the split actually involves.

    Averaging over all nineteen classes when a split holds eight would divide by
    eleven zeros and report a number that is mostly a fact about the taxonomy.
    The classes counted are those that appear as a truth OR as a prediction, so a
    model that invents a class it was never asked for is penalised for it rather
    than having the invention quietly dropped.
    """
    if truth.size == 0:
        return float("nan")
    matrix = confusion_matrix(truth, predicted, n_classes)
    present = (matrix.sum(axis=0) + matrix.sum(axis=1)) > 0
    _, _, f1 = per_class_scores(truth, predicted, n_classes)
    return float(f1[present].mean())


@dataclass
class LogisticModel:
    """Weights, plus the one scalar calibration this project cares about.

    The temperature travels WITH the weights rather than beside them because a
    probability read off an uncalibrated head and a probability read off a
    calibrated one are different numbers, and a decision layer downstream has no
    way to tell which it was handed. See `calibration.py`.
    """

    weights: np.ndarray
    temperature: float = 1.0

    @property
    def n_features(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.weights.shape[1])

    def logits(self, matrix: FeatureMatrix) -> np.ndarray:
        """Unnormalised scores for every row, in chunks to bound the temporary."""
        rows = len(matrix)
        out = np.empty((rows, self.n_classes), dtype=np.float32)
        for start in range(0, rows, EVAL_CHUNK):
            stop = min(start + EVAL_CHUNK, rows)
            out[start:stop] = _batch_logits(self.weights, matrix, start, stop)
        return out

    def probabilities_from_logits(
        self, logits: np.ndarray, temperature: float | None = None
    ) -> np.ndarray:
        scale = self.temperature if temperature is None else temperature
        if not scale > 0:
            raise ValueError(f"a temperature must be above zero; got {scale}")
        return softmax(np.asarray(logits, dtype=np.float64) / scale)

    def probabilities(self, matrix: FeatureMatrix, temperature: float | None = None) -> np.ndarray:
        return self.probabilities_from_logits(self.logits(matrix), temperature)

    def predict(self, matrix: FeatureMatrix) -> np.ndarray:
        """The argmax class per row. Temperature free: scaling cannot move it."""
        return np.asarray(self.logits(matrix).argmax(axis=1), dtype=np.int64)

    def save(self, path: Path | str) -> Path:
        """Persist the weights, the temperature and the taxonomy they assume."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            weights=self.weights,
            tokens=np.asarray([member.value for member in FIELD_TYPES], dtype=object),
            temperature=np.float64(self.temperature),
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> LogisticModel:
        """Read a weights file, refusing one written against another taxonomy.

        The head index of a class is only meaningful relative to the taxonomy it
        was trained under, so a file listing different tokens would produce an
        accuracy computed against the wrong labels and no error at all.
        """
        with np.load(Path(path), allow_pickle=True) as archive:
            tokens = [str(token) for token in archive["tokens"]]
            expected = [member.value for member in FIELD_TYPES]
            if tokens != expected:
                raise ValueError(
                    f"{path} was trained against a different taxonomy "
                    f"({len(tokens)} classes: {', '.join(tokens[:4])}...); this build has "
                    f"{len(expected)}. A head index only means something relative to the "
                    f"taxonomy it was fitted under"
                )
            return cls(
                weights=np.asarray(archive["weights"], dtype=np.float32),
                temperature=float(archive["temperature"]),
            )


@dataclass
class TrainResult:
    """A trained model and the series its training logged."""

    model: LogisticModel
    history: list[dict[str, float]] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    seconds: float = 0.0


def _batch_logits(weights: np.ndarray, matrix: FeatureMatrix, start: int, stop: int) -> np.ndarray:
    """Segment sums of the weight rows every field in `[start, stop)` switches on.

    `reduceat` needs every segment to be non empty, which holds by construction:
    `feature_indices` always emits the bias token, so no row is ever empty.
    """
    first = int(matrix.indptr[start])
    last = int(matrix.indptr[stop])
    segment = matrix.indices[first:last]
    offsets = (matrix.indptr[start:stop] - first).astype(np.intp)
    gathered = weights[segment]
    summed: np.ndarray = np.add.reduceat(gathered, offsets, axis=0)
    return summed


def _cross_entropy(probabilities: np.ndarray, classes: np.ndarray) -> float:
    """Mean negative log likelihood, floored so a zero cannot become an infinity."""
    picked = probabilities[np.arange(classes.size), classes]
    return float(-np.log(np.maximum(picked, 1e-12)).mean())


def _evaluate(model: LogisticModel, matrix: FeatureMatrix) -> dict[str, float]:
    logits = model.logits(matrix)
    probabilities = softmax(logits.astype(np.float64))
    predicted = np.asarray(logits.argmax(axis=1), dtype=np.int64)
    return {
        "val/loss": _cross_entropy(probabilities, matrix.classes),
        "val/accuracy": accuracy(matrix.classes, predicted),
        "val/macro_f1": macro_f1(matrix.classes, predicted, model.n_classes),
    }


def _eval_steps(total_steps: int, eval_points: int) -> list[int]:
    """Evenly spaced step numbers to log at, ending on the last step.

    Ending on the last step matters: the window statistic is taken from the end
    of the series, and a run whose final logged point was ten steps before it
    stopped would be compared on a model that is not the one it saved.
    """
    raw = np.linspace(total_steps / eval_points, total_steps, eval_points)
    return sorted({int(value) for value in np.rint(raw).astype(np.int64)})


def train(
    train_matrix: FeatureMatrix,
    val_matrix: FeatureMatrix,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Fit the head and log the series `triage ingest` will read.

    Seeding is `numpy.random.SeedSequence` with the epoch as the spawn key, so
    epoch three draws the same permutation whether the run stopped at four
    epochs or at twenty, and adding an epoch cannot rewrite the ones before it.
    """
    import time

    config = config or TrainConfig()
    n_train = len(train_matrix)
    if n_train == 0:
        raise ValueError("no training rows: the train split of this corpus is empty")

    n_classes = len(FIELD_TYPES)
    weights = np.zeros((train_matrix.n_features, n_classes), dtype=np.float32)
    velocity = np.zeros_like(weights)

    steps_per_epoch = math.ceil(n_train / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    log_at = set(_eval_steps(total_steps, config.eval_points))

    model = LogisticModel(weights=weights)
    history: list[dict[str, float]] = []
    running: list[float] = []
    step = 0
    started = time.perf_counter()

    for epoch in range(config.epochs):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=config.seed, spawn_key=(epoch,)))
        shuffled = train_matrix.take(rng.permutation(n_train))
        for start in range(0, n_train, config.batch_size):
            stop = min(start + config.batch_size, n_train)
            step += 1
            running.append(_step(weights, velocity, shuffled, start, stop, config, n_classes))
            if step in log_at:
                point = {"step": float(step), "train/loss": float(np.mean(running))}
                point.update(_evaluate(model, val_matrix))
                history.append(point)
                running = []

    return TrainResult(
        model=model,
        history=history,
        n_train=n_train,
        n_val=len(val_matrix),
        seconds=time.perf_counter() - started,
    )


def _step(
    weights: np.ndarray,
    velocity: np.ndarray,
    matrix: FeatureMatrix,
    start: int,
    stop: int,
    config: TrainConfig,
    n_classes: int,
) -> float:
    """One minibatch: forward, softmax cross entropy, sparse momentum update."""
    size = stop - start
    logits = _batch_logits(weights, matrix, start, stop).astype(np.float64)
    probabilities = softmax(logits)
    classes = matrix.classes[start:stop]
    loss = _cross_entropy(probabilities, classes)

    delta = probabilities
    delta[np.arange(size), classes] -= 1.0
    delta /= size

    first = int(matrix.indptr[start])
    last = int(matrix.indptr[stop])
    segment = matrix.indices[first:last]
    lengths = np.diff(matrix.indptr[start : stop + 1])
    per_nonzero = delta[np.repeat(np.arange(size), lengths)]

    touched, inverse = np.unique(segment, return_inverse=True)
    flat = (inverse[:, None] * n_classes + np.arange(n_classes)).ravel()
    gradient = np.bincount(
        flat, weights=per_nonzero.ravel(), minlength=touched.size * n_classes
    ).reshape(touched.size, n_classes)

    # L2 and momentum on the touched rows only: see the module docstring. The
    # cast keeps the accumulation in float64 and the storage in float32.
    if config.l2:
        gradient += config.l2 * weights[touched]
    updated = config.momentum * velocity[touched] - config.learning_rate * gradient
    velocity[touched] = updated.astype(np.float32)
    weights[touched] += updated.astype(np.float32)
    return loss


def write_run_dir(
    path: Path | str,
    result: TrainResult,
    config: TrainConfig,
    locale_mix: str,
    variant: str | None = None,
    extra_config: dict[str, Any] | None = None,
    save_weights: bool = True,
) -> Path:
    """Write `config.json`, `metrics.jsonl` and `weights.npz` into one run directory.

    This is the whole of the integration with the rest of the tool. There is no
    autofill aware parser, no special case in `triage.ingest`, and no flag: the
    trainer writes the format the tool already reads, which is what acceptance
    criterion 1 asks for and the only arrangement that could keep being true.

    `save_weights=False` is for the sweep, where thirty heads of 65,536 by 19
    float32 would be tens of megabytes of scaffolding for a comparison that reads
    none of it; the sweep keeps the best head alone, beside its own summary.
    """
    run = Path(path)
    run.mkdir(parents=True, exist_ok=True)

    settings: dict[str, Any] = {
        "variant": variant or config.variant,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "l2": config.l2,
        "momentum": config.momentum,
        "epochs": config.epochs,
        "n_features": result.model.n_features,
        "locale_mix": locale_mix,
        "seed": config.seed,
    }
    settings.update(extra_config or {})
    (run / "config.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    (run / "metrics.jsonl").write_text(
        "".join(
            json.dumps(
                {"step": int(point["step"]), **{tag: point[tag] for tag in METRIC_TAGS}},
                sort_keys=True,
            )
            + "\n"
            for point in result.history
        ),
        encoding="utf-8",
        newline="\n",
    )
    if save_weights:
        result.model.save(run / "weights.npz")
    return run


def training_config_dict(config: TrainConfig) -> dict[str, Any]:
    """The config as plain data, for metadata files that record what was run."""
    return dict(asdict(config))


#: Re exported so a caller does not have to import two modules to state the
#: feature space a run was trained in.
FEATURE_SPACE = FEATURE_DIM
