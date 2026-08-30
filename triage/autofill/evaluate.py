"""Scoring the two engines, and writing the two artifact shapes that result.

**Two engines, because one number is not a result.** The model is compared
against a heuristic that is a serious baseline rather than a straw one: it reads
the `autocomplete` attribute when there is one, and otherwise matches a keyword
table that contains every canonical term the specification names, in both
locales, longest phrase first so that `name on card` beats `name` and
`Bundesland` is never read as `Land`. That is roughly what a hand written rule
engine is, and on English it scores well. Where it loses is the synonym tail:
`Rufnummer`, `Wohnort`, `Sicherheitscode`, `Anschrift`, the things a keyword
table only has if somebody thought of them. Widening the table is exactly the
work `Autofill_audit` did 392 times over nine locales, and the point of the
comparison is to measure what that work is worth against learning it.

**Two artifact shapes, because this repository has two ingest paths and the demo
should exercise both.**

*Triage native.* One run directory per bootstrap replicate per engine, holding
`config.json` and a `metrics.jsonl` whose series is the CUMULATIVE macro F1 over
a resample of the evaluation split. The series is not a training curve and is not
pretending to be: every point is an estimate of the same quantity from a growing
prefix of the same resample, so the final window that `triage compare` takes is
the full resample estimate with a little extra variance, and the spread ACROSS
replicates is what the seed replicated permutation test consumes. Replicate `k`
is the seed, per 5.2, and the same resample scores both engines so the
comparison is paired at the replicate level rather than at the field level.

*The consumer's schema.* One `run.jsonl` row per classified field per engine, in
`Autofill_audit`'s E5 schema verbatim, which makes this repository's own demo the
fixture for its own `OutcomesParser` and `paired_permutation`: the ecosystem
contract and the reference workload check each other, and neither can rot alone.

**On `latency_us`.** The schema carries a per row latency and this writes the
measured per row MEAN for the engine rather than a per row timing, because at
this speed a per row `perf_counter` measures the clock rather than the classifier.
It is the one field of the outcomes file that is not a pure function of the seed,
and it is labelled here rather than left to be discovered.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from triage.autofill.calibration import (
    DEFAULT_BINS,
    calibrate,
    expected_calibration_error,
    maximum_calibration_error,
)
from triage.autofill.features import FeatureMatrix, featurise
from triage.autofill.generator import FieldRecord
from triage.autofill.model import (
    LogisticModel,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_scores,
    softmax,
)
from triage.autofill.taxonomy import FIELD_TYPES, FieldType, index_of

#: The schema version the consumer's rows declare, copied rather than invented.
OUTCOMES_SCHEMA_VERSION = "1.0.0"

#: CLI policy name to the `engine` value written into the outcomes rows. The
#: values are the consumer's own vocabulary, so their fixture and ours are the
#: same file shape down to the strings.
ENGINE_NAMES: dict[str, str] = {"model": "ngram", "heuristic": "rules", "llm": "llm"}

#: Points on a bootstrap replicate's cumulative curve. Sixty, so that the
#: comparison layer's default window (the last ten percent, floored at twenty
#: points) is the last third of the curve, every point of which is an estimate
#: from at least two thirds of the resample.
CURVE_POINTS = 60

#: Fewer rows than this and a cumulative curve is mostly its own warm up.
MINIMUM_EVAL_ROWS = 40

#: Series a bootstrap replicate logs. Both names carry their direction in them,
#: so `triage compare` infers "higher is better" without being told.
EVAL_TAGS: tuple[str, ...] = ("eval/macro_f1", "eval/accuracy")

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NOT_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]+")


def tokenise(text: str) -> tuple[str, ...]:
    """Lowercase word tokens, splitting camel case as well as punctuation.

    `billingPostalCode`, `billing_postal_code` and `Billing postal code` are one
    markup style apart and must tokenise the same way, or the keyword table
    would need three entries for every term and would still miss the fourth.
    """
    spaced = _CAMEL.sub(" ", text)
    return tuple(token for token in _NOT_ALPHANUMERIC.split(spaced.lower()) if token)


#: The keyword table, written as phrases. Every canonical term the specification
#: names for de_DE is here, plus the English equivalents, plus the handful of
#: attribute spellings a rule engine would obviously carry. What is NOT here is
#: the synonym tail, and that is the point: this is a good keyword table, not an
#: exhaustive one, and no keyword table is exhaustive.
KEYWORD_PHRASES: dict[str, FieldType] = {
    # en_US
    "first name": FieldType.GIVEN_NAME,
    "given name": FieldType.GIVEN_NAME,
    "firstname": FieldType.GIVEN_NAME,
    "fname": FieldType.GIVEN_NAME,
    "last name": FieldType.FAMILY_NAME,
    "family name": FieldType.FAMILY_NAME,
    "surname": FieldType.FAMILY_NAME,
    "lastname": FieldType.FAMILY_NAME,
    "lname": FieldType.FAMILY_NAME,
    "full name": FieldType.NAME,
    "name": FieldType.NAME,
    "email address": FieldType.EMAIL,
    "e mail": FieldType.EMAIL,
    "email": FieldType.EMAIL,
    "mail": FieldType.EMAIL,
    "phone number": FieldType.TEL,
    "telephone": FieldType.TEL,
    "phone": FieldType.TEL,
    "tel": FieldType.TEL,
    "street address": FieldType.ADDRESS_LINE1,
    "address line 1": FieldType.ADDRESS_LINE1,
    "address1": FieldType.ADDRESS_LINE1,
    "street": FieldType.ADDRESS_LINE1,
    "address": FieldType.ADDRESS_LINE1,
    "address line 2": FieldType.ADDRESS_LINE2,
    "address2": FieldType.ADDRESS_LINE2,
    "apartment": FieldType.ADDRESS_LINE2,
    "city": FieldType.ADDRESS_LEVEL2,
    "town": FieldType.ADDRESS_LEVEL2,
    "state": FieldType.ADDRESS_LEVEL1,
    "province": FieldType.ADDRESS_LEVEL1,
    "zip code": FieldType.POSTAL_CODE,
    "postal code": FieldType.POSTAL_CODE,
    "postcode": FieldType.POSTAL_CODE,
    "zip": FieldType.POSTAL_CODE,
    "country": FieldType.COUNTRY_NAME,
    "company": FieldType.ORGANIZATION,
    "organization": FieldType.ORGANIZATION,
    "name on card": FieldType.CC_NAME,
    "cardholder name": FieldType.CC_NAME,
    "cardholder": FieldType.CC_NAME,
    "card holder": FieldType.CC_NAME,
    "credit card number": FieldType.CC_NUMBER,
    "card number": FieldType.CC_NUMBER,
    "cardnumber": FieldType.CC_NUMBER,
    "expiration month": FieldType.CC_EXP_MONTH,
    "expiry month": FieldType.CC_EXP_MONTH,
    "exp month": FieldType.CC_EXP_MONTH,
    "mm": FieldType.CC_EXP_MONTH,
    "expiration year": FieldType.CC_EXP_YEAR,
    "expiry year": FieldType.CC_EXP_YEAR,
    "exp year": FieldType.CC_EXP_YEAR,
    "yyyy": FieldType.CC_EXP_YEAR,
    "security code": FieldType.CC_CSC,
    "cvc": FieldType.CC_CSC,
    "cvv": FieldType.CC_CSC,
    "username": FieldType.USERNAME,
    "user name": FieldType.USERNAME,
    "login": FieldType.USERNAME,
    # de_DE: every canonical term 5.2 names, and nothing beyond it.
    "vorname": FieldType.GIVEN_NAME,
    "nachname": FieldType.FAMILY_NAME,
    "e mail adresse": FieldType.EMAIL,
    "emailadresse": FieldType.EMAIL,
    "telefonnummer": FieldType.TEL,
    "telefon": FieldType.TEL,
    "strasse und hausnummer": FieldType.ADDRESS_LINE1,
    "strasse": FieldType.ADDRESS_LINE1,
    "adresszusatz": FieldType.ADDRESS_LINE2,
    "ort": FieldType.ADDRESS_LEVEL2,
    "bundesland": FieldType.ADDRESS_LEVEL1,
    "postleitzahl": FieldType.POSTAL_CODE,
    "plz": FieldType.POSTAL_CODE,
    "land": FieldType.COUNTRY_NAME,
    "firma": FieldType.ORGANIZATION,
    "karteninhaber": FieldType.CC_NAME,
    "kartennummer": FieldType.CC_NUMBER,
    "ablaufmonat": FieldType.CC_EXP_MONTH,
    "ablaufjahr": FieldType.CC_EXP_YEAR,
    "pruefziffer": FieldType.CC_CSC,
    "benutzername": FieldType.USERNAME,
}

#: The table keyed by token tuple, which is what the matcher walks.
KEYWORDS: dict[tuple[str, ...], FieldType] = {
    tokenise(phrase): field_type for phrase, field_type in KEYWORD_PHRASES.items()
}

#: The longest phrase in the table, so the matcher knows where to start.
LONGEST_PHRASE = max(len(key) for key in KEYWORDS)


class HeuristicClassifier:
    """The rule baseline: the attribute if it is there, then a keyword table.

    Its confidences are TIERED rather than estimated: a rule engine has no
    likelihood, so what it can honestly report is which rule fired. The tiers
    are ordered and plausible and they are not calibrated, which is the point
    `calibration.py` is making and which the reported ECE of this engine
    demonstrates rather than asserts.
    """

    AUTOCOMPLETE_CONFIDENCE = 0.95
    LABEL_CONFIDENCE = 0.80
    ATTRIBUTE_CONFIDENCE = 0.65
    TYPE_CONFIDENCE = 0.60
    PLACEHOLDER_CONFIDENCE = 0.55
    FALLBACK_CONFIDENCE = 0.30

    #: `input type -> field type`, for the two types that name one.
    TYPE_RULES: ClassVar[dict[str, FieldType]] = {
        "email": FieldType.EMAIL,
        "tel": FieldType.TEL,
    }

    @staticmethod
    def match(text: str) -> FieldType | None:
        """The longest keyword phrase anywhere in `text`, or nothing.

        Longest first is what keeps `name on card` from being read as `name` and
        `zip code` from being read as `zip`, and what makes a one token entry
        like `land` safe: `Bundesland` and `Lieferland` are single tokens that
        are not `land`, so they simply do not match.
        """
        tokens = tokenise(text)
        if not tokens:
            return None
        for size in range(min(LONGEST_PHRASE, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                found = KEYWORDS.get(tokens[start : start + size])
                if found is not None:
                    return found
        return None

    @classmethod
    def classify(cls, record: FieldRecord) -> tuple[FieldType, float]:
        """One field's predicted type and the confidence tier that produced it."""
        token = record.autocomplete
        if token and token not in {"off", "on"}:
            try:
                return FieldType(token), cls.AUTOCOMPLETE_CONFIDENCE
            except ValueError:
                # An attribute holding something outside the taxonomy is not a
                # reason to stop looking, only a reason not to trust it.
                pass
        found = cls.match(record.label)
        if found is not None:
            return found, cls.LABEL_CONFIDENCE
        for attribute in (record.name, record.element_id):
            found = cls.match(attribute)
            if found is not None:
                return found, cls.ATTRIBUTE_CONFIDENCE
        found = cls.match(record.placeholder)
        if found is not None:
            return found, cls.PLACEHOLDER_CONFIDENCE
        by_type = cls.TYPE_RULES.get(record.input_type)
        if by_type is not None:
            return by_type, cls.TYPE_CONFIDENCE
        return FieldType.UNKNOWN, cls.FALLBACK_CONFIDENCE


@dataclass(frozen=True)
class ScoredEngine:
    """One engine's predictions over one split."""

    engine: str
    predicted: np.ndarray
    confidence: np.ndarray
    latency_us: float


def heuristic_predict(records: list[FieldRecord]) -> tuple[np.ndarray, np.ndarray, float]:
    """`(predicted, confidence, mean latency in microseconds)` for the rules."""
    started = time.perf_counter()
    decided = [HeuristicClassifier.classify(record) for record in records]
    elapsed = time.perf_counter() - started
    predicted = np.asarray([index_of(field_type) for field_type, _ in decided], dtype=np.int64)
    confidence = np.asarray([score for _, score in decided], dtype=np.float64)
    per_row = (elapsed * 1e6 / len(records)) if records else 0.0
    return predicted, confidence, per_row


def model_predict(
    model: LogisticModel, matrix: FeatureMatrix
) -> tuple[np.ndarray, np.ndarray, float]:
    """The same, for the trained head, at whatever temperature it carries."""
    started = time.perf_counter()
    logits = model.logits(matrix)
    probabilities = softmax(logits.astype(np.float64) / model.temperature)
    elapsed = time.perf_counter() - started
    predicted = np.asarray(probabilities.argmax(axis=1), dtype=np.int64)
    confidence = probabilities[np.arange(predicted.size), predicted]
    rows = len(matrix)
    return predicted, confidence, (elapsed * 1e6 / rows) if rows else 0.0


def evaluate_engines(
    records: list[FieldRecord], model: LogisticModel | None
) -> dict[str, ScoredEngine]:
    """Score every engine available on these rows, keyed by engine name.

    The heuristic is always scored, because it is the baseline every other
    engine is read against and an evaluation with nothing to compare to is a
    number rather than a result.
    """
    predicted, confidence, latency = heuristic_predict(records)
    scored = {
        "rules": ScoredEngine(
            engine="rules", predicted=predicted, confidence=confidence, latency_us=latency
        )
    }
    if model is not None:
        matrix = featurise(records)
        predicted, confidence, latency = model_predict(model, matrix)
        scored["ngram"] = ScoredEngine(
            engine="ngram", predicted=predicted, confidence=confidence, latency_us=latency
        )
    return scored


def _bins_for(n_rows: int) -> int:
    """Fifteen bins unless there are too few rows to fill them."""
    return max(1, min(DEFAULT_BINS, n_rows // 4))


@dataclass(frozen=True)
class EvaluationReport:
    """Accuracy, macro F1, per type precision and recall, confusion, ECE."""

    engine: str
    split: str
    locale: str
    n_rows: int
    accuracy: float
    macro_f1: float
    ece: float
    mce: float
    mean_latency_us: float
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        engine: str,
        split: str,
        locale: str,
        records: list[FieldRecord],
        scored: ScoredEngine,
    ) -> EvaluationReport:
        truth = np.asarray([index_of(record.field_type) for record in records], dtype=np.int64)
        n_classes = len(FIELD_TYPES)
        precision, recall, _ = per_class_scores(truth, scored.predicted, n_classes)
        matrix = confusion_matrix(truth, scored.predicted, n_classes)
        support = matrix.sum(axis=1)
        correct = scored.predicted == truth
        bins = _bins_for(truth.size)
        return cls(
            engine=engine,
            split=split,
            locale=locale,
            n_rows=int(truth.size),
            accuracy=accuracy(truth, scored.predicted),
            macro_f1=macro_f1(truth, scored.predicted, n_classes),
            ece=expected_calibration_error(scored.confidence, correct, bins),
            mce=maximum_calibration_error(scored.confidence, correct, bins),
            mean_latency_us=scored.latency_us,
            per_type={
                member.value: {
                    "precision": float(precision[position]),
                    "recall": float(recall[position]),
                    "support": float(support[position]),
                }
                for position, member in enumerate(FIELD_TYPES)
                # A class with no rows in this split has no precision and no
                # recall, and printing 0.00 for both would read as a failure
                # rather than as an absence.
                if support[position] > 0
            },
            confusion=[[int(value) for value in row] for row in matrix],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "split": self.split,
            "locale": self.locale,
            "n_rows": self.n_rows,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "ece": self.ece,
            "mce": self.mce,
            "mean_latency_us": self.mean_latency_us,
            "per_type": self.per_type,
            "confusion": self.confusion,
        }


def outcomes_rows(
    records: list[FieldRecord],
    scored: dict[str, ScoredEngine],
    run_id: str,
    split: str,
) -> list[dict[str, Any]]:
    """The consumer's E5 rows: one per field per engine, keys in their order."""
    rows: list[dict[str, Any]] = []
    for name in sorted(scored):
        engine = scored[name]
        for position, record in enumerate(records):
            predicted = FIELD_TYPES[int(engine.predicted[position])]
            rows.append(
                {
                    "schema_version": OUTCOMES_SCHEMA_VERSION,
                    "run_id": run_id,
                    "engine": engine.engine,
                    "split": split,
                    "form_id": record.form_id,
                    "template_id": record.template_id,
                    "true_label": record.field_type.value,
                    "pred_label": predicted.value,
                    "correct": bool(predicted is record.field_type),
                    "confidence": round(float(engine.confidence[position]), 4),
                    "latency_us": round(float(engine.latency_us), 1),
                }
            )
    return rows


def cumulative_curve(
    truth: np.ndarray, predicted: np.ndarray, n_points: int = CURVE_POINTS
) -> list[dict[str, float]]:
    """Macro F1 and accuracy over growing prefixes of one resample.

    Every point estimates the same quantity from more of the same rows, so the
    curve converges rather than improving: it is an evaluation, not a training
    run, and the docstring says so because the chart in the report otherwise
    invites the opposite reading.
    """
    rows = int(truth.size)
    cuts = sorted({int(value) for value in np.rint(np.linspace(rows / n_points, rows, n_points))})
    return [
        {
            "step": float(cut),
            "eval/macro_f1": macro_f1(truth[:cut], predicted[:cut], len(FIELD_TYPES)),
            "eval/accuracy": accuracy(truth[:cut], predicted[:cut]),
        }
        for cut in cuts
        if cut > 0
    ]


@dataclass(frozen=True)
class EvaluationPaths:
    """Where `write_evaluation` put everything."""

    root: Path
    runs: list[Path]
    outcomes: Path
    report: Path
    calibration: Path | None = None


def _write_run(directory: Path, curve: list[dict[str, float]], settings: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (directory / "metrics.jsonl").write_text(
        "".join(
            json.dumps(
                {"step": int(point["step"]), **{tag: point[tag] for tag in EVAL_TAGS}},
                sort_keys=True,
            )
            + "\n"
            for point in curve
        ),
        encoding="utf-8",
        newline="\n",
    )
    return directory


def write_evaluation(
    out_dir: Path | str,
    records: list[FieldRecord],
    model: LogisticModel | None,
    split: str,
    locale: str,
    bootstrap: int,
    seed: int,
    run_id: str | None = None,
) -> EvaluationPaths:
    """Score, calibrate, resample, and write both artifact shapes.

    `locale` is `all` or one of the corpus locales; anything else is refused
    rather than silently scored on an empty split, which would report a macro F1
    of NaN and a verdict about nothing.
    """
    root = Path(out_dir)
    rows = [r for r in records if locale == "all" or r.locale == locale]
    if not rows:
        raise ValueError(
            f"no rows in split {split!r} for locale {locale!r}: the corpus holds "
            f"{', '.join(sorted({r.locale for r in records})) or 'nothing'}"
        )
    if len(rows) < MINIMUM_EVAL_ROWS:
        raise ValueError(
            f"only {len(rows)} row(s) to evaluate, below the floor of {MINIMUM_EVAL_ROWS}: "
            f"a bootstrap over this would report a spread rather than a result"
        )
    if bootstrap < 2:
        raise ValueError(
            f"a seed replicated comparison needs at least 2 replicates; got {bootstrap}"
        )

    calibration_path: Path | None = None
    scoring_model = model
    if model is not None:
        matrix = featurise(rows)
        summary = calibrate(model.logits(matrix), matrix.classes)
        scoring_model = LogisticModel(weights=model.weights, temperature=summary.temperature)
        calibration_path = root / "calibration.json"
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(
            json.dumps(
                {"calibration": summary.to_dict(), "metrics": summary.as_metrics()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    scored = evaluate_engines(rows, scoring_model)
    truth = np.asarray([index_of(record.field_type) for record in rows], dtype=np.int64)

    # One resample per replicate, shared by both engines: the comparison is
    # paired at the replicate level, so the difference between the engines is
    # not also carrying the difference between two draws.
    written: list[Path] = []
    for replicate in range(bootstrap):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(replicate,)))
        resample = rng.integers(0, truth.size, truth.size)
        for name in sorted(scored):
            engine = scored[name]
            curve = cumulative_curve(truth[resample], engine.predicted[resample])
            written.append(
                _write_run(
                    root / "runs" / f"{name}_{locale}_{split}_seed{replicate}",
                    curve,
                    {
                        "variant": f"{name}_{locale}_{split}",
                        "engine": name,
                        "locale": locale,
                        "split": split,
                        "seed": replicate,
                        "n_rows": int(truth.size),
                        "bootstrap": bootstrap,
                    },
                )
            )

    identity = run_id or f"autofill-eval-{locale}-{split}"
    outcomes_dir = root / "outcomes" / "autofill_eval"
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    (outcomes_dir / "config.json").write_text(
        json.dumps(
            {"variant": "autofill_eval", "locale": locale, "split": split, "seed": seed},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    outcomes = outcomes_dir / "run.jsonl"
    outcomes.write_text(
        "".join(
            json.dumps(row, sort_keys=False) + "\n"
            for row in outcomes_rows(rows, scored, identity, split)
        ),
        encoding="utf-8",
        newline="\n",
    )

    report_path = root / "evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "locale": locale,
                "split": split,
                "seed": seed,
                "bootstrap": bootstrap,
                "engines": {
                    name: EvaluationReport.build(name, split, locale, rows, engine).to_dict()
                    for name, engine in sorted(scored.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return EvaluationPaths(
        root=root,
        runs=written,
        outcomes=outcomes,
        report=report_path,
        calibration=calibration_path,
    )


def label_codes(labels: Sequence[Any]) -> np.ndarray:
    """Taxonomy tokens as head indices, for a statistic over label columns.

    The outcomes file carries `true_label` and `pred_label` as text, because
    they are categorical group keys rather than measurements. A statistic
    computed over predictions needs them as numbers, and the mapping has to be
    the taxonomy's own so that a token spelled by another repository lands in
    the same slot.
    """
    return np.asarray([index_of(FieldType(str(label))) for label in labels], dtype=np.int64)


def encode_outcome(true_labels: Sequence[Any], pred_labels: Sequence[Any]) -> np.ndarray:
    """One integer per row carrying BOTH the truth and the prediction.

    `truth * n_classes + prediction`, which is the same packing
    `confusion_matrix` uses internally, and it is what makes macro F1 usable as a
    `paired_permutation` statistic AT ALL.

    The reason is worth stating, because getting it wrong looks like it works.
    A statistic there has to be a pure function of the vector it is handed, and
    `paired_permutation` hands it two different kinds of vector: the label
    swapped ones, which keep every row in place, and the CLUSTER BOOTSTRAP
    resamples, which draw whole clusters with replacement and therefore hand back
    a vector of a different length in a different order. A closure holding a
    fixed truth vector beside the predictions satisfies the first and breaks on
    the second, loudly if the lengths differ and silently if they happen to
    match. Packing the truth into the row travels with the row, so both work.
    """
    packed: np.ndarray = label_codes(true_labels) * len(FIELD_TYPES) + label_codes(pred_labels)
    return packed


def macro_f1_statistic() -> Callable[[np.ndarray], float]:
    """The callable `paired_permutation` takes when the statistic is macro F1.

    This is the case `paired_permutation` accepts a callable FOR (E3). Macro F1
    is not the mean of anything: it is an unweighted average over classes of a
    ratio of counts, so a test that assumed it could be written as a mean of per
    row numbers would be answering a different question. What is permuted is
    which engine's prediction sits at each row, and the statistic is recomputed
    from scratch on every swap.

    Use it with `encode_outcome` for both conditions and `template_id` as
    `clusters`, which is exactly the comparison E5 describes.
    """
    n_classes = len(FIELD_TYPES)

    def statistic(encoded: np.ndarray) -> float:
        codes = np.asarray(encoded, dtype=np.int64)
        return macro_f1(codes // n_classes, codes % n_classes, n_classes)

    return statistic


def summarise(paths: EvaluationPaths) -> dict[str, Any]:
    """The evaluation report as data, for a caller that wants to print it."""
    return dict(json.loads(paths.report.read_text(encoding="utf-8")))
