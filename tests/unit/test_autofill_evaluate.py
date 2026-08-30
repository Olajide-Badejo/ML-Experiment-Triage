"""Evaluation: the heuristic baseline, the metrics, and the two artifact shapes.

Three of the four acceptance criteria live here. Criterion 2, that the trained
model beats the heuristic on de_DE validation macro F1 by a margin that is real
and not trivial, is asserted against the generator's DEFAULT knobs, so tuning
the corpus until the margin disappears, or until it becomes a walkover, fails
the suite. Criterion 4, that the outcomes file round trips through
`OutcomesParser` and `paired_permutation`, is asserted with this repository's own
parser and its own test, on the file the evaluation actually writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage.analysis.comparison import paired_permutation
from triage.autofill.calibration import CalibrationSummary
from triage.autofill.evaluate import (
    ENGINE_NAMES,
    OUTCOMES_SCHEMA_VERSION,
    EvaluationReport,
    HeuristicClassifier,
    encode_outcome,
    evaluate_engines,
    heuristic_predict,
    label_codes,
    macro_f1_statistic,
    outcomes_rows,
    write_evaluation,
)
from triage.autofill.features import featurise
from triage.autofill.generator import (
    GeneratorConfig,
    generate_fields,
    write_corpus,
)
from triage.autofill.model import TrainConfig, macro_f1, train
from triage.autofill.taxonomy import FIELD_TYPES, FieldType, index_of
from triage.parsers.outcomes_parser import OutcomesParser

#: The corpus the acceptance numbers are measured on. Small enough to train
#: inside a unit test, large enough that a de_DE macro F1 means something.
CORPUS = GeneratorConfig(n_fields=4000)


@pytest.fixture(scope="module")
def fitted() -> dict:
    records = generate_fields(CORPUS, seed=0)
    train_rows = [r for r in records if r.split == "train"]
    val_rows = [r for r in records if r.split == "val"]
    result = train(featurise(train_rows), featurise(val_rows), TrainConfig(epochs=8, seed=0))
    return {"records": records, "val": val_rows, "model": result.model}


# ------------------------------------------------------------------ heuristic


def test_the_heuristic_trusts_a_present_autocomplete_token() -> None:
    from dataclasses import replace

    record = generate_fields(GeneratorConfig(n_fields=40), seed=0)[0]
    annotated = replace(record, autocomplete="postal-code", label="", name="input_3")
    predicted, confidence, _ = heuristic_predict([annotated])
    assert FIELD_TYPES[int(predicted[0])] is FieldType.POSTAL_CODE
    assert confidence[0] == pytest.approx(HeuristicClassifier.AUTOCOMPLETE_CONFIDENCE)


def test_the_heuristic_falls_through_when_the_attribute_says_off() -> None:
    from dataclasses import replace

    record = generate_fields(GeneratorConfig(n_fields=40), seed=0)[0]
    annotated = replace(record, autocomplete="off", label="Postleitzahl", placeholder="")
    predicted, _, _ = heuristic_predict([annotated])
    assert FIELD_TYPES[int(predicted[0])] is FieldType.POSTAL_CODE


def test_the_heuristic_knows_the_canonical_german_terms() -> None:
    """The baseline is a fair one: every term the specification names is in it."""

    record = generate_fields(GeneratorConfig(n_fields=40), seed=0)[0]
    cases = {
        "Vorname": FieldType.GIVEN_NAME,
        "Nachname": FieldType.FAMILY_NAME,
        "E-Mail-Adresse": FieldType.EMAIL,
        "Telefonnummer": FieldType.TEL,
        "Strasse und Hausnummer": FieldType.ADDRESS_LINE1,
        "Adresszusatz": FieldType.ADDRESS_LINE2,
        "Ort": FieldType.ADDRESS_LEVEL2,
        "Bundesland": FieldType.ADDRESS_LEVEL1,
        "Postleitzahl": FieldType.POSTAL_CODE,
        "PLZ": FieldType.POSTAL_CODE,
        "Land": FieldType.COUNTRY_NAME,
        "Firma": FieldType.ORGANIZATION,
        "Karteninhaber": FieldType.CC_NAME,
        "Kartennummer": FieldType.CC_NUMBER,
        "Ablaufmonat": FieldType.CC_EXP_MONTH,
        "Ablaufjahr": FieldType.CC_EXP_YEAR,
        "Pruefziffer": FieldType.CC_CSC,
    }
    blank = [
        record.__class__(
            **{
                **record.__dict__,
                "label": label,
                "autocomplete": "",
                "name": "x1",
                "element_id": "x1",
                "placeholder": "",
            }
        )
        for label in cases
    ]
    predicted, _, _ = heuristic_predict(blank)
    got = {label: FIELD_TYPES[int(code)] for label, code in zip(cases, predicted, strict=True)}
    assert got == cases


def test_the_heuristic_says_unknown_rather_than_guessing() -> None:
    record = generate_fields(GeneratorConfig(n_fields=40), seed=0)[0]
    blank = record.__class__(
        **{
            **record.__dict__,
            "label": "",
            "name": "input_1",
            "element_id": "input_1",
            "placeholder": "",
            "autocomplete": "",
            "input_type": "text",
            "section": "",
            "previous_label": "",
            "next_label": "",
        }
    )
    predicted, confidence, _ = heuristic_predict([blank])
    assert FIELD_TYPES[int(predicted[0])] is FieldType.UNKNOWN
    assert confidence[0] == pytest.approx(HeuristicClassifier.FALLBACK_CONFIDENCE)


def test_the_heuristic_is_a_serious_baseline_on_english() -> None:
    """If the baseline were weak everywhere, beating it would prove nothing."""
    records = [r for r in generate_fields(CORPUS, seed=0) if r.locale == "en_US"]
    predicted, _, _ = heuristic_predict(records)
    truth = np.array([index_of(r.field_type) for r in records])
    assert macro_f1(truth, predicted) > 0.60


# ------------------------------------------------------------- the two engines


def test_both_engines_score_every_row(fitted: dict) -> None:
    scored = evaluate_engines(fitted["val"], fitted["model"])
    assert set(scored) == {"ngram", "rules"}
    for engine in scored.values():
        assert engine.predicted.size == len(fitted["val"])
        assert engine.confidence.size == len(fitted["val"])
        assert float(engine.confidence.min()) >= 0.0
        assert float(engine.confidence.max()) <= 1.0
        assert engine.latency_us > 0


def test_the_engine_names_are_the_consumers_own_vocabulary() -> None:
    """The fixture in tests/fixtures/outcomes uses `rules` and `ngram`."""
    assert ENGINE_NAMES == {"model": "ngram", "heuristic": "rules", "llm": "llm"}


def test_the_model_beats_the_heuristic_on_german_macro_f1(fitted: dict) -> None:
    """Acceptance criterion 2, at the generator's default noise settings.

    The margin has to be REAL and NOT TRIVIAL, so both ends are asserted: a
    corpus tuned until the heuristic collapses would prove nothing about the
    model, and one tuned until the two agree would prove nothing at all.
    """
    german = [r for r in fitted["val"] if r.locale == "de_DE"]
    scored = evaluate_engines(german, fitted["model"])
    truth = np.array([index_of(r.field_type) for r in german])
    model = macro_f1(truth, scored["ngram"].predicted)
    heuristic = macro_f1(truth, scored["rules"].predicted)
    assert model > heuristic + 0.05, (model, heuristic)
    assert model < heuristic + 0.45, (model, heuristic)
    assert heuristic > 0.35, heuristic


def test_the_report_carries_every_metric_the_specification_lists(fitted: dict) -> None:
    scored = evaluate_engines(fitted["val"], fitted["model"])
    report = EvaluationReport.build(
        engine="ngram",
        split="val",
        locale="all",
        records=fitted["val"],
        scored=scored["ngram"],
    )
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0
    assert set(report.per_type) <= {member.value for member in FIELD_TYPES}
    for row in report.per_type.values():
        assert {"precision", "recall", "support"} <= set(row)
    assert np.asarray(report.confusion).shape == (len(FIELD_TYPES), len(FIELD_TYPES))
    assert 0.0 <= report.ece <= 1.0


# ------------------------------------------------------- the outcomes artifact


def test_the_outcomes_rows_are_the_consumer_schema_verbatim(fitted: dict) -> None:
    scored = evaluate_engines(fitted["val"], fitted["model"])
    rows = outcomes_rows(fitted["val"], scored, run_id="autofill-eval-0", split="val")
    assert rows
    assert set(rows[0]) == {
        "schema_version",
        "run_id",
        "engine",
        "split",
        "form_id",
        "template_id",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "latency_us",
    }
    assert rows[0]["schema_version"] == OUTCOMES_SCHEMA_VERSION
    assert {row["engine"] for row in rows} == {"ngram", "rules"}
    assert all(row["true_label"] in {m.value for m in FIELD_TYPES} for row in rows)


def test_the_outcomes_file_round_trips_through_the_parser_and_the_test(
    fitted: dict, tmp_path: Path
) -> None:
    """Acceptance criterion 4, end to end on the file evaluate actually writes."""
    paths = write_evaluation(
        tmp_path,
        records=fitted["val"],
        model=fitted["model"],
        split="val",
        locale="all",
        bootstrap=3,
        seed=0,
    )
    outcomes = OutcomesParser().parse(paths.outcomes.parent)
    assert outcomes.n_rows == 2 * len(fitted["val"])

    rules, ngram = outcomes.pair_on("form_id", "engine", "rules", "ngram")
    result = paired_permutation(
        rules.field("correct"),
        ngram.field("correct"),
        statistic=lambda values: float(values.mean()),
        clusters=list(ngram.group("template_id")),
        higher_is_better=True,
    )
    assert result.mode == "paired_cluster"
    assert 0.0 <= result.p_value <= 1.0
    assert result.effect > 0


def test_the_outcomes_file_supports_the_macro_f1_statistic_it_was_built_for(
    fitted: dict, tmp_path: Path
) -> None:
    """E5's stated use: statistic macro F1, clusters template_id.

    This is the case `paired_permutation` takes a CALLABLE for. Macro F1 cannot
    be written as a mean of per row numbers, so what is permuted is which
    engine's prediction sits at each row and the statistic is recomputed from
    scratch on every swap. Each row carries its own truth, which is what keeps
    the statistic a pure function of its argument under the clustered bootstrap
    as well as under the label swap.
    """
    paths = write_evaluation(
        tmp_path,
        records=fitted["val"],
        model=fitted["model"],
        split="val",
        locale="de_DE",
        bootstrap=2,
        seed=0,
    )
    outcomes = OutcomesParser().parse(paths.outcomes.parent)
    rules, ngram = outcomes.pair_on("form_id", "engine", "rules", "ngram")
    np.testing.assert_array_equal(
        label_codes(ngram.group("true_label")), label_codes(rules.group("true_label"))
    )

    result = paired_permutation(
        encode_outcome(rules.group("true_label"), rules.group("pred_label")),
        encode_outcome(ngram.group("true_label"), ngram.group("pred_label")),
        statistic=macro_f1_statistic(),
        clusters=list(ngram.group("template_id")),
        higher_is_better=True,
    )
    assert result.mode == "paired_cluster"
    assert result.candidate_statistic > result.baseline_statistic + 0.05
    assert result.p_value <= 0.05, result.p_value


def test_the_packed_outcome_survives_a_resample_which_a_closure_would_not() -> None:
    """The reason `encode_outcome` exists, asserted rather than described.

    `paired_permutation` hands its statistic a CLUSTER BOOTSTRAP resample as
    well as a label swap, and a resample is a different length in a different
    order. A packed row still knows its own truth, so the statistic means the
    same thing on it.
    """
    truth = ["email", "postal-code", "email", "tel", "postal-code", "tel"]
    predicted = ["email", "postal-code", "tel", "tel", "postal-code", "tel"]
    packed = encode_outcome(truth, predicted)
    statistic = macro_f1_statistic()

    whole = statistic(packed)
    doubled = statistic(np.concatenate([packed, packed]))
    assert doubled == pytest.approx(whole)
    subset = statistic(packed[[0, 1, 3, 4]])
    assert subset == pytest.approx(1.0)


def test_the_outcomes_directory_is_claimed_by_the_parser_without_a_flag(
    fitted: dict, tmp_path: Path
) -> None:
    """Strict recognition: the rows declare a `schema_version`, so they qualify."""
    paths = write_evaluation(
        tmp_path,
        records=fitted["val"],
        model=fitted["model"],
        split="val",
        locale="all",
        bootstrap=2,
        seed=0,
    )
    assert OutcomesParser(strict=True).can_parse(paths.outcomes.parent)


# ----------------------------------------------------- the triage native shape


def test_the_bootstrap_writes_one_seed_replicated_run_per_engine(
    fitted: dict, tmp_path: Path
) -> None:
    paths = write_evaluation(
        tmp_path,
        records=fitted["val"],
        model=fitted["model"],
        split="val",
        locale="de_DE",
        bootstrap=5,
        seed=0,
    )
    names = sorted(p.name for p in paths.runs)
    assert len(names) == 10
    assert sum(name.startswith("ngram_") for name in names) == 5
    assert sum(name.startswith("rules_") for name in names) == 5


def test_the_bootstrap_runs_are_ingestible_and_long_enough_to_compare(
    fitted: dict, tmp_path: Path
) -> None:
    from triage.parsers.json_parser import JsonlParser

    paths = write_evaluation(
        tmp_path,
        records=fitted["val"],
        model=fitted["model"],
        split="val",
        locale="de_DE",
        bootstrap=3,
        seed=0,
    )
    parser = JsonlParser()
    for run in paths.runs:
        experiment = parser.parse(run, root=paths.root)
        assert "eval/macro_f1" in experiment.tags
        assert len(experiment.series("eval/macro_f1")) >= 20
        assert experiment.config["engine"] in {"ngram", "rules"}


def test_the_calibration_file_is_written_for_the_report_to_read(
    fitted: dict, tmp_path: Path
) -> None:
    paths = write_evaluation(
        tmp_path,
        records=fitted["val"],
        model=fitted["model"],
        split="val",
        locale="all",
        bootstrap=2,
        seed=0,
    )
    summary = CalibrationSummary.from_dict(
        json.loads(paths.calibration.read_text(encoding="utf-8"))["calibration"]
    )
    assert summary.ece_post < summary.ece_pre
    logged = json.loads(paths.calibration.read_text(encoding="utf-8"))["metrics"]
    assert set(logged) == {"val/ece_pre", "val/ece_post", "val/temperature"}


def test_the_same_seed_gives_the_same_bootstrap_curves(fitted: dict, tmp_path: Path) -> None:
    """Everything but the measured latency is a pure function of the seed."""
    written = []
    for name in ("a", "b"):
        paths = write_evaluation(
            tmp_path / name,
            records=fitted["val"],
            model=fitted["model"],
            split="val",
            locale="de_DE",
            bootstrap=2,
            seed=4,
        )
        written.append(sorted(paths.runs))
    for left, right in zip(written[0], written[1], strict=True):
        assert (left / "metrics.jsonl").read_bytes() == (right / "metrics.jsonl").read_bytes()


def test_a_locale_with_no_rows_is_refused(fitted: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        write_evaluation(
            tmp_path,
            records=fitted["val"],
            model=fitted["model"],
            split="val",
            locale="fr_FR",
            bootstrap=2,
            seed=0,
        )


def test_evaluating_without_a_model_still_scores_the_heuristic(
    fitted: dict, tmp_path: Path
) -> None:
    """`--policy heuristic` needs no weights file, and says so by working."""
    paths = write_evaluation(
        tmp_path,
        records=fitted["val"],
        model=None,
        split="val",
        locale="all",
        bootstrap=2,
        seed=0,
    )
    assert all(p.name.startswith("rules_") for p in paths.runs)
    assert paths.calibration is None or not paths.calibration.exists()


def test_a_corpus_written_to_disk_evaluates_from_disk(tmp_path: Path) -> None:
    """The path the CLI takes: generate, load a split, evaluate it."""
    from triage.autofill.generator import load_split

    write_corpus(tmp_path / "data", GeneratorConfig(n_fields=800), seed=0)
    rows = load_split(tmp_path / "data", "val")
    predicted, confidence, latency = heuristic_predict(rows)
    assert predicted.size == len(rows) == confidence.size
    assert latency > 0
