"""Retrieval augmented few shot classification, against a fake chat model.

The fake copies the label of the nearest example it is shown, which is a
caricature of what the real thing is supposed to do and is exactly what makes it
useful here: with retrieval it can be right, without retrieval it has nothing to
copy, and the ablation therefore has a real and deterministic difference to
measure in CI with no model anywhere.

The assertion this file exists for is the cache one. A second pass over the same
fields must make zero HTTP calls, and the only way to know that is to count them
in the transport rather than to ask the annotator how it thinks it did.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from llm_fakes import (
    CHAT_DIGEST,
    TAGS,
    FakeTransport,
    fake_chat_copies_the_nearest_example,
    fake_chat_says,
    fake_embeddings,
)

from triage.autofill.generator import GeneratorConfig, generate_fields
from triage.autofill.taxonomy import FieldType
from triage.core.store import Store
from triage.llm.annotator import (
    ENGINE_KNN,
    ENGINE_ZERO_SHOT,
    FIRST_PASS_CONFIDENCE,
    RETRY_CONFIDENCE,
    UNKNOWN_CONFIDENCE,
    Annotator,
    AnnotatorConfig,
    parse_field_type,
    render_signals,
    response_cache_key,
    run_ablation,
)
from triage.llm.ollama_client import DEFAULT_CHAT_MODEL, OllamaClient


def corpus(n_fields: int = 60, seed: int = 0) -> list:
    return generate_fields(GeneratorConfig(n_fields=n_fields, locales=("de_DE",)), seed=seed)


def transport(chat: object | None = None) -> FakeTransport:
    return FakeTransport(
        {
            "/api/tags": TAGS,
            "/api/embed": fake_embeddings(256),
            "/api/chat": chat or fake_chat_copies_the_nearest_example(),
        }
    )


def annotator(store: Store, fake: FakeTransport, examples: list, **overrides: object) -> Annotator:
    settings = {"k": 4, "retrieval": True}
    settings.update(overrides)
    return Annotator(
        OllamaClient(transport=fake),
        store,
        AnnotatorConfig(**settings),  # type: ignore[arg-type]
        examples=examples if settings["retrieval"] else (),
    )


# ------------------------------------------------------------------ rendering


def test_one_rendering_serves_both_retrieval_and_the_prompt() -> None:
    record = corpus(20)[0]
    text = render_signals(record)

    assert render_signals(record) == text, "the rendering must be a pure function"
    assert f"locale={record.locale}" in text
    assert f'label="{record.label}"' in text
    assert f'next="{record.next_label}"' in text


def test_an_empty_signal_still_occupies_its_place() -> None:
    """Two fields differing in one attribute must differ in one place."""
    records = corpus(40)
    rendered = [render_signals(record) for record in records]
    assert all(text.count("placeholder=") == 1 for text in rendered)


# --------------------------------------------------------------------- keys


def test_the_cache_key_changes_with_the_model_the_digest_and_the_prompt() -> None:
    base = response_cache_key("m", "d", "p")

    assert response_cache_key("other", "d", "p") != base
    assert response_cache_key("m", "other", "p") != base
    assert response_cache_key("m", "d", "other") != base
    assert response_cache_key("m", "d", "p") == base


def test_the_key_cannot_be_confused_by_a_concatenation() -> None:
    """`m + d` and `md + ''` must not hash to the same key."""
    assert response_cache_key("ab", "c", "p") != response_cache_key("a", "bc", "p")


# ------------------------------------------------------------------ parsing


def test_a_plain_object_parses() -> None:
    assert parse_field_type('{"field_type": "given-name"}') is FieldType.GIVEN_NAME


def test_a_fenced_object_parses_because_instruct_models_fence_things() -> None:
    fenced = '```json\n{"field_type": "postal-code"}\n```'
    assert parse_field_type(fenced) is FieldType.POSTAL_CODE


@pytest.mark.parametrize(
    "answer",
    [
        "This field is clearly a postal code.",
        '{"label": "postal-code"}',
        '{"field_type": "postleitzahl"}',
        '{"field_type": 3}',
        '["postal-code"]',
        "",
        '{"field_type": "postal-code"',
    ],
)
def test_anything_that_is_not_the_asked_for_object_is_refused(answer: str) -> None:
    assert parse_field_type(answer) is None


# ---------------------------------------------------------------- retrieval


def test_the_zero_shot_arm_needs_no_examples_and_the_knn_arm_refuses_without_them(
    temp_database: Path,
) -> None:
    fake = transport()
    with Store(temp_database) as store:
        Annotator(OllamaClient(transport=fake), store, AnnotatorConfig(retrieval=False))
        with pytest.raises(ValueError, match="labelled examples"):
            Annotator(OllamaClient(transport=fake), store, AnnotatorConfig(retrieval=True))


def test_the_nearest_example_is_shown_last(temp_database: Path) -> None:
    records = corpus(40)
    fake = transport()
    with Store(temp_database) as store:
        instance = annotator(store, fake, records[10:])
        target = records[0]
        found = instance.neighbours(target)
        index = instance.index()
        query = instance.embedder.embed_query(render_signals(target))
        best = index.search(query, k=4)[0]

    assert len(found) == 4
    assert render_signals(found[-1]) == best.text


def test_a_knn_prompt_carries_the_examples_and_a_zero_shot_one_does_not(
    temp_database: Path,
) -> None:
    records = corpus(40)
    fake = transport()
    with Store(temp_database) as store:
        knn = annotator(store, fake, records[10:])
        prompt = knn.build_prompt(records[0], knn.neighbours(records[0]))
        bare = annotator(store, fake, [], retrieval=False).build_prompt(records[0], [])

    assert prompt.count('"field_type"') == 5, "four examples plus the format line"
    assert render_signals(records[0]) in prompt
    assert bare.count('"field_type"') == 1
    assert "Labelled examples" not in bare


def test_the_corpus_is_embedded_once_for_the_whole_pass(temp_database: Path) -> None:
    records = corpus(60)
    fake = transport()
    with Store(temp_database) as store:
        annotator(store, fake, records[20:]).annotate(records[:5])

    corpus_calls = [
        body
        for body in fake.bodies("/api/embed")
        if len(body["input"]) > 1  # type: ignore[arg-type]
    ]
    assert len(corpus_calls) == 1


# ---------------------------------------------------------------- answering


def test_the_retrieval_arm_copies_the_nearest_label_and_the_zero_shot_arm_cannot(
    temp_database: Path,
) -> None:
    records = corpus(60)
    fake = transport()
    with Store(temp_database) as store:
        knn = annotator(store, fake, records[20:]).annotate(records[:8])
        bare = annotator(store, fake, [], retrieval=False).annotate(records[:8])

    assert all(item.confidence == FIRST_PASS_CONFIDENCE for item in knn.annotations)
    assert {item.field_type for item in bare.annotations} == {FieldType.UNKNOWN}
    assert knn.requests == 8
    assert knn.unknown == 0


def test_a_second_invocation_makes_zero_http_calls(temp_database: Path) -> None:
    """6.3's reproducibility claim, asserted where it can be seen."""
    records = corpus(60)
    fake = transport()
    with Store(temp_database) as store:
        first = annotator(store, fake, records[20:]).annotate(records[:6])
    after = len(fake.bodies("/api/chat"))

    with Store(temp_database) as store:
        second = annotator(store, fake, records[20:]).annotate(records[:6])

    assert after == 6
    assert len(fake.bodies("/api/chat")) == after, "a warm cache must not ask the model again"
    assert second.requests == 0
    assert second.cache_hits == 6
    assert [item.field_type for item in second.annotations] == [
        item.field_type for item in first.annotations
    ]


def test_the_response_cache_is_keyed_by_the_prompt_the_model_saw(
    temp_database: Path,
) -> None:
    records = corpus(60)
    fake = transport()
    with Store(temp_database) as store:
        instance = annotator(store, fake, records[20:])
        prompt = instance.build_prompt(records[0], instance.neighbours(records[0]))
        instance.annotate(records[:1])
        key = response_cache_key(DEFAULT_CHAT_MODEL, CHAT_DIGEST, prompt)
        assert store.get_response(key) is not None


def test_a_malformed_answer_is_retried_once_and_the_retry_is_counted(
    temp_database: Path,
) -> None:
    records = corpus(40)
    fake = transport(fake_chat_says("I think this is an email.", '{"field_type": "email"}'))
    with Store(temp_database) as store:
        run = annotator(store, fake, records[20:]).annotate(records[:1])

    assert run.retried == 1
    assert run.unknown == 0
    assert run.annotations[0].field_type is FieldType.EMAIL
    assert run.annotations[0].confidence == RETRY_CONFIDENCE
    assert run.annotations[0].attempts == 2


def test_the_retry_prompt_says_what_was_wrong_with_the_first_answer(
    temp_database: Path,
) -> None:
    records = corpus(40)
    fake = transport(fake_chat_says("prose", '{"field_type": "email"}'))
    with Store(temp_database) as store:
        annotator(store, fake, records[20:]).annotate(records[:1])

    second = fake.bodies("/api/chat")[1]
    assert "not a single JSON object" in second["messages"][-1]["content"]


def test_a_second_failure_becomes_unknown_and_is_counted(temp_database: Path) -> None:
    records = corpus(40)
    fake = transport(fake_chat_says("still prose"))
    with Store(temp_database) as store:
        run = annotator(store, fake, records[20:]).annotate(records[:3])

    assert run.unknown == 3
    assert run.retried == 3
    assert {item.field_type for item in run.annotations} == {FieldType.UNKNOWN}
    assert {item.confidence for item in run.annotations} == {UNKNOWN_CONFIDENCE}


def test_a_token_outside_the_taxonomy_is_a_failure_rather_than_a_new_class(
    temp_database: Path,
) -> None:
    records = corpus(40)
    fake = transport(fake_chat_says('{"field_type": "phone-number"}'))
    with Store(temp_database) as store:
        run = annotator(store, fake, records[20:]).annotate(records[:1])

    assert run.annotations[0].field_type is FieldType.UNKNOWN
    assert run.unknown == 1


def test_limit_shortens_the_pass(temp_database: Path) -> None:
    records = corpus(60)
    fake = transport()
    with Store(temp_database) as store:
        run = annotator(store, fake, records[20:], limit=3).annotate(records[:10])

    assert len(run.annotations) == 3
    assert len(fake.bodies("/api/chat")) == 3


def test_the_run_reports_itself_in_the_shape_every_other_engine_uses(
    temp_database: Path,
) -> None:
    records = corpus(60)
    fake = transport()
    with Store(temp_database) as store:
        run = annotator(store, fake, records[20:], engine="llm").annotate(records[:5])

    scored = run.scored()
    assert scored.engine == "llm"
    assert scored.predicted.shape == (5,)
    assert scored.confidence.shape == (5,)
    assert scored.latency_us > 0.0
    assert "kNN k=4" in run.describe()


# ----------------------------------------------------------------- ablation


def test_the_ablation_writes_both_artifact_shapes_and_returns_a_verdict(
    tmp_path: Path, temp_database: Path
) -> None:
    records = corpus(400)
    train = [record for record in records if record.split == "train"]
    evaluate = [record for record in records if record.split == "val"]
    fake = transport()
    out = tmp_path / "ablation"

    with Store(temp_database) as store:
        result = run_ablation(
            OllamaClient(transport=fake),
            store,
            records=evaluate,
            examples=train,
            out_dir=out,
            split="val",
            locale="de_DE",
            bootstrap=3,
            config=AnnotatorConfig(k=4),
        )

    # (a) the triage native shape, one run directory per engine per replicate.
    names = sorted(path.name for path in (out / "runs").iterdir())
    assert f"{ENGINE_KNN}_de_DE_val_seed0" in names
    assert f"{ENGINE_ZERO_SHOT}_de_DE_val_seed0" in names
    # (b) the consumer's E5 outcomes shape, with a row per field per engine.
    rows = [
        json.loads(line)
        for line in (out / "outcomes" / "autofill_eval" / "run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["engine"] for row in rows} == {"rules", ENGINE_KNN, ENGINE_ZERO_SHOT}

    assert result.macro_f1[ENGINE_KNN] > result.macro_f1[ENGINE_ZERO_SHOT]
    assert result.baseline == f"{ENGINE_ZERO_SHOT}_de_DE_val"
    verdicts = [row for row in result.findings if row.candidate == f"{ENGINE_KNN}_de_DE_val"]
    assert verdicts, "the comparison layer must have judged the retrieval arm"
    assert all(0.0 < row.p_value <= 1.0 for row in verdicts)
    assert result.verdict is not None
    assert result.lift > 0.0
    assert "p " in result.headline()


def test_the_ablation_leaves_the_callers_database_free_of_its_runs(
    tmp_path: Path, temp_database: Path
) -> None:
    """It measures itself in its own database, not in somebody's sweep."""
    records = corpus(400)
    fake = transport()
    with Store(temp_database) as store:
        run_ablation(
            OllamaClient(transport=fake),
            store,
            records=[r for r in records if r.split == "val"],
            examples=[r for r in records if r.split == "train"],
            out_dir=tmp_path / "ablation",
            bootstrap=3,
            config=AnnotatorConfig(k=4),
        )
        assert store.run_ids() == []
        assert store.cache_counts()["responses"] > 0

    assert (tmp_path / "ablation" / "ablation.db").is_file()


def test_the_evaluation_report_scores_all_three_engines(
    tmp_path: Path, temp_database: Path
) -> None:
    records = corpus(400)
    fake = transport()
    with Store(temp_database) as store:
        result = run_ablation(
            OllamaClient(transport=fake),
            store,
            records=[r for r in records if r.split == "val"],
            examples=[r for r in records if r.split == "train"],
            out_dir=tmp_path / "ablation",
            bootstrap=3,
            config=AnnotatorConfig(k=4),
        )

    assert sorted(result.macro_f1) == sorted([ENGINE_KNN, ENGINE_ZERO_SHOT, "rules"])
    assert np.isfinite(list(result.macro_f1.values())).all()
