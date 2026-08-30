"""Scoped question answering: prose is retrieved, numbers are computed.

Two properties are the whole point of the module and both are asserted here. A
question this cannot ground is REFUSED, with the answerable questions named,
rather than answered from whatever the retrieval happened to return. And every
number in an answer comes from a SQL template or from the analysis layer, so a
generated figure that is not in the computed table takes its sentence with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from llm_fakes import TAGS, FakeTransport, fake_chat_says, fake_embeddings

from triage.core.experiment import Experiment, MetricSeries
from triage.core.store import Store
from triage.llm.ask import (
    INTENTS,
    AnswerContext,
    Chunk,
    build_prompt,
    chunk_markdown,
    collect_prose,
    database_context,
    detect_intent,
    refusal_text,
)
from triage.llm.ask import ask as ask_question
from triage.llm.ollama_client import OllamaClient
from triage.llm.summarizer import ContextEntry


def transport(answer: str = "There are 2 runs in the database.") -> FakeTransport:
    return FakeTransport(
        {
            "/api/tags": TAGS,
            "/api/embed": fake_embeddings(256),
            "/api/chat": fake_chat_says(answer),
        }
    )


def seeded(path: Path, n_runs: int = 2) -> None:
    rng = np.random.default_rng(0)
    with Store(path) as store:
        for index in range(n_runs):
            steps = np.arange(50, dtype=np.int64)
            store.upsert(
                Experiment(
                    run_id=f"run_{index}",
                    source_path=f"/logs/run_{index}",
                    source_format="jsonl",
                    config={"variant": f"lr{index}", "learning_rate": 0.1 * (index + 1)},
                    metrics={
                        "val/accuracy": MetricSeries(
                            tag="val/accuracy",
                            steps=steps,
                            values=rng.normal(0.8, 0.01, 50).astype(np.float32),
                        )
                    },
                ),
                source_hash=f"hash_{index}",
            )


def prose_tree(root: Path) -> Path:
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text(
        "# ML Experiment Triage\n\n"
        "This tool ranks training runs and refuses comparisons it cannot make.\n\n"
        "## Verdicts\n\n"
        "A verdict clears a statistical gate and a practical gate.\n",
        encoding="utf-8",
    )
    (root / "docs" / "methodology.md").write_text(
        "# Methodology\n\n"
        "The permutation test is paired across seeds.\n\n"
        "## Sensitivity\n\n"
        "Spearman rank correlation relates a hyperparameter to a metric.\n",
        encoding="utf-8",
    )
    return root


# ------------------------------------------------------------------ chunking


def test_a_document_is_split_at_its_headings() -> None:
    chunks = chunk_markdown("README.md", "# One\n\nalpha\n\n## Two\n\nbeta\n")

    assert [chunk.heading for chunk in chunks] == ["One", "Two"]
    assert chunks[0].text == "alpha"
    assert all(chunk.source == "README.md" for chunk in chunks)


def test_a_long_section_is_split_at_paragraph_boundaries() -> None:
    body = "\n\n".join("word " * 60 for _ in range(6))
    chunks = chunk_markdown("d.md", f"# Long\n\n{body}", target=400)

    assert len(chunks) > 1
    assert all(chunk.text.strip() == chunk.text for chunk in chunks)
    assert all(chunk.heading == "Long" for chunk in chunks)


def test_a_document_with_no_headings_still_chunks() -> None:
    chunks = chunk_markdown("d.md", "just a paragraph")
    assert [chunk.heading for chunk in chunks] == [""]


def test_collecting_prose_is_stable_and_reads_only_prose(tmp_path: Path) -> None:
    root = prose_tree(tmp_path)
    (root / "docs" / "data.json").write_text('{"not": "prose"}', encoding="utf-8")

    first = collect_prose(root)
    second = collect_prose(root)

    assert [chunk.source for chunk in first] == [chunk.source for chunk in second]
    assert "docs/methodology.md" in {chunk.source for chunk in first}
    assert not any(chunk.source.endswith(".json") for chunk in first)


# -------------------------------------------------------------------- intent


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("which conditions regressed?", "verdicts"),
        ("what did it refuse to compare?", "refusals"),
        ("is the learning rate sensitive?", "sensitivity"),
        ("how many runs are in the database?", "database"),
    ],
)
def test_a_question_is_routed_to_the_intent_it_names(question: str, expected: str) -> None:
    intent = detect_intent(question)
    assert intent is not None
    assert intent.name == expected


def test_a_question_about_nothing_this_records_has_no_intent() -> None:
    assert detect_intent("what is the capital of Germany?") is None


def test_a_refusal_names_everything_that_could_be_answered() -> None:
    text = refusal_text("what is the weather", "it names nothing this records")

    for intent in INTENTS:
        assert intent.description in text
    assert "triage compare" in text


# ------------------------------------------------------------------- context


def test_the_database_context_comes_out_of_the_sql_templates(temp_database: Path) -> None:
    seeded(temp_database, n_runs=3)
    with Store(temp_database) as store:
        context = database_context(store)

    assert context.table().values["runs in the database"] == 3.0
    assert context.table().values["val/accuracy: runs logging it"] == 3.0
    assert "Runs in the database: 3" in context.render()


def test_the_prompt_carries_the_question_the_prose_and_the_figures() -> None:
    context = AnswerContext(
        lines=("Runs in the database: 2",),
        entries=(ContextEntry("runs in the database", 2.0),),
    )
    prompt = build_prompt(
        "how many runs?", [Chunk("README.md", "Verdicts", "A verdict clears two gates.")], context
    )

    assert "how many runs?" in prompt
    assert "[README.md (Verdicts)]" in prompt
    assert "Runs in the database: 2" in prompt
    assert "only numbers you may use" in prompt


# ------------------------------------------------------------------ the verb


def test_an_unanswerable_question_is_refused_without_asking_the_model(
    tmp_path: Path, temp_database: Path
) -> None:
    seeded(temp_database)
    fake = transport()
    with Store(temp_database) as store:
        result = ask_question(
            OllamaClient(transport=fake), store, "what is the capital of Germany?", tmp_path
        )

    assert result.refused is True
    assert fake.bodies("/api/chat") == []
    assert "What I can answer" in result.text


def test_a_verdict_question_without_a_baseline_is_refused_naming_the_flag(
    tmp_path: Path, temp_database: Path
) -> None:
    seeded(temp_database)
    fake = transport()
    with Store(temp_database) as store:
        result = ask_question(
            OllamaClient(transport=fake), store, "which conditions regressed?", tmp_path
        )

    assert result.refused is True
    assert "--baseline" in result.text
    assert fake.bodies("/api/chat") == []


def test_a_database_question_is_answered_from_prose_and_sql(
    tmp_path: Path, temp_database: Path
) -> None:
    root = prose_tree(tmp_path)
    seeded(temp_database, n_runs=2)
    fake = transport("The database holds 2 runs. Each one records a validation accuracy series.")
    with Store(temp_database) as store:
        result = ask_question(
            OllamaClient(transport=fake), store, "how many runs are in the database?", root
        )

    assert result.refused is False
    assert result.intent == "database"
    assert result.sources, "the answer must say which documents it read"
    assert "2 runs" in result.text
    assert result.n_dropped == 0


def test_a_fabricated_figure_takes_its_sentence_with_it(
    tmp_path: Path, temp_database: Path
) -> None:
    root = prose_tree(tmp_path)
    seeded(temp_database, n_runs=2)
    fake = transport(
        "The database holds 2 runs. It also holds 47 refused comparisons. Nothing else is recorded."
    )
    with Store(temp_database) as store:
        result = ask_question(
            OllamaClient(transport=fake), store, "how many runs are in the database?", root
        )

    assert result.n_dropped == 1
    assert "47" not in result.text
    assert "Nothing else is recorded." in result.text


def test_a_second_identical_question_asks_the_model_nothing(
    tmp_path: Path, temp_database: Path
) -> None:
    root = prose_tree(tmp_path)
    seeded(temp_database, n_runs=2)
    fake = transport("The database holds 2 runs.")
    with Store(temp_database) as store:
        first = ask_question(
            OllamaClient(transport=fake), store, "how many runs are recorded?", root
        )
        second = ask_question(
            OllamaClient(transport=fake), store, "how many runs are recorded?", root
        )

    assert len(fake.bodies("/api/chat")) == 1
    assert second.cached is True
    assert first.text == second.text


def test_the_result_describes_where_its_answer_came_from(
    tmp_path: Path, temp_database: Path
) -> None:
    root = prose_tree(tmp_path)
    seeded(temp_database)
    fake = transport("The database holds 2 runs.")
    with Store(temp_database) as store:
        result = ask_question(OllamaClient(transport=fake), store, "what is in the database?", root)

    described = result.describe()
    assert "intent database" in described
    assert "computed figure" in described


def test_no_prose_at_all_is_not_an_error(tmp_path: Path, temp_database: Path) -> None:
    """A database somebody points at from an empty directory still answers."""
    seeded(temp_database)
    fake = transport("The database holds 2 runs.")
    with Store(temp_database) as store:
        result = ask_question(
            OllamaClient(transport=fake), store, "how many runs?", tmp_path / "empty"
        )

    assert result.refused is False
    assert result.sources == ()
    assert result.text.startswith("The database holds 2 runs")
