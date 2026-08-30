"""The one test in this repository that talks to a real model.

Marked `ollama` and skipped automatically when `health()` fails, which is what
makes it safe to leave in the suite: on CI, and on any machine without a local
Ollama, it disappears. `pytest -m ollama` runs it deliberately, and Section 6.6
asks for it to be run once before a release with the result recorded in
`docs/ENGINEERING_LOG.md`.

What it is FOR is the half of this layer that a fake transport cannot check: the
prompts have to be ones a real instruction tuned model answers in the format the
parser demands, the embedder has to return the width the model record claims,
and the two models have to fit on the card at the same time. Every one of those
is a fact about the machine rather than about this code, and a canned response
would assert it into existence rather than measure it.

Nothing here leaves localhost, and there is no network fallback anywhere in this
package: a machine with no Ollama skips these tests rather than reaching for a
hosted model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from triage.autofill.generator import GeneratorConfig, generate_fields
from triage.core.store import Store
from triage.llm.annotator import Annotator, AnnotatorConfig, parse_field_type, render_signals
from triage.llm.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, config_for
from triage.llm.ollama_client import LlmError, OllamaClient

pytestmark = pytest.mark.ollama


@pytest.fixture
def client() -> OllamaClient:
    """A client against the real service, or a skip when there is not one."""
    instance = OllamaClient()
    try:
        instance.health()
    except LlmError as error:
        pytest.skip(f"no local Ollama: {error}")
    return instance


def test_the_whole_layer_works_against_the_real_service(
    client: OllamaClient, tmp_path: Path
) -> None:
    """Health, digests, embeddings and one classified field, end to end."""
    report = client.health()
    chat_model = client.resolve_chat_model()
    assert chat_model in report.names

    for name in (chat_model, DEFAULT_EMBEDDING_MODEL):
        digest = client.digest_of(name)
        assert len(digest) >= 32, "a digest short enough to collide is not a cache key"

    records = generate_fields(GeneratorConfig(n_fields=80, locales=("de_DE",)), seed=0)
    database = tmp_path / "triage.db"

    with Store(database) as store:
        embedder = Embedder(client, store, model=DEFAULT_EMBEDDING_MODEL)
        matrix = embedder.embed_documents([render_signals(record) for record in records[:16]])
        assert matrix.shape == (16, config_for(DEFAULT_EMBEDDING_MODEL).dimensions)
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
        # Two different fields must not embed to the same vector, which is the
        # cheapest check that the prefixes and the truncation left real signal.
        assert not np.allclose(matrix[0], matrix[1])

        annotator = Annotator(
            client,
            store,
            AnnotatorConfig(k=4, chat_model=chat_model),
            examples=records[20:],
        )
        run = annotator.annotate(records[:4])

    assert len(run.annotations) == 4
    assert run.requests == 4
    assert run.unknown <= 1, "a real model should parse at least three of four first time"
    print(f"\nreal service: {run.describe()}")

    # And the second pass over the same fields must ask it nothing at all.
    with Store(database) as store:
        again = Annotator(
            client, store, AnnotatorConfig(k=4, chat_model=chat_model), examples=records[20:]
        ).annotate(records[:4])
    assert again.requests == 0
    assert again.cache_hits == 4
    assert [item.field_type for item in again.annotations] == [
        item.field_type for item in run.annotations
    ]


def test_the_model_answers_in_the_format_the_parser_demands(client: OllamaClient) -> None:
    """The prompt is only as good as what a real model does with it."""
    records = generate_fields(GeneratorConfig(n_fields=40, locales=("de_DE",)), seed=1)
    chat_model = client.resolve_chat_model()
    prompt = (
        f"Field to label:\n{render_signals(records[0])}\n\n"
        'Answer with one JSON object: {"field_type": "<token>"}'
    )
    from triage.llm.annotator import SYSTEM_PROMPT

    answer = client.chat(prompt, system=SYSTEM_PROMPT, model=chat_model)
    assert parse_field_type(answer) is not None, f"the model answered {answer!r}"


def test_the_chat_model_and_the_embedder_are_resident_together(client: OllamaClient) -> None:
    """Section 2's one unverified assumption, checked rather than assumed.

    The arithmetic said 7.5 GB of weights plus 0.6 GB plus caches fits inside
    12,227 MiB, and the fallback if it did not was sequential loading, which the
    batch design in Section 6 already permits. This asks the server which models
    it currently holds after both have been used, and prints the answer so a
    reader of the log sees the measurement rather than the conclusion.

    It does not FAIL on eviction. Whether a scheduler keeps two models resident
    is a property of the machine and the moment, not of this code, and a test
    that turned a scheduling decision into a red build would be measuring the
    wrong thing. The measurement is recorded in the engineering log.
    """
    chat_model = client.resolve_chat_model()
    client.embed(["title: none | text: Vorname"], model=DEFAULT_EMBEDDING_MODEL)
    client.chat("Answer with the single word: ready.", model=chat_model)

    # Through the private helper on purpose. `/api/ps` is a fifth endpoint, and
    # the client is capped at the four this layer actually uses (E7d: it stays
    # minimal). Widening its public surface so that one test could ask a
    # question about the machine would be the tail wagging the dog.
    loaded = client._round_trip("/api/ps", None)
    resident = {
        str(entry.get("name", "")): float(entry.get("size_vram", 0) or 0) / 1e9
        for entry in loaded.get("models", [])
    }
    print(f"\nresident after using both: {resident}")
    assert resident, "the server reports nothing loaded straight after two requests"
