"""The embedding cache and the exact brute force search over it.

Three claims are tested here and they are the three that decide whether
retrieval works at all: the asymmetric prompt prefixes are applied from the
model config record and never from a call site; a cached vector is served
without a second HTTP call; and the search returns the true nearest neighbours
rather than an approximation, which is the whole reason there is no vector
database in this project.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from llm_fakes import TAGS, FakeTransport, fake_embeddings

from triage.core.store import Store
from triage.llm.embeddings import (
    BGE_M3,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    Embedder,
    EmbeddingIndex,
    config_for,
    l2_normalise,
    text_key,
)
from triage.llm.ollama_client import OllamaClient


def embedding_transport(dimensions: int = 256) -> FakeTransport:
    """The shared fake, wired to answer the two endpoints an embedder uses."""
    return FakeTransport({"/api/tags": TAGS, "/api/embed": fake_embeddings(dimensions)})


def embedder(store: Store, transport: FakeTransport) -> Embedder:
    return Embedder(client=OllamaClient(transport=transport), store=store)


def test_the_default_model_carries_the_asymmetric_prefixes_of_section_two() -> None:
    config = config_for(DEFAULT_EMBEDDING_MODEL)

    assert config.query_prefix == "task: search result | query: "
    assert config.document_prefix == "title: none | text: "
    # Each prefix includes its trailing space, which is part of the string the
    # model was trained with and the easiest thing in this file to lose.
    assert config.query_prefix.endswith(" ")
    assert config.document_prefix.endswith(" ")


def test_the_opt_in_quality_tier_uses_no_prefixes_at_all() -> None:
    config = config_for(BGE_M3)

    assert config.query_prefix == ""
    assert config.document_prefix == ""


def test_a_model_outside_the_registry_names_the_two_that_are_in_it() -> None:
    with pytest.raises(ValueError) as error:
        config_for("nomic-embed-text")

    message = str(error.value)
    assert DEFAULT_EMBEDDING_MODEL in message
    assert BGE_M3 in message


def test_every_registered_model_is_spelled_the_way_ollama_spells_it() -> None:
    for name, config in EMBEDDING_MODELS.items():
        assert config.name == name
        assert config.dimensions > 0


def test_the_prefixes_are_applied_by_the_embedder_not_the_caller(temp_database: Path) -> None:
    transport = embedding_transport()
    with Store(temp_database) as store:
        instance = embedder(store, transport)
        instance.embed_documents(["Vorname"])
        instance.embed_queries(["Vorname"])

    sent = [call[1]["input"] for call in transport.calls if call[0].endswith("/api/embed")]  # type: ignore[index]
    assert sent[0] == ["title: none | text: Vorname"]
    assert sent[1] == ["task: search result | query: Vorname"]


def test_a_document_and_a_query_of_the_same_text_are_cached_apart(
    temp_database: Path,
) -> None:
    """They are different strings by the time they are hashed, and must be."""
    transport = embedding_transport()
    with Store(temp_database) as store:
        instance = embedder(store, transport)
        document = instance.embed_documents(["Vorname"])[0]
        query = instance.embed_queries(["Vorname"])[0]

    assert not np.allclose(document, query)


def test_vectors_are_l2_normalised_at_write(temp_database: Path) -> None:
    transport = embedding_transport()
    with Store(temp_database) as store:
        matrix = embedder(store, transport).embed_documents(["a", "b", "c"])

    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_the_whole_batch_goes_out_in_one_call(temp_database: Path) -> None:
    transport = embedding_transport()
    with Store(temp_database) as store:
        embedder(store, transport).embed_documents([f"field {index}" for index in range(30)])

    embeds = [call for call in transport.calls if call[0].endswith("/api/embed")]
    assert len(embeds) == 1
    assert len(embeds[0][1]["input"]) == 30  # type: ignore[index]


def test_a_second_pass_over_the_same_corpus_makes_zero_http_calls(
    temp_database: Path,
) -> None:
    texts = [f"field {index}" for index in range(12)]
    transport = embedding_transport()
    with Store(temp_database) as store:
        first = embedder(store, transport).embed_documents(texts)
    assert len(transport.bodies("/api/embed")) == 1

    with Store(temp_database) as store:
        second = embedder(store, transport).embed_documents(texts)

    assert len(transport.bodies("/api/embed")) == 1, "a warm cache must not embed anything again"
    assert np.array_equal(first, second)


def test_only_the_misses_are_sent_on_a_partly_warm_cache(temp_database: Path) -> None:
    transport = embedding_transport()
    with Store(temp_database) as store:
        instance = embedder(store, transport)
        instance.embed_documents(["a", "b"])
        instance.embed_documents(["a", "b", "c", "d"])

    last = transport.calls[-1]
    assert last[0].endswith("/api/embed")
    assert last[1]["input"] == ["title: none | text: c", "title: none | text: d"]  # type: ignore[index]


def test_a_repeated_text_in_one_batch_is_embedded_once(temp_database: Path) -> None:
    transport = embedding_transport()
    with Store(temp_database) as store:
        matrix = embedder(store, transport).embed_documents(["same", "same", "other"])

    assert len(transport.calls[-1][1]["input"]) == 2  # type: ignore[index]
    assert np.array_equal(matrix[0], matrix[1])


def test_matryoshka_truncation_is_applied_and_renormalised(temp_database: Path) -> None:
    """embeddinggemma returns 768 dims and the config record asks for 256."""
    transport = embedding_transport(dimensions=768)
    with Store(temp_database) as store:
        matrix = embedder(store, transport).embed_documents(["a"])

    assert matrix.shape == (1, config_for(DEFAULT_EMBEDDING_MODEL).dimensions)
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0)


def test_a_vector_shorter_than_the_configured_width_is_refused(temp_database: Path) -> None:
    transport = embedding_transport(dimensions=4)
    with Store(temp_database) as store, pytest.raises(ValueError, match="256"):
        embedder(store, transport).embed_documents(["a"])


def test_text_key_hashes_the_prefixed_string() -> None:
    assert text_key("abc") == hashlib.sha256(b"abc").hexdigest()


def test_l2_normalise_leaves_a_zero_vector_alone() -> None:
    """A zero row has no direction, and dividing by its norm would be a NaN."""
    matrix = l2_normalise(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))

    assert np.array_equal(matrix[0], np.zeros(2, dtype=np.float32))
    assert np.allclose(matrix[1], [0.6, 0.8])


def _index(vectors: list[list[float]]) -> EmbeddingIndex:
    matrix = l2_normalise(np.asarray(vectors, dtype=np.float32))
    return EmbeddingIndex(texts=tuple(f"t{i}" for i in range(len(vectors))), matrix=matrix)


def test_search_returns_the_true_nearest_neighbours_in_order() -> None:
    index = _index([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    found = index.search(l2_normalise(np.asarray([[1.0, 0.0]], dtype=np.float32))[0], k=2)

    assert [neighbour.index for neighbour in found] == [0, 1]
    assert found[0].score > found[1].score
    assert found[0].text == "t0"


def test_search_agrees_with_a_full_sort_on_a_larger_index() -> None:
    """argpartition is an optimisation, so it has to give the sort's answer."""
    rng = np.random.default_rng(0)
    matrix = l2_normalise(rng.normal(size=(400, 16)).astype(np.float32))
    index = EmbeddingIndex(texts=tuple(f"t{i}" for i in range(400)), matrix=matrix)
    query = l2_normalise(rng.normal(size=(1, 16)).astype(np.float32))[0]

    scores = matrix @ query
    expected = sorted(range(400), key=lambda i: (-scores[i], i))[:8]

    assert [neighbour.index for neighbour in index.search(query, k=8)] == expected


def test_asking_for_more_neighbours_than_there_are_returns_all_of_them() -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]])
    assert len(index.search(np.asarray([1.0, 0.0], dtype=np.float32), k=99)) == 2


def test_an_empty_index_returns_nothing_rather_than_raising() -> None:
    index = EmbeddingIndex(texts=(), matrix=np.zeros((0, 4), dtype=np.float32))
    assert index.search(np.zeros(4, dtype=np.float32), k=4) == []


def test_ties_are_broken_by_position_so_two_runs_agree() -> None:
    index = _index([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    found = index.search(np.asarray([1.0, 0.0], dtype=np.float32), k=2)

    assert [neighbour.index for neighbour in found] == [0, 1]


def test_an_index_whose_matrix_does_not_match_its_texts_is_refused() -> None:
    with pytest.raises(ValueError, match="3 text"):
        EmbeddingIndex(texts=("a", "b", "c"), matrix=np.zeros((2, 4), dtype=np.float32))
