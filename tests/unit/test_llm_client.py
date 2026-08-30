"""The local Ollama client, driven entirely through an injected fake transport.

Nothing here opens a socket. The client's whole contract is that it speaks a
documented HTTP shape and turns every failure of that shape into one of two
exceptions, so a fake that returns canned bytes exercises all of it, including
the cases a real service produces rarely and a CI runner must never depend on:
a refused connection, a truncated body, a response missing the field the caller
asked for.
"""

from __future__ import annotations

import email.message
import io
import urllib.error

import pytest
from llm_fakes import TAGS, FakeTransport

from triage.llm.ollama_client import (
    DEFAULT_CHAT_MODEL,
    FALLBACK_CHAT_MODEL,
    MAX_NUM_CTX,
    LlmProtocolError,
    LlmUnavailableError,
    OllamaClient,
)


def client(transport: FakeTransport) -> OllamaClient:
    return OllamaClient(transport=transport)


def test_the_module_imports_with_no_service_running() -> None:
    """Importing must never touch the network, or a core install would fail."""
    import triage.llm.ollama_client as module

    assert module.DEFAULT_HOST.startswith("http://localhost")


def test_health_reports_the_models_and_their_digests() -> None:
    transport = FakeTransport({"/api/tags": TAGS})
    report = client(transport).health()

    assert report.names[0] == DEFAULT_CHAT_MODEL
    assert "embeddinggemma:300m" in report.names
    assert report.digest_of(DEFAULT_CHAT_MODEL) == "d" * 64
    assert transport.calls[0][0].endswith("/api/tags")


def test_a_refused_connection_names_the_remedy() -> None:
    transport = FakeTransport({"/api/tags": OSError("connection refused")})
    with pytest.raises(LlmUnavailableError) as error:
        client(transport).health()

    message = str(error.value)
    assert "ollama serve" in message
    assert "localhost:11434" in message


def test_a_model_that_is_not_pulled_names_the_pull_command() -> None:
    transport = FakeTransport({"/api/tags": TAGS})
    with pytest.raises(LlmUnavailableError) as error:
        client(transport).digest_of("bge-m3")

    assert "ollama pull bge-m3" in str(error.value)


def test_available_is_the_boolean_form_and_swallows_nothing_else() -> None:
    assert client(FakeTransport({"/api/tags": TAGS})).available() is True
    assert client(FakeTransport({"/api/tags": OSError("no")})).available() is False


def test_chat_sends_temperature_zero_a_fixed_seed_and_the_context_window() -> None:
    transport = FakeTransport({"/api/chat": {"message": {"content": "  hello  "}}})
    answer = client(transport).chat("say hello")

    assert answer == "hello"
    url, body = transport.calls[0]
    assert url.endswith("/api/chat")
    assert body is not None
    assert body["model"] == DEFAULT_CHAT_MODEL
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["seed"] == 0
    assert body["options"]["num_ctx"] == 4096
    assert body["messages"][-1] == {"role": "user", "content": "say hello"}


def test_a_system_prompt_is_sent_ahead_of_the_user_turn() -> None:
    transport = FakeTransport({"/api/chat": {"message": {"content": "ok"}}})
    client(transport).chat("question", system="be terse")

    body = transport.calls[0][1]
    assert body is not None
    assert body["messages"][0] == {"role": "system", "content": "be terse"}


def test_generate_uses_the_completion_endpoint() -> None:
    transport = FakeTransport({"/api/generate": {"response": "text"}})
    assert client(transport).generate("prompt") == "text"
    assert transport.calls[0][0].endswith("/api/generate")


def test_embed_sends_the_whole_batch_in_one_call() -> None:
    transport = FakeTransport({"/api/embed": {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}})
    vectors = client(transport).embed(["a", "b"], model="embeddinggemma:300m")

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert transport.n_calls == 1
    body = transport.calls[0][1]
    assert body is not None
    assert body["input"] == ["a", "b"]


def test_embedding_an_empty_batch_makes_no_call_at_all() -> None:
    transport = FakeTransport({})
    assert client(transport).embed([], model="embeddinggemma:300m") == []
    assert transport.n_calls == 0


def test_a_body_that_is_not_json_is_a_protocol_error() -> None:
    transport = FakeTransport({"/api/chat": b"<html>502 bad gateway</html>"})
    with pytest.raises(LlmProtocolError) as error:
        client(transport).chat("hi")

    assert "not JSON" in str(error.value)


def test_a_response_missing_the_field_the_caller_wanted_is_a_protocol_error() -> None:
    transport = FakeTransport({"/api/chat": {"done": True}})
    with pytest.raises(LlmProtocolError) as error:
        client(transport).chat("hi")

    assert "message" in str(error.value)


def test_an_embedding_batch_of_the_wrong_length_is_refused() -> None:
    transport = FakeTransport({"/api/embed": {"embeddings": [[1.0, 0.0]]}})
    with pytest.raises(LlmProtocolError) as error:
        client(transport).embed(["a", "b"], model="embeddinggemma:300m")

    assert "2" in str(error.value)


def test_a_context_window_beyond_the_maximum_is_refused_at_construction() -> None:
    with pytest.raises(ValueError) as error:
        OllamaClient(transport=FakeTransport(), num_ctx=MAX_NUM_CTX + 1)

    assert str(MAX_NUM_CTX) in str(error.value)


def test_the_fallback_chat_model_is_chosen_when_the_default_is_not_pulled() -> None:
    tags = {"models": [{"name": FALLBACK_CHAT_MODEL, "digest": "f" * 64, "size": 4_683_000_000}]}
    assert client(FakeTransport({"/api/tags": tags})).resolve_chat_model() == FALLBACK_CHAT_MODEL


def test_the_default_chat_model_wins_when_both_are_pulled() -> None:
    tags = {
        "models": [
            {"name": FALLBACK_CHAT_MODEL, "digest": "f" * 64, "size": 1},
            {"name": DEFAULT_CHAT_MODEL, "digest": "d" * 64, "size": 2},
        ]
    }
    assert client(FakeTransport({"/api/tags": tags})).resolve_chat_model() == DEFAULT_CHAT_MODEL


def test_no_chat_model_at_all_names_both_models_to_pull() -> None:
    transport = FakeTransport({"/api/tags": {"models": []}})
    with pytest.raises(LlmUnavailableError) as error:
        client(transport).resolve_chat_model()

    assert DEFAULT_CHAT_MODEL in str(error.value)
    assert FALLBACK_CHAT_MODEL in str(error.value)


def test_the_health_report_is_read_once_and_cached() -> None:
    """Two questions about the service must not cost two round trips."""
    transport = FakeTransport({"/api/tags": TAGS})
    instance = client(transport)
    instance.digest_of(DEFAULT_CHAT_MODEL)
    instance.digest_of("embeddinggemma:300m")

    assert transport.n_calls == 1


# ------------------------------------------------- a server that says no
#
# An `HTTPError` is an `OSError`, so before these tests a server that answered
# 400 was reported as "no Ollama at localhost" and sent the reader off to
# restart a service that was running. The real one does exactly this when an
# embedding batch is too large for it; see docs/ENGINEERING_LOG.md.


def http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://localhost:11434/api/embed",
        code=code,
        msg="refused",
        hdrs=email.message.Message(),
        fp=io.BytesIO(body),
    )


def test_a_refused_request_is_not_reported_as_a_missing_service() -> None:
    body = b'{"error":"Post \\"http://127.0.0.1:59907/tokenize\\": connection refused"}'
    transport = FakeTransport({"/api/embed": http_error(400, body)})

    with pytest.raises(LlmProtocolError) as error:
        client(transport).embed(["a"], model="embeddinggemma:300m")

    message = str(error.value)
    assert "HTTP 400" in message
    assert "tokenize" in message, "the server's own explanation has to survive"
    assert "restarting it will not help" in message
    assert "ollama serve" not in message


def test_a_404_is_a_missing_model_and_says_how_to_get_one() -> None:
    transport = FakeTransport({"/api/chat": http_error(404, b'{"error":"model not found"}')})

    with pytest.raises(LlmUnavailableError) as error:
        client(transport).chat("hi")

    assert "ollama pull" in str(error.value)
    assert "model not found" in str(error.value)


def test_an_error_body_that_is_not_json_is_still_reported() -> None:
    transport = FakeTransport({"/api/chat": http_error(502, b"<html>bad gateway</html>")})

    with pytest.raises(LlmProtocolError) as error:
        client(transport).chat("hi")

    assert "bad gateway" in str(error.value)
