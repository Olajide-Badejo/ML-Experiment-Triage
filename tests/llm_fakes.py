"""The injected fake transport every LLM test in CI runs against.

Section 6.6: no test in this repository opens a socket to a model, and this is
the object that makes that true. It is shared rather than copied per test module
because the interesting fakes are the awkward ones (a truncated body, a chat
turn that answers with prose instead of JSON, an embedding batch that comes back
the wrong length), and a copy of those in five files would drift until only one
of them still tested the case it was written for.

The call COUNT is as much of the contract as the payloads. The caches in this
layer are only worth having if a repeated request never reaches the network, and
a transport that reports how many times it was used is the only way to assert
that without trusting the thing under test to tell the truth about itself.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from triage.llm.ollama_client import DEFAULT_CHAT_MODEL, FALLBACK_CHAT_MODEL

#: The digests the fakes report. Real ones are 64 hex characters and these are
#: too, because a cache key is built from them and a short one would not catch a
#: slicing mistake.
CHAT_DIGEST = "d" * 64
EMBED_DIGEST = "e" * 64

EMBED_MODEL = "embeddinggemma:300m"

#: What `/api/tags` answers: the models Section 2 names, as this machine has
#: them.
TAGS: dict[str, Any] = {
    "models": [
        {"name": DEFAULT_CHAT_MODEL, "digest": CHAT_DIGEST, "size": 7_477_000_000},
        {"name": FALLBACK_CHAT_MODEL, "digest": "f" * 64, "size": 4_683_000_000},
        {"name": EMBED_MODEL, "digest": EMBED_DIGEST, "size": 621_900_000},
    ]
}


class FakeTransport:
    """Canned responses matched by URL suffix, counting every call made.

    A value in `responses` may be a JSON serialisable object, raw `bytes` (for a
    body that is not JSON at all), an exception instance (raised, which is how a
    refused connection is spelled), or a callable taking the decoded request body
    and returning the object to answer with.
    """

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses: dict[str, Any] = responses or {}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def __call__(self, url: str, payload: bytes | None, timeout: float) -> bytes:
        body = None if payload is None else json.loads(payload.decode("utf-8"))
        self.calls.append((url, body))
        for suffix, response in self.responses.items():
            if url.endswith(suffix):
                if isinstance(response, BaseException):
                    raise response
                if isinstance(response, bytes):
                    return response
                if callable(response):
                    return json.dumps(response(body)).encode("utf-8")
                return json.dumps(response).encode("utf-8")
        raise AssertionError(f"the fake transport has no canned response for {url}")

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def bodies(self, suffix: str) -> list[dict[str, Any]]:
        """The request bodies sent to one endpoint, in order."""
        return [body for url, body in self.calls if url.endswith(suffix) and body is not None]


def fake_embeddings(dimensions: int = 8) -> Any:
    """An embedder whose vector is a deterministic function of its input text.

    Text dependent on purpose. A fake that returned one constant vector would
    let a prefix bug and a cache bug both pass, because serving the wrong vector
    would be indistinguishable from serving the right one.
    """

    def embed(body: dict[str, Any]) -> dict[str, Any]:
        rows = []
        for text in body["input"]:
            seed = int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            rows.append([float(value) for value in rng.normal(size=dimensions)])
        return {"embeddings": rows}

    return embed
