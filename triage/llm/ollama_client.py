"""A small stdlib client for a local Ollama server, and deliberately small.

Four endpoints, `urllib.request`, no dependency, about two hundred lines. It is
the THIRD such client across these three repositories and that is a decision
rather than an oversight (E7d): a shared client library would need a home, a
release cadence and a compatibility promise between three projects that
currently share nothing but a value space, and it would buy each of them a
hundred lines. So this one stays minimal and does not try to be general.

**Determinism is a property of the call, not of the caller.** Every request
carries `temperature: 0` and a fixed `seed`, set here rather than at the call
sites, because a generated summary that changes between two runs of the same
command would break the byte determinism the reports are held to. `num_ctx`
defaults to 4096 and refuses anything above 8192, which is the ceiling Section 2
measured: a 12B model at Q4_K_M is 7.5 GB of weights on a 12,227 MiB card, and
the headroom that keeps it fully GPU resident is the KV cache, so the context
length is the knob that decides whether this runs on the GPU or spills to CPU.

**Two exceptions, and the difference between them matters.** An
`LlmUnavailableError` means the service or the model is not there, and its
message always names the command that would fix it: `ollama serve`,
`ollama pull <model>`. An `LlmProtocolError` means something answered but not
with the shape this speaks, which is a bug or a proxy rather than something the
reader can install. Callers treat the first as a reason to skip and the second
as a reason to stop.

**Importing never touches the network.** A module that probed the service on
import would make `pip install ml-experiment-triage` fail on a machine with no
Ollama, and every entry point below raises rather than returning a fallback, so
nothing can silently produce numbers no model generated.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

#: Where a stock Ollama listens. Overridable, but never at a call site: a host
#: is a property of the machine, so it belongs on the client.
DEFAULT_HOST = "http://localhost:11434"

#: Section 2. 7.5 GB of weights at Q4_K_M, fully resident on a 12,227 MiB card
#: with a 4096 to 8192 token context. The Q5_K_M variant of the same model is
#: on disk here and is deliberately NOT the default: 8.7 GB plus a long context
#: spills into CPU offload, which is the failure mode this project avoids.
DEFAULT_CHAT_MODEL = "mistral-nemo:12b-instruct-2407-q4_K_M"

#: The fast fallback, chosen automatically when the default is not pulled.
FALLBACK_CHAT_MODEL = "qwen2.5:7b-instruct-q4_K_M"

#: Context window defaults and the ceiling above. Both from Section 2.
DEFAULT_NUM_CTX = 4096
MAX_NUM_CTX = 8192

#: The seed every request carries. Zero, and fixed: see the module docstring.
DEFAULT_SEED = 0

#: Seconds to wait for one response. Generous, because a 12B model answering a
#: few hundred tokens on a consumer card takes tens of seconds and a timeout
#: that fires mid sweep is indistinguishable from a broken service.
DEFAULT_TIMEOUT = 300.0


class LlmError(RuntimeError):
    """Anything this client refuses to pretend it can do."""


class LlmUnavailableError(LlmError):
    """The service or the model is absent. The message names what to type."""


class LlmProtocolError(LlmError):
    """Something answered, but not in the shape this client speaks."""


class Transport(Protocol):
    """One HTTP round trip: bytes in, bytes out.

    Injectable so that every test in CI runs against canned bytes. The signature
    is deliberately narrower than an HTTP library's: this client sends JSON to a
    localhost URL and reads a body, and anything richer would be surface nobody
    needs and a fake would have to imitate.
    """

    def __call__(self, url: str, payload: bytes | None, timeout: float) -> bytes: ...


def urllib_transport(url: str, payload: bytes | None, timeout: float) -> bytes:
    """The real transport: one `urllib.request` round trip against localhost."""
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return bytes(body)


def _server_message(error: urllib.error.HTTPError) -> str:
    """Ollama's own explanation of a refusal, or its status line.

    The body of an Ollama error is `{"error": "..."}` and it is far more useful
    than the status: the 400 that motivated this said the embedding runner had
    died, which is a fact about the batch that was sent and not about anything
    the status code could express. Reading it can itself fail, on a body already
    consumed or one that is not JSON, and a diagnostic that raises while
    reporting an error is worse than a vague one.
    """
    try:
        raw = error.read()
    except Exception:
        # Deliberately every exception: this function's whole job is to explain
        # another failure, and one that raised while doing so would replace a
        # message a reader can act on with a traceback from the reporting path.
        return error.reason if isinstance(error.reason, str) else str(error)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw[:200].decode("utf-8", errors="replace") or str(error)
    if isinstance(decoded, dict) and "error" in decoded:
        return str(decoded["error"])
    return raw[:200].decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ModelInfo:
    """One model the server has locally, with the digest that keys a cache."""

    name: str
    digest: str
    size: int = 0


@dataclass(frozen=True)
class HealthReport:
    """What `/api/tags` said, which is the only health check that means anything.

    A server that answers `/api/tags` can serve; one that does not is down. The
    digests come back in the same call, which is why the health check is also
    where cache keys are sourced from: an embedding is deterministic for a fixed
    model AND a fixed build of it, and the digest is the only thing that changes
    when the same tag is repulled after an upstream requantisation.
    """

    host: str
    models: tuple[ModelInfo, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(model.name for model in self.models)

    def digest_of(self, model: str) -> str | None:
        for info in self.models:
            if info.name == model:
                return info.digest
        return None


@dataclass
class OllamaClient:
    """Chat, completion, embeddings and health against one local Ollama."""

    host: str = DEFAULT_HOST
    chat_model: str = DEFAULT_CHAT_MODEL
    num_ctx: int = DEFAULT_NUM_CTX
    seed: int = DEFAULT_SEED
    timeout: float = DEFAULT_TIMEOUT
    #: `None` means the real one. It is resolved at the moment of the call
    #: rather than bound as a dataclass default, and that is the difference
    #: between a seam and a decoration: a default bound at class creation cannot
    #: be replaced afterwards, so an integration test driving a whole command
    #: could not have substituted one. Looking the module attribute up per call
    #: costs a dictionary read and makes the substitution possible.
    transport: Transport | None = None
    _health: HealthReport | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.num_ctx <= MAX_NUM_CTX:
            raise ValueError(
                f"num_ctx {self.num_ctx} is outside 1 to {MAX_NUM_CTX}: a longer context on "
                f"this card pushes the KV cache past the VRAM the weights leave free, and the "
                f"model silently offloads to CPU instead of failing"
            )

    # ------------------------------------------------------------- plumbing

    def _url(self, path: str) -> str:
        return f"{self.host.rstrip('/')}{path}"

    def _round_trip(self, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = self._url(path)
        encoded = None if payload is None else json.dumps(payload).encode("utf-8")
        send = self.transport if self.transport is not None else urllib_transport
        try:
            body = send(url, encoded, self.timeout)
        # An `HTTPError` FIRST, and the order is the whole point of this pair.
        # It subclasses `URLError` which subclasses `OSError`, so a server that
        # answered 400 used to be reported as "no Ollama at localhost", which
        # sent a reader off to restart a service that was running perfectly and
        # said nothing about the request that had actually been refused. This
        # was found by running the real thing (docs/ENGINEERING_LOG.md).
        except urllib.error.HTTPError as error:
            detail = _server_message(error)
            if error.code == 404:
                raise LlmUnavailableError(
                    f"{url} answered 404: {detail}. That is usually a model this server does "
                    f"not have; `ollama list` shows what it has and `ollama pull` fetches one"
                ) from error
            raise LlmProtocolError(
                f"{url} answered HTTP {error.code}: {detail}. The server is running and "
                f"refused this request, so restarting it will not help; the message above "
                f"is Ollama's own"
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise LlmUnavailableError(
                f"no Ollama at {self.host} ({error}). Start it with `ollama serve`, or point "
                f"--host at the machine that is running one. Everything in this layer is local: "
                f"nothing here falls back to a hosted model"
            ) from error
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LlmProtocolError(
                f"{url} answered with something that is not JSON ({error}); the first bytes "
                f"were {body[:60]!r}. Something other than Ollama is listening on that port"
            ) from error
        if not isinstance(decoded, dict):
            raise LlmProtocolError(f"{url} answered with {type(decoded).__name__}, not an object")
        return decoded

    @staticmethod
    def _field(body: dict[str, Any], key: str, url: str) -> Any:
        if key not in body:
            raise LlmProtocolError(
                f"{url} answered without a {key!r} field; it held {sorted(body)}. This client "
                f"speaks the Ollama REST API and something else answered"
            )
        return body[key]

    # --------------------------------------------------------------- health

    def health(self, refresh: bool = False) -> HealthReport:
        """The models the server holds, or `LlmUnavailableError` if it is down.

        Cached, because every later call wants a digest and a client that asked
        the service twice for the same list would double the latency of a cache
        hit, which is the case this whole layer is optimised for.
        """
        if self._health is None or refresh:
            body = self._round_trip("/api/tags", None)
            raw = self._field(body, "models", self._url("/api/tags"))
            self._health = HealthReport(
                host=self.host,
                models=tuple(
                    ModelInfo(
                        name=str(entry.get("name", "")),
                        digest=str(entry.get("digest", "")),
                        size=int(entry.get("size", 0) or 0),
                    )
                    for entry in raw
                ),
            )
        return self._health

    def available(self) -> bool:
        """`True` when the service answers. The boolean form of `health`."""
        try:
            self.health()
        except LlmError:
            return False
        return True

    def digest_of(self, model: str) -> str:
        """The local digest of one model, or a refusal naming the pull command."""
        digest = self.health().digest_of(model)
        if digest is None:
            raise LlmUnavailableError(
                f"{model} is not pulled on {self.host}. Run `ollama pull {model}` first; the "
                f"models present are {', '.join(self.health().names) or 'none'}"
            )
        return digest

    def resolve_chat_model(self, preferred: str | None = None) -> str:
        """The best chat model actually present, preferring the configured one."""
        wanted = preferred or self.chat_model
        present = set(self.health().names)
        for candidate in (wanted, DEFAULT_CHAT_MODEL, FALLBACK_CHAT_MODEL):
            if candidate in present:
                return candidate
        raise LlmUnavailableError(
            f"neither {DEFAULT_CHAT_MODEL} nor {FALLBACK_CHAT_MODEL} is pulled on {self.host}. "
            f"Run `ollama pull {DEFAULT_CHAT_MODEL}` for the default, or "
            f"`ollama pull {FALLBACK_CHAT_MODEL}` for the faster fallback"
        )

    # ------------------------------------------------------------ inference

    def _options(self) -> dict[str, Any]:
        return {"temperature": 0.0, "seed": self.seed, "num_ctx": self.num_ctx}

    def chat(self, prompt: str, system: str | None = None, model: str | None = None) -> str:
        """One turn against `/api/chat`, returning the assistant's text."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = self._round_trip(
            "/api/chat",
            {
                "model": model or self.chat_model,
                "messages": messages,
                "stream": False,
                "options": self._options(),
            },
        )
        message = self._field(body, "message", self._url("/api/chat"))
        if not isinstance(message, dict) or "content" not in message:
            raise LlmProtocolError(
                f"{self._url('/api/chat')} answered with a 'message' that carries no 'content'"
            )
        return str(message["content"]).strip()

    def generate(self, prompt: str, model: str | None = None) -> str:
        """One completion against `/api/generate`."""
        body = self._round_trip(
            "/api/generate",
            {
                "model": model or self.chat_model,
                "prompt": prompt,
                "stream": False,
                "options": self._options(),
            },
        )
        return str(self._field(body, "response", self._url("/api/generate"))).strip()

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        """Embed a whole batch in ONE call, which is what `/api/embed` is for.

        The corpus this embeds is a few thousand short strings. One request per
        string would spend its entire wall clock on HTTP round trips against a
        model that batches internally, so the batch is the unit here and the
        caller above deduplicates against the cache before calling.
        """
        if not texts:
            return []
        body = self._round_trip(
            "/api/embed",
            {"model": model, "input": list(texts), "options": self._options()},
        )
        raw = self._field(body, "embeddings", self._url("/api/embed"))
        vectors = [[float(value) for value in row] for row in raw]
        if len(vectors) != len(texts):
            raise LlmProtocolError(
                f"asked {model} for {len(texts)} embedding(s) and got {len(vectors)}; a batch "
                f"that comes back a different length cannot be matched to its inputs"
            )
        return vectors
