"""Embedding a corpus once, and searching it exactly.

**No vector database, and that is the finding rather than the omission.** The
corpus this searches is the labelled training split of the reference workload:
about 10^3 vectors of 256 dimensions, which is a 1 MB matrix. One
`matrix @ query` over it is a single BLAS call costing tens of microseconds,
returns the TRUE nearest neighbours rather than an approximation, and is
reproducible to the bit. An approximate index would add a dependency, a build
step, a recall parameter and a second thing that can be stale, in exchange for
being slower at this size. The measured numbers are in `docs/llm.md`; the
one line version is that the search is not the bottleneck and the HTTP round
trip to the embedder is.

**The prompt prefixes live here, in the model config record, and nowhere else.**
`embeddinggemma` is trained asymmetrically: a query is embedded behind
`"task: search result | query: "` and a document behind `"title: none | text: "`,
each including its trailing space. Getting this wrong does not raise anything.
It degrades retrieval silently, by an amount nobody notices without an ablation,
which is exactly the class of defect this project exists to be careful about. So
no call site ever writes a prefix: it names a model, and the model record knows
what its own prefixes are. `bge-m3`, the opt in quality tier, uses none, and
that difference is data here rather than an `if` somewhere else.

**Cache keys include the model digest.** An embedding is deterministic for one
model at one quantisation and is not deterministic across Ollama versions,
drivers or a repull of the same tag, so the key is
`(model_name, model_digest, sha256(prefixed text))`. The text is hashed AFTER
the prefix is applied, which is what keeps the query and document embeddings of
one string in two different cache rows: they are two different inputs to the
model and would otherwise collide into whichever was computed first.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from triage.core.store import Store
from triage.llm.ollama_client import OllamaClient

#: Section 2's default embedder: 622 MB, multilingual including German, which
#: the de_DE half of the corpus requires. `nomic-embed-text` is rejected in the
#: specification for exactly that reason and is deliberately not registered.
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:300m"

#: The opt in quality tier. Bigger, 1024 dimensions, no prefixes, and NOT pulled
#: by anything here: it is a choice a reader makes, not a default.
BGE_M3 = "bge-m3"

#: How many texts go out in one `/api/embed` request. The endpoint takes an
#: array and the batch is the unit, which is the point: one request per string
#: would spend the whole wall clock on round trips. The chunk exists only to
#: bound one request's size, and at this corpus size the whole train split is
#: two or three requests rather than three thousand.
DEFAULT_BATCH_SIZE = 512


@dataclass(frozen=True)
class EmbeddingModel:
    """One embedder, its prefixes, and the width its vectors are kept at.

    `dimensions` is a truncation target rather than a description.
    `embeddinggemma` is a Matryoshka model: its 768 dimensional output is
    trained so that a prefix of it is itself a usable embedding, so this keeps
    the first 256 and renormalises. That is a quarter of the storage and a
    quarter of the dot product for a documented and small loss, and it is
    recorded here so the number is a property of the model rather than a
    constant somebody chose in a search function.
    """

    name: str
    query_prefix: str
    document_prefix: str
    dimensions: int
    note: str = ""


#: The registry. Two entries, because the specification names two and rejects a
#: third by name.
EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    DEFAULT_EMBEDDING_MODEL: EmbeddingModel(
        name=DEFAULT_EMBEDDING_MODEL,
        query_prefix="task: search result | query: ",
        document_prefix="title: none | text: ",
        dimensions=256,
        note="768 dimensions truncated to 256 (Matryoshka); asymmetric prefixes are mandatory",
    ),
    BGE_M3: EmbeddingModel(
        name=BGE_M3,
        query_prefix="",
        document_prefix="",
        dimensions=1024,
        note="opt in quality tier; 8192 token context and no prompt prefixes",
    ),
}


def config_for(name: str) -> EmbeddingModel:
    """The model record for `name`, or a refusal naming the ones that exist.

    An unregistered model is refused rather than run with empty prefixes,
    because empty prefixes are correct for one of the two supported models and
    silently wrong for the other, and a default that is right half the time is
    the worst possible default here.
    """
    config = EMBEDDING_MODELS.get(name)
    if config is None:
        raise ValueError(
            f"{name!r} has no embedding model record, so this does not know whether it needs "
            f"prompt prefixes. The registered models are "
            f"{', '.join(sorted(EMBEDDING_MODELS))}; add a record rather than a call site prefix"
        )
    return config


def text_key(text: str) -> str:
    """The cache key of one already prefixed string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving a zero row alone.

    Normalised at write, so that a cosine similarity is a dot product and the
    search is one matrix multiply with no per query division. A zero row has no
    direction to preserve and dividing by its norm would put a NaN into the
    matrix, which would then poison every score in a column rather than one.
    """
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(array / safe, dtype=np.float32)


@dataclass(frozen=True)
class Neighbour:
    """One retrieved item: where it is, what it is, and how close it is."""

    index: int
    score: float
    text: str


@dataclass(frozen=True)
class EmbeddingIndex:
    """A corpus and its normalised matrix, searched by exact brute force."""

    texts: tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.shape[0] != len(self.texts):
            raise ValueError(
                f"the index holds {len(self.texts)} text(s) and a matrix of "
                f"{self.matrix.shape[0]} row(s); a retrieved row would name the wrong document"
            )

    def __len__(self) -> int:
        return len(self.texts)

    def search(self, query: np.ndarray, k: int) -> list[Neighbour]:
        """The `k` nearest rows, highest score first, ties broken by position.

        `argpartition` selects the top k in linear time and the sort only
        orders those k, which is the whole optimisation; it is checked against a
        full sort in the tests, because an optimisation that gives a different
        answer from the thing it replaces is not an optimisation.

        Ties are broken by position rather than left to the partition, whose
        order among equal scores is unspecified. Two identical form fields in
        the corpus are exactly the case that produces a tie, and a few shot
        prompt that changed between two runs of the same command would take the
        determinism of this layer with it.
        """
        if len(self) == 0 or k <= 0:
            return []
        scores = np.asarray(self.matrix @ np.asarray(query, dtype=np.float32), dtype=np.float64)
        wanted = min(k, scores.size)
        if wanted < scores.size:
            candidates = np.argpartition(-scores, wanted - 1)[:wanted]
        else:
            candidates = np.arange(scores.size)
        order = sorted((int(index) for index in candidates), key=lambda i: (-scores[i], i))
        return [Neighbour(index=i, score=float(scores[i]), text=self.texts[i]) for i in order]


class Embedder:
    """Embeds text through Ollama, reading and writing the database cache."""

    def __init__(
        self,
        client: OllamaClient,
        store: Store,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.client = client
        self.store = store
        self.config = config_for(model)
        self.batch_size = max(1, batch_size)
        self._digest: str | None = None

    @property
    def digest(self) -> str:
        """The local digest of the model, read once and reused."""
        if self._digest is None:
            self._digest = self.client.digest_of(self.config.name)
        return self._digest

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus items, behind the model's DOCUMENT prefix."""
        return self._embed(texts, self.config.document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed queries, behind the model's QUERY prefix."""
        return self._embed(texts, self.config.query_prefix)

    def embed_query(self, text: str) -> np.ndarray:
        """One query, as a vector rather than a one row matrix."""
        return np.asarray(self.embed_queries([text])[0], dtype=np.float32)

    def index(self, texts: Sequence[str]) -> EmbeddingIndex:
        """Embed a corpus and hand back something searchable."""
        return EmbeddingIndex(texts=tuple(texts), matrix=self.embed_documents(texts))

    def _embed(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        width = self.config.dimensions
        if not texts:
            return np.zeros((0, width), dtype=np.float32)
        prefixed = [f"{prefix}{text}" for text in texts]
        keys = [text_key(text) for text in prefixed]

        vectors = self.store.get_embeddings(self.config.name, self.digest, keys)
        # Deduplicated before the request rather than after it: a corpus of form
        # fields repeats itself (two checkout pages spell a postal code the same
        # way), and paying the model twice for one string is the easiest waste
        # here to leave in place unnoticed.
        missing = list(dict.fromkeys(key for key in keys if key not in vectors))
        if missing:
            by_key = dict(zip(keys, prefixed, strict=True))
            fresh: dict[str, np.ndarray] = {}
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                raw = self.client.embed([by_key[key] for key in chunk], model=self.config.name)
                fresh.update(zip(chunk, (self._shape(row, width) for row in raw), strict=True))
            self.store.put_embeddings(self.config.name, self.digest, fresh)
            vectors.update(fresh)

        return np.vstack([vectors[key] for key in keys]).astype(np.float32)

    def _shape(self, row: Sequence[float], width: int) -> np.ndarray:
        """Truncate to the configured width and normalise, or refuse.

        A vector NARROWER than the record says is the shape of a model that is
        not the model the record describes, and truncating cannot fix that, so
        it raises. Truncating a wider one is the Matryoshka property and is what
        the record is asking for.
        """
        vector = np.asarray(row, dtype=np.float32)
        if vector.size < width:
            raise ValueError(
                f"{self.config.name} returned {vector.size} dimension(s) and its model record "
                f"asks for {width}. The record describes a different model from the one that "
                f"answered; check what `ollama pull {self.config.name}` fetched"
            )
        return np.asarray(l2_normalise(vector[:width].reshape(1, width))[0], dtype=np.float32)
