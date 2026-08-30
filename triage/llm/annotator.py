"""Classifying a form field by showing the model its nearest labelled examples.

**Why retrieval is here and nowhere else in this package.** Retrieval augmented
FEW SHOT SELECTION for classification is the one use in Section 6 with a
documented, measurable gain: instead of a fixed handful of examples chosen by
whoever wrote the prompt, each field is classified beside the k labelled fields
that are actually most like it, which is the KATE pattern. That claim is not
taken on faith here. `--ablation` runs the same eval split twice, zero shot and
retrieval augmented, writes both as ordinary triage run directories, and reports
the difference as a verdict with a p value through this repository's own
comparison layer. If the lift is not real, the tool says so.

**Three tiers of confidence, and none of them is a probability.** A chat model
emits a token, not a posterior, and the number this attaches is the same kind of
thing the heuristic baseline attaches: a record of which path produced the
answer. First pass parses are more trustworthy than answers that needed the
retry, which are more trustworthy than the `unknown` a second failure produces.
Calling that a calibrated probability would be the exact dishonesty
`calibration.py` exists to argue against, so the tiers are named constants and
the docstring says what they are.

**Strict parsing, one retry, then `unknown`, counted.** The model is asked for
`{"field_type": "..."}` and nothing else. A response that is not that is retried
once with a corrective instruction; a second failure is recorded as `unknown`
and counted, because a classifier that silently substituted its own guess for a
model's non answer would report an accuracy that is partly its own.

**The cache is what makes this reproducible.** Every request is keyed by
`sha256(model + digest + prompt)` in the same SQLite database as everything
else, so a second invocation of the same command makes ZERO HTTP calls, an
interrupted pass resumes, and a report built twice from one database is the same
report. The digest is in the key because the same model tag repulled is a
different model and its answers are not interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triage.autofill.evaluate import ScoredEngine
from triage.autofill.generator import FieldRecord
from triage.autofill.taxonomy import FIELD_TYPES, FieldType, index_of
from triage.core.store import Store
from triage.llm.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, EmbeddingIndex
from triage.llm.ollama_client import DEFAULT_CHAT_MODEL, OllamaClient

#: Section 6.3's default: eight nearest labelled fields per prompt.
DEFAULT_K = 8

#: The engine names the two ablation arms write into their run directories and
#: their outcomes rows. They are variant keys as well as labels, so they are
#: constants rather than f strings at three call sites.
ENGINE_LLM = "llm"
ENGINE_KNN = "llm_knn"
ENGINE_ZERO_SHOT = "llm_zero_shot"

#: Confidence TIERS, not probabilities. See the module docstring.
FIRST_PASS_CONFIDENCE = 0.80
RETRY_CONFIDENCE = 0.55
UNKNOWN_CONFIDENCE = 0.20

#: The taxonomy, as the prompt spells it.
ALLOWED_TOKENS = ", ".join(member.value for member in FIELD_TYPES)

SYSTEM_PROMPT = (
    "You label HTML form fields with WHATWG autocomplete tokens. "
    "You answer with one JSON object and nothing else: no prose, no code fence, "
    "no explanation. The object has exactly one key, field_type, whose value is "
    f"one of these tokens: {ALLOWED_TOKENS}. "
    "Use unknown when the field is none of the others."
)

#: What the retry adds. Short and specific: a long scolding costs context and
#: the failure it is correcting is almost always a format failure rather than a
#: comprehension one.
RETRY_INSTRUCTION = (
    "Your previous answer was not a single JSON object with one key, field_type. "
    "Answer again with exactly that and nothing else."
)


def render_signals(record: FieldRecord) -> str:
    """One field as the single line that is both embedded and shown to the model.

    One rendering for both jobs on purpose. If the retrieval text and the prompt
    text differed, the examples chosen as nearest would be nearest to something
    the model never sees, and the resulting drop in quality would look like the
    model being bad at the task.

    The order is fixed and every signal is present even when empty, so that two
    fields differing in one attribute produce strings differing in one place.
    """
    return (
        f"locale={record.locale} section={record.section} "
        f'label="{record.label}" name="{record.name}" id="{record.element_id}" '
        f'placeholder="{record.placeholder}" autocomplete="{record.autocomplete}" '
        f'type={record.input_type} previous="{record.previous_label}" '
        f'next="{record.next_label}"'
    )


def response_cache_key(model: str, digest: str, prompt: str) -> str:
    """`sha256(model + digest + prompt)`, per 6.3.

    All three parts, and the digest for the same reason it keys the embedding
    cache: one model tag repulled is a different model, and a cache that served
    the old model's answers under the new one's name would make a rerun
    reproduce a number the current model does not produce.
    """
    return hashlib.sha256(f"{model}\x00{digest}\x00{prompt}".encode()).hexdigest()


def parse_field_type(text: str) -> FieldType | None:
    """The taxonomy token in a strict `{"field_type": "..."}`, or `None`.

    A markdown code fence around the object is stripped first, and that is the
    only latitude given. Instruction tuned models fence JSON often enough that
    refusing it would make the retry the common path rather than the exception,
    and a fence is a formatting wrapper rather than prose: what is inside it
    still has to parse strictly, be an object, carry exactly the expected key,
    and hold a token this taxonomy has.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or "field_type" not in parsed:
        return None
    try:
        return FieldType(str(parsed["field_type"]).strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class AnnotatorConfig:
    """Everything about one annotation pass that a reader might want to change."""

    k: int = DEFAULT_K
    retrieval: bool = True
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBEDDING_MODEL
    #: The engine name written into the run directories and the outcomes rows.
    engine: str = ENGINE_LLM
    #: How many eval rows to classify. `None` is all of them; a number is the
    #: quick pass, and is recorded in the artifacts so a small run is never
    #: mistaken for a full one.
    limit: int | None = None


@dataclass(frozen=True)
class Annotation:
    """One field's answer, and how it was arrived at."""

    field_type: FieldType
    confidence: float
    attempts: int
    cached: bool


@dataclass(frozen=True)
class AnnotationRun:
    """A whole pass: the answers, what they cost, and what went wrong."""

    engine: str
    chat_model: str
    annotations: tuple[Annotation, ...] = ()
    unknown: int = 0
    retried: int = 0
    cache_hits: int = 0
    requests: int = 0
    seconds: float = 0.0
    retrieval: bool = True
    k: int = DEFAULT_K

    @property
    def predicted(self) -> np.ndarray:
        return np.asarray([index_of(item.field_type) for item in self.annotations], dtype=np.int64)

    @property
    def confidence(self) -> np.ndarray:
        return np.asarray([item.confidence for item in self.annotations], dtype=np.float64)

    @property
    def mean_latency_us(self) -> float:
        """Wall clock per field, in microseconds, including cache hits.

        Including the hits is the honest measure of what the pass cost, and it
        is why this number moves by three orders of magnitude between a cold and
        a warm run. The evaluation artifacts carry it; `docs/llm.md` says what
        both look like.
        """
        if not self.annotations:
            return 0.0
        return self.seconds * 1e6 / len(self.annotations)

    def scored(self) -> ScoredEngine:
        """The same shape every other engine reports, so it joins the table."""
        return ScoredEngine(
            engine=self.engine,
            predicted=self.predicted,
            confidence=self.confidence,
            latency_us=self.mean_latency_us,
        )

    def describe(self) -> str:
        mode = f"kNN k={self.k}" if self.retrieval else "zero shot"
        return (
            f"{self.engine} ({mode}, {self.chat_model}): {len(self.annotations)} field(s) in "
            f"{self.seconds:.1f} s, {self.requests} request(s), {self.cache_hits} cache hit(s), "
            f"{self.retried} retried, {self.unknown} unparsed"
        )


class Annotator:
    """Classifies fields with a local chat model, with or without retrieval."""

    def __init__(
        self,
        client: OllamaClient,
        store: Store,
        config: AnnotatorConfig | None = None,
        examples: Sequence[FieldRecord] = (),
    ) -> None:
        self.client = client
        self.store = store
        self.config = config or AnnotatorConfig()
        self.examples = list(examples)
        if self.config.retrieval and not self.examples:
            raise ValueError(
                "retrieval augmented annotation needs labelled examples to retrieve from: pass "
                "the training split, or set retrieval=False for the zero shot arm"
            )
        self._index: EmbeddingIndex | None = None
        self._embedder: Embedder | None = None

    # ------------------------------------------------------------- retrieval

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.client, self.store, model=self.config.embed_model)
        return self._embedder

    def index(self) -> EmbeddingIndex:
        """The example corpus, embedded once and reused for every field."""
        if self._index is None:
            self._index = self.embedder.index([render_signals(item) for item in self.examples])
        return self._index

    def neighbours(self, record: FieldRecord) -> list[FieldRecord]:
        """The k labelled fields nearest this one, LEAST similar first.

        Least similar first so the closest example sits immediately before the
        field being classified. Recency in a prompt is not free, and the whole
        point of choosing examples by similarity is that the nearest one is the
        one most worth imitating.
        """
        query = self.embedder.embed_query(render_signals(record))
        found = self.index().search(query, k=self.config.k)
        return [self.examples[neighbour.index] for neighbour in reversed(found)]

    # ---------------------------------------------------------------- prompt

    def build_prompt(self, record: FieldRecord, examples: Sequence[FieldRecord]) -> str:
        """The user turn: the examples, then the field, then the format rule."""
        parts: list[str] = []
        if examples:
            parts.append("Labelled examples, least similar first:")
            for item in examples:
                parts.append(f'{render_signals(item)}\n{{"field_type": "{item.field_type.value}"}}')
            parts.append("")
        parts.append("Field to label:")
        parts.append(render_signals(record))
        parts.append("")
        parts.append('Answer with one JSON object: {"field_type": "<token>"}')
        return "\n".join(parts)

    # ------------------------------------------------------------- inference

    def _ask(self, prompt: str, digest: str, counters: dict[str, int]) -> str:
        """One turn, served from the cache when the cache has it."""
        key = response_cache_key(self.config.chat_model, digest, prompt)
        cached = self.store.get_response(key)
        if cached is not None:
            counters["cache_hits"] += 1
            return cached
        answer = self.client.chat(prompt, system=SYSTEM_PROMPT, model=self.config.chat_model)
        counters["requests"] += 1
        self.store.put_response(key, self.config.chat_model, digest, answer)
        return answer

    def classify(self, record: FieldRecord, digest: str, counters: dict[str, int]) -> Annotation:
        """One field: ask, parse, retry once, then give up and count it."""
        examples = self.neighbours(record) if self.config.retrieval else []
        prompt = self.build_prompt(record, examples)
        before = counters["cache_hits"]
        answer = self._ask(prompt, digest, counters)
        parsed = parse_field_type(answer)
        cached = counters["cache_hits"] > before
        if parsed is not None:
            return Annotation(
                field_type=parsed,
                confidence=FIRST_PASS_CONFIDENCE,
                attempts=1,
                cached=cached,
            )

        counters["retried"] += 1
        retry_prompt = f"{prompt}\n\n{RETRY_INSTRUCTION}"
        parsed = parse_field_type(self._ask(retry_prompt, digest, counters))
        if parsed is not None:
            return Annotation(
                field_type=parsed, confidence=RETRY_CONFIDENCE, attempts=2, cached=False
            )
        counters["unknown"] += 1
        return Annotation(
            field_type=FieldType.UNKNOWN,
            confidence=UNKNOWN_CONFIDENCE,
            attempts=2,
            cached=False,
        )

    def annotate(self, records: Sequence[FieldRecord]) -> AnnotationRun:
        """Classify every field (or the first `--limit` of them), timed."""
        rows = list(records)
        if self.config.limit is not None:
            rows = rows[: self.config.limit]
        digest = self.client.digest_of(self.config.chat_model)
        counters = {"cache_hits": 0, "requests": 0, "retried": 0, "unknown": 0}
        started = time.perf_counter()
        annotations = tuple(self.classify(record, digest, counters) for record in rows)
        elapsed = time.perf_counter() - started
        return AnnotationRun(
            engine=self.config.engine,
            chat_model=self.config.chat_model,
            annotations=annotations,
            unknown=counters["unknown"],
            retried=counters["retried"],
            cache_hits=counters["cache_hits"],
            requests=counters["requests"],
            seconds=elapsed,
            retrieval=self.config.retrieval,
            k=self.config.k,
        )


@dataclass(frozen=True)
class AblationFinding:
    """One row of the comparison layer's verdict table, as plain data.

    Copied out of the analysis layer's `Finding` rather than carried as one, so
    that a caller can print the ablation without importing the analysis stack
    and so that this dataclass is what gets serialised into the artifacts. The
    numbers are the ones the comparison produced; nothing here recomputes any of
    them.
    """

    candidate: str
    tag: str
    relative_effect_pct: float
    p_value: float
    adjusted_p: float
    verdict: str
    weak_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "tag": self.tag,
            "relative_effect_pct": self.relative_effect_pct,
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
            "verdict": self.verdict,
            "weak_mode": self.weak_mode,
        }


@dataclass(frozen=True)
class AblationResult:
    """The two arms, and the verdict the comparison layer returned about them."""

    zero_shot: AnnotationRun
    retrieval: AnnotationRun
    root: Path
    findings: tuple[AblationFinding, ...] = ()
    baseline: str = ""
    seconds: float = 0.0
    macro_f1: dict[str, float] = field(default_factory=dict)

    @property
    def lift(self) -> float:
        """The macro F1 difference retrieval bought, in the metric's own units."""
        return self.macro_f1.get(ENGINE_KNN, 0.0) - self.macro_f1.get(ENGINE_ZERO_SHOT, 0.0)

    @property
    def verdict(self) -> AblationFinding | None:
        """The macro F1 finding for the retrieval arm, which is the claim."""
        wanted = f"{ENGINE_KNN}_{self.retrieval_key}"
        for finding in self.findings:
            if finding.tag == "eval/macro_f1" and finding.candidate == wanted:
                return finding
        return None

    #: The `locale_split` suffix both arms' variant keys carry, recovered from
    #: the baseline so this object stays a pure record of what happened.
    retrieval_key: str = ""

    def headline(self) -> str:
        """The one sentence the specification asks for: a lift with a p value."""
        both = (
            f"retrieval augmented macro F1 {self.macro_f1.get(ENGINE_KNN, 0.0):.4f} against "
            f"zero shot {self.macro_f1.get(ENGINE_ZERO_SHOT, 0.0):.4f} ({self.lift:+.4f})"
        )
        finding = self.verdict
        if finding is None:
            return f"{both}; the comparison layer returned no verdict on eval/macro_f1"
        return (
            f"{both}: {finding.verdict} at p {finding.p_value:.4f}, "
            f"adjusted p {finding.adjusted_p:.4f}"
        )


def run_ablation(
    client: OllamaClient,
    store: Store,
    records: Sequence[FieldRecord],
    examples: Sequence[FieldRecord],
    out_dir: Path | str,
    split: str = "val",
    locale: str = "de_DE",
    bootstrap: int = 5,
    seed: int = 0,
    config: AnnotatorConfig | None = None,
) -> AblationResult:
    """Score both arms, write them as run directories, and ask triage the question.

    The measurement is deliberately not computed here. Both arms are written as
    ordinary run directories, ingested with `triage ingest` and compared with
    `compare_all`, so the lift is judged by the same seed replicated permutation
    test, the same practical gate and the same false discovery correction as
    every other claim this tool makes. A number this package computed about
    itself with its own arithmetic would be exactly the kind of self report the
    rest of the project refuses to accept from anybody else.
    """
    from triage.analysis.comparison import ComparisonConfig, compare_all
    from triage.analysis.regression import RegressionConfig, rank
    from triage.analysis.regression import classify as classify_findings
    from triage.autofill.evaluate import summarise, write_evaluation
    from triage.ingest import ingest
    from triage.parsers import parsers_for

    base = config or AnnotatorConfig()
    rows = [record for record in records if locale == "all" or record.locale == locale]
    if base.limit is not None:
        rows = rows[: base.limit]
    pool = [record for record in examples if locale == "all" or record.locale == locale]

    started = time.perf_counter()
    zero_shot = Annotator(
        client,
        store,
        AnnotatorConfig(
            k=base.k,
            retrieval=False,
            chat_model=base.chat_model,
            embed_model=base.embed_model,
            engine=ENGINE_ZERO_SHOT,
        ),
    ).annotate(rows)
    retrieval = Annotator(
        client,
        store,
        AnnotatorConfig(
            k=base.k,
            retrieval=True,
            chat_model=base.chat_model,
            embed_model=base.embed_model,
            engine=ENGINE_KNN,
        ),
        examples=pool,
    ).annotate(rows)

    root = Path(out_dir)
    paths = write_evaluation(
        root,
        records=list(rows),
        model=None,
        split=split,
        locale=locale,
        bootstrap=bootstrap,
        seed=seed,
        extra_engines={ENGINE_ZERO_SHOT: zero_shot.scored(), ENGINE_KNN: retrieval.scored()},
    )
    report = summarise(paths)
    macro_f1 = {
        name: float(engine["macro_f1"]) for name, engine in sorted(report["engines"].items())
    }

    baseline = f"{ENGINE_ZERO_SHOT}_{locale}_{split}"
    database = root / "ablation.db"
    # Its own database, not the caller's. The caller's holds the response cache
    # and whatever else they have ingested; an ablation is a self contained
    # measurement and should not quietly add ten runs to somebody's sweep.
    with Store(database) as ablation_store:
        ingest(paths.root / "runs", ablation_store, parsers=parsers_for(), show_progress=False)
    with Store.open_read_only(database) as ablation_store:
        experiments = ablation_store.load_all()
    results = compare_all(experiments, baseline, None, ComparisonConfig(seed=seed))
    findings = rank(classify_findings(results, RegressionConfig()))

    return AblationResult(
        zero_shot=zero_shot,
        retrieval=retrieval,
        root=root,
        findings=tuple(
            AblationFinding(
                candidate=finding.result.candidate,
                tag=finding.result.tag,
                relative_effect_pct=finding.result.relative_effect_pct,
                p_value=finding.result.p_value,
                adjusted_p=finding.adjusted_p,
                verdict=finding.verdict,
                weak_mode=finding.result.is_weak_mode,
            )
            for finding in findings
        ),
        baseline=baseline,
        seconds=time.perf_counter() - started,
        macro_f1=macro_f1,
        retrieval_key=f"{locale}_{split}",
    )
