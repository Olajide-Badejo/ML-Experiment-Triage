"""`triage llm ask "question"`: retrieval over prose, arithmetic over SQLite.

**The split is the design.** Retrieval runs over the PROSE corpus only, the
Markdown documents, the README and the engineering log, because prose is what
embeddings are for: a question phrased in a reader's words has to find a
paragraph phrased in the author's. Numbers are never retrieved. Every figure in
an answer is computed here, from the database, by a SQL template chosen by
intent, and then the generated answer is checked against exactly those figures
by the same grounding pass the summarizer uses.

Embedding the numeric tables would be the obvious thing and would be wrong twice
over: nearest neighbour search over a table of run metrics answers "which rows
look like this text" when the question was "which conditions regressed", and it
would put a number into the model's context that nothing had computed, which is
how a confident wrong answer gets made.

**This verb is a convenience, and is documented as one.** It cannot answer
anything the analysis layer cannot already answer, it answers less precisely
than the report does, and a question it cannot ground it REFUSES, naming what it
can answer instead. That refusal is the feature: a question answering box that
guesses when it does not know is worse than no box.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from triage.core.store import Store
from triage.llm.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder
from triage.llm.ollama_client import OllamaClient
from triage.llm.summarizer import (
    KIND_COUNT,
    KIND_MEASUREMENT,
    KIND_PERCENT,
    KIND_PROBABILITY,
    ContextEntry,
    ContextTable,
    ground,
)

#: How many prose chunks go into one prompt. Four of about 900 characters is a
#: few hundred tokens, which leaves the 4096 token window mostly to the computed
#: context and the answer.
DEFAULT_K = 4

#: Roughly how long a chunk is, in characters. Chunks are split at paragraph
#: boundaries, so this is a target rather than a limit: a chunk that ends mid
#: sentence retrieves badly and reads worse.
CHUNK_TARGET = 900

#: The documents this reads, relative to the repository root. Prose only: the
#: fixtures, the JSON and the database are data, and a question about them is
#: answered by the SQL side of this module rather than by retrieval.
PROSE_GLOBS = ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "docs/*.md")

SYSTEM_PROMPT = (
    "You answer questions about a machine learning experiment triage run. "
    "You are given documentation extracts and a table of computed figures. "
    "Use ONLY those figures: every number you write is checked against the table "
    "and any sentence containing one that is not there will be deleted. "
    "If the material does not answer the question, say so plainly. "
    "Write short plain prose, no headings and no markdown."
)

_HEADING = re.compile(r"^#{1,6}\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage of prose, with where it came from."""

    source: str
    heading: str
    text: str

    def render(self) -> str:
        where = f"{self.source}" + (f" ({self.heading})" if self.heading else "")
        return f"[{where}]\n{self.text}"


def chunk_markdown(source: str, text: str, target: int = CHUNK_TARGET) -> list[Chunk]:
    """Split one document at headings, then group paragraphs up to `target`.

    Headings first because a Markdown heading is the author's own statement of
    what the following text is about, which makes it both a better boundary than
    a character count and a label worth carrying into the prompt so that an
    answer can say where it came from.
    """
    chunks: list[Chunk] = []
    heading = ""
    paragraphs: list[str] = []

    def flush() -> None:
        if not paragraphs:
            return
        buffer: list[str] = []
        size = 0
        for paragraph in paragraphs:
            if buffer and size + len(paragraph) > target:
                chunks.append(Chunk(source, heading, "\n\n".join(buffer)))
                buffer, size = [], 0
            buffer.append(paragraph)
            size += len(paragraph)
        if buffer:
            chunks.append(Chunk(source, heading, "\n\n".join(buffer)))
        paragraphs.clear()

    for block in text.split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        match = _HEADING.match(stripped.splitlines()[0])
        if match is not None:
            flush()
            heading = match.group(1).strip()
            remainder = "\n".join(stripped.splitlines()[1:]).strip()
            if remainder:
                paragraphs.append(remainder)
            continue
        paragraphs.append(stripped)
    flush()
    return chunks


def collect_prose(root: Path | str, extra: Sequence[Path] = ()) -> list[Chunk]:
    """Every prose chunk under `root`, in a stable order.

    Stable because the order is part of what the index is: two runs of the same
    command over the same tree must retrieve the same chunk for the same
    question, and a directory listing is not sorted on every platform.
    """
    base = Path(root)
    paths: list[Path] = []
    for pattern in PROSE_GLOBS:
        paths.extend(sorted(base.glob(pattern)))
    paths.extend(path for path in extra if path.is_file())

    chunks: list[Chunk] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        relative = path.relative_to(base) if path.is_relative_to(base) else path
        chunks.extend(chunk_markdown(relative.as_posix(), text))
    return chunks


# --------------------------------------------------------------------- intent


@dataclass(frozen=True)
class Intent:
    """One kind of question, and what answering it needs."""

    name: str
    keywords: tuple[str, ...]
    description: str
    needs_baseline: bool = False


#: The questions this can answer. Keyword matching rather than a classifier
#: model, and deliberately: an intent chosen by a second generation would be one
#: more thing that can be confidently wrong, and the failure would be invisible
#: because the answer would still read well.
INTENTS: tuple[Intent, ...] = (
    Intent(
        name="verdicts",
        keywords=("regress", "improve", "verdict", "significant", "better", "worse", "winner"),
        description="which conditions regressed or improved, with their p values",
        needs_baseline=True,
    ),
    Intent(
        name="refusals",
        keywords=("refus", "declin", "not compared", "skipped", "why no"),
        description="which comparisons were refused, and on what grounds",
        needs_baseline=True,
    ),
    Intent(
        name="sensitivity",
        keywords=("sensitiv", "hyperparameter", "learning rate", "batch size", "correlat"),
        description="what the hyperparameter sensitivity table says",
        needs_baseline=True,
    ),
    Intent(
        name="database",
        keywords=("how many", "runs", "database", "metrics", "tags", "conditions", "what is in"),
        description="what the database holds: runs, conditions, metrics and their sizes",
    ),
)


def detect_intent(question: str) -> Intent | None:
    """The intent whose keywords the question uses, or `None`.

    First match in declaration order, and the order is not arbitrary: a question
    mentioning both a regression and the database is a question about the
    regression, so the specific intents are declared ahead of the general one.
    """
    lowered = question.lower()
    for intent in INTENTS:
        if any(keyword in lowered for keyword in intent.keywords):
            return intent
    return None


def refusal_text(question: str, reason: str) -> str:
    """A refusal that names what this CAN answer, per 6.5."""
    answerable = "\n".join(f"  - {intent.description}" for intent in INTENTS)
    return (
        f"I cannot answer that from this database: {reason}\n"
        f"What I can answer:\n{answerable}\n"
        f"For anything else, `triage compare` and `triage report` compute the whole table, "
        f"and this verb is a convenience over them rather than a replacement."
    )


# ------------------------------------------------------------- answer context

#: The SQL this runs, written out rather than built. A template a reader can
#: read is a template a reader can check, and every figure that reaches an
#: answer comes from one of these or from the analysis layer reading the same
#: rows. Nothing numeric is ever retrieved by similarity.
SQL_TEMPLATES: dict[str, str] = {
    "runs": "SELECT COUNT(*) AS n FROM experiments",
    "series": "SELECT COUNT(*) AS n, COALESCE(SUM(n_points), 0) AS points FROM metrics",
    "tags": (
        "SELECT tag, COUNT(*) AS runs, SUM(n_points) AS points "
        "FROM metrics GROUP BY tag ORDER BY tag"
    ),
    "outcomes": "SELECT COUNT(*) AS n, COALESCE(SUM(n_rows), 0) AS rows FROM outcomes",
}


@dataclass(frozen=True)
class AnswerContext:
    """The machine computed half of the prompt: lines to read, numbers to check."""

    lines: tuple[str, ...] = ()
    entries: tuple[ContextEntry, ...] = ()

    def table(self) -> ContextTable:
        return ContextTable(entries=self.entries)

    def render(self) -> str:
        return "\n".join(self.lines)


def database_context(store: Store) -> AnswerContext:
    """Counts and metric names, straight out of the SQL templates above."""
    runs = store.connection.execute(SQL_TEMPLATES["runs"]).fetchone()
    series = store.connection.execute(SQL_TEMPLATES["series"]).fetchone()
    outcomes = store.connection.execute(SQL_TEMPLATES["outcomes"]).fetchone()
    tags = store.connection.execute(SQL_TEMPLATES["tags"]).fetchall()

    lines = [
        f"Runs in the database: {int(runs['n'])}",
        f"Metric series: {int(series['n'])}, holding {int(series['points'])} points",
        f"Cross sectional outcome files: {int(outcomes['n'])}, "
        f"holding {int(outcomes['rows'])} rows",
    ]
    entries = [
        ContextEntry("runs in the database", float(runs["n"]), KIND_COUNT),
        ContextEntry("metric series", float(series["n"]), KIND_COUNT),
        ContextEntry("points", float(series["points"]), KIND_COUNT),
        ContextEntry("outcome files", float(outcomes["n"]), KIND_COUNT),
        ContextEntry("outcome rows", float(outcomes["rows"]), KIND_COUNT),
    ]
    for row in tags:
        tag = str(row["tag"])
        lines.append(
            f"Metric {tag}: logged by {int(row['runs'])} run(s), {int(row['points'])} points"
        )
        entries.append(ContextEntry(f"{tag}: runs logging it", float(row["runs"]), KIND_COUNT))
        entries.append(ContextEntry(f"{tag}: points", float(row["points"]), KIND_COUNT))
    return AnswerContext(lines=tuple(lines), entries=tuple(entries))


def analysis_context(store: Store, intent: Intent, baseline: str) -> AnswerContext:
    """Verdicts, refusals or sensitivity, computed by the analysis layer.

    Computed rather than remembered. A verdict is a function of the runs, the
    gates and the permutation seed, and storing one would create a second place
    for it to be true; running the same `compare_all` the `compare` verb runs
    means this cannot answer with a number the tool itself would not report.
    """
    from triage.analysis.comparison import ComparisonConfig, compare_all
    from triage.analysis.regression import RegressionConfig, classify, rank
    from triage.analysis.sensitivity import analyse

    experiments = store.load_all()
    config = ComparisonConfig()
    results = compare_all(experiments, baseline, None, config)

    lines: list[str] = []
    entries: list[ContextEntry] = []
    if intent.name == "refusals":
        lines.append(f"Comparisons refused against {baseline}: {len(results.refusals)}")
        entries.append(
            ContextEntry("comparisons refused", float(len(results.refusals)), KIND_COUNT)
        )
        lines.extend(f"- {refusal.describe()}" for refusal in results.refusals)
        return AnswerContext(lines=tuple(lines), entries=tuple(entries))

    if intent.name == "sensitivity":
        found = analyse(experiments, None, config)
        lines.append(f"Hyperparameter sensitivity rows: {len(found)}")
        entries.append(ContextEntry("sensitivity rows", float(len(found)), KIND_COUNT))
        for row in found:
            # A correlation over too few conditions has no p value at all, and
            # printing one anyway is the failure this whole verb is written
            # against. It says so instead, and the number never enters the table.
            spelled = "not computable" if row.p_value is None else f"{row.p_value:.4f}"
            lines.append(
                f"- {row.parameter} against {row.tag}: Spearman rho "
                f"{row.correlation:+.4f}, p {spelled}, over {row.n_variants} condition(s)"
            )
            label = f"{row.parameter} vs {row.tag}"
            entries.append(ContextEntry(f"{label}: rho", row.correlation, KIND_MEASUREMENT))
            if row.p_value is not None:
                entries.append(ContextEntry(f"{label}: p", row.p_value, KIND_PROBABILITY))
            entries.append(ContextEntry(f"{label}: conditions", float(row.n_variants), KIND_COUNT))
        return AnswerContext(lines=tuple(lines), entries=tuple(entries))

    findings = rank(classify(results, RegressionConfig()))
    regressions = [finding for finding in findings if finding.is_regression]
    improvements = [finding for finding in findings if finding.is_improvement]
    lines.append(
        f"Against {baseline}: {len(findings)} comparison(s), {len(regressions)} regression(s), "
        f"{len(improvements)} improvement(s)"
    )
    entries.extend(
        [
            ContextEntry("comparisons", float(len(findings)), KIND_COUNT),
            ContextEntry("regressions", float(len(regressions)), KIND_COUNT),
            ContextEntry("improvements", float(len(improvements)), KIND_COUNT),
        ]
    )
    for finding in findings:
        result = finding.result
        lines.append(
            f"- {result.candidate} on {result.tag}: {finding.verdict}, "
            f"{result.relative_effect_pct:+.2f} percent, p {result.p_value:.4f}, "
            f"adjusted p {finding.adjusted_p:.4f}"
        )
        label = f"{result.candidate} on {result.tag}"
        entries.extend(
            [
                ContextEntry(f"{label}: change percent", result.relative_effect_pct, KIND_PERCENT),
                ContextEntry(f"{label}: p", result.p_value, KIND_PROBABILITY),
                ContextEntry(f"{label}: adjusted p", finding.adjusted_p, KIND_PROBABILITY),
            ]
        )
    return AnswerContext(lines=tuple(lines), entries=tuple(entries))


def build_context(store: Store, intent: Intent, baseline: str | None) -> AnswerContext:
    """Dispatch to the SQL or the analysis half, by intent."""
    if intent.needs_baseline:
        if not baseline:
            raise ValueError(
                f"answering {intent.description} needs a baseline to compare against: pass "
                f"--baseline, the same one `triage compare` would be given"
            )
        return analysis_context(store, intent, baseline)
    return database_context(store)


# ------------------------------------------------------------------- the verb


@dataclass(frozen=True)
class AskResult:
    """What was asked, what was retrieved and computed, and what survived."""

    question: str
    text: str
    intent: str = ""
    refused: bool = False
    sources: tuple[str, ...] = ()
    n_dropped: int = 0
    n_context: int = 0
    cached: bool = False
    dropped_sentences: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        if self.refused:
            return "refused"
        where = ", ".join(self.sources) or "no documents"
        return (
            f"intent {self.intent}; retrieved from {where}; {self.n_context} computed figure(s); "
            f"{self.n_dropped} sentence(s) dropped by the grounding pass"
        )


def build_prompt(question: str, chunks: Sequence[Chunk], context: AnswerContext) -> str:
    """The user turn: the question, the prose, and the computed figures."""
    documents = "\n\n".join(chunk.render() for chunk in chunks) or "(no documentation extracts)"
    return (
        f"Question: {question}\n\n"
        f"Documentation extracts:\n{documents}\n\n"
        f"Computed from the database:\n{context.render()}\n\n"
        f"These are the only numbers you may use:\n{context.table().render()}\n\n"
        f"Answer the question in at most six sentences."
    )


def ask(
    client: OllamaClient,
    store: Store,
    question: str,
    root: Path | str = ".",
    baseline: str | None = None,
    k: int = DEFAULT_K,
    model: str | None = None,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
) -> AskResult:
    """Answer one question, or refuse it naming what can be answered instead."""
    intent = detect_intent(question)
    if intent is None:
        return AskResult(
            question=question,
            text=refusal_text(question, "it does not name anything this database records"),
            refused=True,
        )
    try:
        context = build_context(store, intent, baseline)
    except ValueError as error:
        return AskResult(question=question, text=refusal_text(question, str(error)), refused=True)

    chunks = collect_prose(root)
    retrieved: list[Chunk] = []
    if chunks:
        embedder = Embedder(client, store, model=embed_model)
        index = embedder.index([chunk.render() for chunk in chunks])
        query = embedder.embed_query(question)
        retrieved = [chunks[found.index] for found in index.search(query, k=k)]

    chat_model = model or client.chat_model
    prompt = build_prompt(question, retrieved, context)
    digest = client.digest_of(chat_model)

    from triage.llm.annotator import response_cache_key

    key = response_cache_key(chat_model, digest, prompt)
    cached = store.get_response(key)
    if cached is None:
        generated = client.chat(prompt, system=SYSTEM_PROMPT, model=chat_model)
        store.put_response(key, chat_model, digest, generated)
    else:
        generated = cached

    grounded = ground(generated, context.table())
    return AskResult(
        question=question,
        text=grounded.text,
        intent=intent.name,
        sources=tuple(dict.fromkeys(chunk.source for chunk in retrieved)),
        n_dropped=grounded.n_dropped,
        n_context=len(context.entries),
        cached=cached is not None,
        dropped_sentences=grounded.dropped,
    )
