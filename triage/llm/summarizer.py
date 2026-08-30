"""A plain language summary of a triage report, with every number checked.

**The grounding pass, and where the idea comes from.** The pattern implemented
here is adopted from the PyTorch Performance Toolkit's `tpt.triage` explainer,
with credit (E7d): generate freely, then extract every number from the generated
text and verify each one against a Python computed table of the numbers the
report actually holds, and DELETE any sentence containing a number the table
does not support, recording how many sentences were dropped. The two projects
share the idea and no code; a shared client library across the three
repositories is explicitly out of scope.

**Why deletion rather than a warning.** A summary is read by somebody who is not
going to check it, which is the entire reason it exists. A hallucinated p value
with a caveat attached is still a hallucinated p value in a document this
project's whole argument says you should be able to trust. Deleting the sentence
costs a sentence; leaving it costs the report. The drop count is reported beside
the summary so that a reader can see when the model was inventing, rather than
being told a clean story about a pass that removed half the text.

**No retrieval.** A few dozen findings fit in a 4096 token context whole, so the
prompt carries the entire report. Retrieving parts of it would add exactly one
failure mode that this cannot otherwise have: a silent omission, where the
summary is confidently wrong because the regression it should have mentioned was
not in the chunks that came back.

**Rounding tolerance, both ways.** A model writes 4.53 for 4.5312 and writes 5
percent for 0.05, and neither is a fabrication. So a number in the text is
supported when it matches a table entry to the precision it was WRITTEN at, and
the table registers the percent and fraction spellings of each entry. What it
will not do is accept a number that is merely close: 0.041 does not support
0.049, because two p values on either side of a gate are exactly the pair a
reader must not have confused for them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triage.analysis.regression import TriageReport
from triage.core.store import Store
from triage.llm.annotator import response_cache_key
from triage.llm.ollama_client import OllamaClient

#: The length the prompt asks for. A summary shorter than this says nothing a
#: verdict table does not; one longer stops being a summary.
MIN_WORDS = 150
MAX_WORDS = 250

#: Relative slack on a match, on top of the tolerance implied by how the number
#: was written. Small on purpose: it exists for float formatting, not to let a
#: nearby number through.
RELATIVE_TOLERANCE = 1e-9

SYSTEM_PROMPT = (
    "You summarise the output of a statistical experiment triage tool for a reader "
    "who has not seen the table. You are precise about uncertainty: a comparison "
    "that did not clear a gate did not clear it, and a refusal is a result. "
    "You never state a number that is not in the material you were given. "
    "You write plain prose in complete sentences, with no headings, no lists and "
    "no markdown."
)

#: A number as written in prose: an optional sign, digits with optional
#: thousands separators, an optional decimal part, an optional percent sign.
#: Exponents are deliberately absent: nothing in a report of this kind is
#: written that way, and a pattern that accepted them would also swallow the `e`
#: of a following word.
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")

#: Sentence boundary: a full stop, question mark or exclamation mark followed by
#: whitespace and something that starts a sentence. The lookbehind excludes a
#: digit before the stop, so `0.05.` at the end of a sentence and `p = 0.05` in
#: the middle of one are both left intact.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass(frozen=True)
class Number:
    """One number as it appears in the generated text."""

    text: str
    value: float
    decimals: int

    @property
    def tolerance(self) -> float:
        """Half a unit in the last place the number was written to."""
        return 0.5 * (10.0**-self.decimals)


def extract_numbers(text: str) -> list[Number]:
    """Every number in `text`, with the precision each was written at."""
    found: list[Number] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        cleaned = raw.rstrip("%").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:  # pragma: no cover - the pattern cannot produce one
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        # A percent sign in the text means the value is a percentage, and the
        # table holds percentages as percentages, so it is left as written.
        found.append(Number(text=raw, value=value, decimals=decimals))
    return found


def split_sentences(text: str) -> list[str]:
    """The text as sentences, keeping every character that was in it."""
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in _SENTENCE.split(stripped) if part.strip()]


#: What kind of quantity an entry is, which decides the spellings it licenses.
#: Declared per entry rather than guessed from the magnitude, because guessing
#: is how a count of 1 comes to license the number 100.
KIND_COUNT = "count"
KIND_PROBABILITY = "probability"
KIND_PERCENT = "percent"
KIND_MEASUREMENT = "measurement"


@dataclass(frozen=True)
class ContextEntry:
    """One checkable number, and the spellings of it a writer may use."""

    label: str
    value: float
    kind: str = KIND_MEASUREMENT

    def spellings(self) -> tuple[float, ...]:
        """The forms of this number that mean the same fact about the report.

        A probability may honestly be written as a percentage (`p 0.0079` as
        "0.79 percent") and a percentage as a fraction, so those two kinds
        license the second spelling. A count and a measurement license nothing
        but themselves: "three regressions" is not "300" of anything, and a
        macro F1 of 0.9643 rewritten as 96.43 would be a different quantity.
        """
        if self.kind == KIND_PROBABILITY:
            return (self.value, self.value * 100.0)
        if self.kind == KIND_PERCENT:
            return (self.value, self.value / 100.0)
        return (self.value,)


@dataclass(frozen=True)
class ContextTable:
    """The numbers the report actually holds, and the only ones allowed through.

    The labels are in the prompt as well as in the check: a model given the
    table in the same words it will be checked against has little reason to
    invent, which makes the grounding pass a safety net rather than the main
    mechanism.
    """

    entries: tuple[ContextEntry, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def values(self) -> dict[str, float]:
        return {entry.label: entry.value for entry in self.entries}

    def supports(self, number: Number) -> bool:
        """True when some entry rounds to the number as it was written."""
        for entry in self.entries:
            for candidate in entry.spellings():
                slack = number.tolerance + RELATIVE_TOLERANCE * abs(candidate)
                if abs(candidate - number.value) <= slack:
                    return True
        return False

    def render(self) -> str:
        """The table as the prompt shows it."""
        return "\n".join(f"- {entry.label}: {entry.value:g}" for entry in self.entries)


def build_context_table(
    report: TriageReport, extra: Mapping[str, float] | None = None
) -> ContextTable:
    """Every number a summary of this report is allowed to contain.

    Built from the report object rather than from the rendered table, because
    the rendered table is already rounded and a summary checked against rounded
    numbers could not quote an unrounded one. Small counts are included
    explicitly: a sentence saying "three conditions regressed" is a claim about
    the report and has to be checkable like any other.
    """
    entries: list[ContextEntry] = [
        ContextEntry("comparisons", float(len(report.findings)), KIND_COUNT),
        ContextEntry("regressions", float(len(report.regressions)), KIND_COUNT),
        ContextEntry("improvements", float(len(report.improvements)), KIND_COUNT),
        ContextEntry("inadmissible comparisons", float(len(report.inadmissible)), KIND_COUNT),
        ContextEntry("significance gate (alpha)", report.config.alpha, KIND_PROBABILITY),
        ContextEntry("false discovery rate", report.config.false_discovery_rate, KIND_PROBABILITY),
    ]
    if report.config.practical_threshold_pct is not None:
        entries.append(
            ContextEntry(
                "practical threshold, percent",
                report.config.practical_threshold_pct,
                KIND_PERCENT,
            )
        )
    if report.config.practical_threshold_absolute is not None:
        entries.append(
            ContextEntry(
                "practical threshold, absolute",
                report.config.practical_threshold_absolute,
                KIND_MEASUREMENT,
            )
        )
    for finding in report.findings:
        result = finding.result
        prefix = f"{result.candidate} on {result.tag}"
        entries.extend(
            [
                ContextEntry(f"{prefix}: change percent", result.relative_effect_pct, KIND_PERCENT),
                ContextEntry(f"{prefix}: p", result.p_value, KIND_PROBABILITY),
                ContextEntry(f"{prefix}: adjusted p", finding.adjusted_p, KIND_PROBABILITY),
                ContextEntry(f"{prefix}: baseline", result.baseline_statistic, KIND_MEASUREMENT),
                ContextEntry(f"{prefix}: candidate", result.candidate_statistic, KIND_MEASUREMENT),
                ContextEntry(f"{prefix}: effect", result.effect, KIND_MEASUREMENT),
                ContextEntry(
                    f"{prefix}: runs compared",
                    float(result.n_baseline + result.n_candidate),
                    KIND_COUNT,
                ),
            ]
        )
    entries.extend(
        ContextEntry(label, float(value), KIND_MEASUREMENT)
        for label, value in (extra or {}).items()
    )
    return ContextTable(entries=tuple(entries))


@dataclass(frozen=True)
class GroundedText:
    """What survived the grounding pass, and what did not."""

    text: str
    kept: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def describe(self) -> str:
        if not self.dropped:
            return f"{self.word_count} words, every number checked against the report"
        return (
            f"{self.word_count} words; {self.n_dropped} sentence(s) dropped for containing "
            f"numbers the report does not support ({', '.join(sorted(set(self.unsupported)))})"
        )


def ground(text: str, table: ContextTable) -> GroundedText:
    """Delete every sentence carrying a number the table does not support."""
    kept: list[str] = []
    dropped: list[str] = []
    unsupported: list[str] = []
    for sentence in split_sentences(text):
        bad = [number.text for number in extract_numbers(sentence) if not table.supports(number)]
        if bad:
            dropped.append(sentence)
            unsupported.extend(bad)
        else:
            kept.append(sentence)
    return GroundedText(
        text=" ".join(kept),
        kept=tuple(kept),
        dropped=tuple(dropped),
        unsupported=tuple(unsupported),
    )


def render_report(report: TriageReport, refusals: Sequence[str] = ()) -> str:
    """The whole report as the prompt sees it: verdicts, refusals, headlines.

    Whole, and in one place: 6.4 puts the entire `TriageReport` in the prompt,
    and this is the rendering, so that what the model was shown is one function
    a reader can read rather than a prompt assembled across three call sites.
    """
    lines = [f"Baseline: {report.baseline}", report.summary(), report.family_note(), ""]
    if report.findings:
        lines.append("Findings, most severe first:")
        for finding in report.findings:
            result = finding.result
            mode = " (weak mode: one run per condition)" if result.is_weak_mode else ""
            lines.append(
                f"- {result.candidate} on {result.tag}: {finding.verdict}, "
                f"{result.relative_effect_pct:+.2f} percent "
                f"({result.baseline_statistic:g} to {result.candidate_statistic:g}), "
                f"p {result.p_value:.4f}, adjusted p {finding.adjusted_p:.4f}, "
                f"{result.mode_label}{mode}"
            )
    else:
        lines.append("No comparison in this database could be made against the baseline.")
    if refusals:
        lines.append("")
        lines.append("Comparisons that were refused:")
        lines.extend(f"- {refusal}" for refusal in refusals)
    best = report.best_finding
    if best is not None:
        lines.append("")
        lines.append(
            f"Largest significant improvement: {best.candidate} on {best.tag}, "
            f"{best.result.signed_improvement_pct:+.2f} percent"
        )
    return "\n".join(lines)


def build_prompt(report: TriageReport, table: ContextTable, refusals: Sequence[str] = ()) -> str:
    """The user turn: the report, the checkable numbers, and the length rule."""
    return (
        f"Here is the output of an experiment triage run.\n\n"
        f"{render_report(report, refusals)}\n\n"
        f"These are the only numbers you may use, and you must write each one exactly "
        f"as it is here:\n{table.render()}\n\n"
        f"Write a {MIN_WORDS} to {MAX_WORDS} word plain language summary for a reader who "
        f"has not seen this table. Say what changed, how confident the tool is, and what "
        f"it declined to conclude. Any sentence containing a number that is not in the "
        f"list above will be deleted before anyone reads it."
    )


@dataclass(frozen=True)
class LlmSummary:
    """A grounded summary, ready to be embedded in the HTML report."""

    model: str
    text: str
    n_dropped: int
    n_context: int
    word_count: int
    cached: bool = False
    dropped_sentences: tuple[str, ...] = ()

    @property
    def heading(self) -> str:
        """The heading 6.4 specifies, with the model that wrote it named."""
        return f"Automated summary (local LLM: {self.model})"

    def caption(self) -> str:
        checked = f"{self.n_context} numbers from this report"
        if not self.n_dropped:
            return (
                f"Generated locally by {self.model} at temperature 0, then checked against "
                f"{checked}. No sentence was dropped."
            )
        return (
            f"Generated locally by {self.model} at temperature 0, then checked against "
            f"{checked}. {self.n_dropped} sentence(s) were deleted for containing a number "
            f"this report does not support."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "text": self.text,
            "n_dropped": self.n_dropped,
            "n_context": self.n_context,
            "word_count": self.word_count,
            "cached": self.cached,
            "dropped_sentences": list(self.dropped_sentences),
        }


def summarise_report(
    client: OllamaClient,
    store: Store,
    report: TriageReport,
    refusals: Sequence[str] = (),
    extra_context: Mapping[str, float] | None = None,
    model: str | None = None,
) -> LlmSummary:
    """Generate, ground, and return the summary, going through the cache.

    The cache is what makes `--llm-summary` compatible with byte determinism.
    Without the flag nothing here runs and the report is unchanged; with it, the
    same database and the same model produce the same prompt, which produces the
    same cached response, which grounds to the same text.
    """
    chat_model = model or client.chat_model
    table = build_context_table(report, extra_context)
    prompt = build_prompt(report, table, refusals)
    digest = client.digest_of(chat_model)
    key = response_cache_key(chat_model, digest, prompt)

    cached = store.get_response(key)
    if cached is None:
        generated = client.chat(prompt, system=SYSTEM_PROMPT, model=chat_model)
        store.put_response(key, chat_model, digest, generated)
    else:
        generated = cached

    grounded = ground(generated, table)
    return LlmSummary(
        model=chat_model,
        text=grounded.text,
        n_dropped=grounded.n_dropped,
        n_context=len(table),
        word_count=grounded.word_count,
        cached=cached is not None,
        dropped_sentences=grounded.dropped,
    )
