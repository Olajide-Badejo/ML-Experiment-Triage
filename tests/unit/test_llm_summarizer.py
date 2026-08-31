"""Grounded generation: the model writes, and then the numbers are checked.

The test this file exists for is the poisoned one. A fake transport returns a
summary containing a p value that is not in the report, and the pass must delete
the sentence carrying it and say that it did. Everything else here is about the
edges of that: a number written to fewer decimals is the same number, a number
written as a percentage of a probability is the same number, and a number that
is merely nearby is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from llm_fakes import TAGS, FakeTransport, fake_chat_says

from triage.analysis.comparison import ComparisonResult
from triage.analysis.regression import Finding, RegressionConfig, TriageReport
from triage.core.store import Store
from triage.llm.ollama_client import DEFAULT_CHAT_MODEL, OllamaClient
from triage.llm.summarizer import (
    KIND_COUNT,
    KIND_PERCENT,
    KIND_PROBABILITY,
    MAX_WORDS,
    MIN_WORDS,
    ContextEntry,
    ContextTable,
    build_context_table,
    build_prompt,
    extract_numbers,
    ground,
    render_report,
    split_sentences,
    summarise_report,
)


def result(candidate: str = "lr0.0100_bs32", tag: str = "val/accuracy") -> ComparisonResult:
    return ComparisonResult(
        tag=tag,
        baseline="lr0.0010_bs32",
        candidate=candidate,
        mode="seed_replicated",
        test_name="paired permutation",
        baseline_statistic=0.8123,
        candidate_statistic=0.8654,
        effect=0.0531,
        relative_effect_pct=6.5371,
        effect_size=1.42,
        effect_size_name="Cohen's d",
        ci_low=0.02,
        ci_high=0.09,
        ci_method="bootstrap",
        ci_level=0.95,
        p_value=0.0079,
        n_permutations=10000,
        exact=False,
        min_attainable_p=0.0002,
        n_baseline=5,
        n_candidate=5,
        window_points=40,
        higher_is_better=True,
        seed=0,
    )


def report(n: int = 1) -> TriageReport:
    findings = [
        Finding(
            result=result(candidate=f"condition_{index}"),
            adjusted_p=0.0158,
            verdict="improvement",
            severity=1.0,
            passes_statistical_gate=True,
            passes_practical_gate=True,
            family="seed_replicated",
        )
        for index in range(n)
    ]
    return TriageReport(findings=findings, config=RegressionConfig(), baseline="lr0.0010_bs32")


# ------------------------------------------------------------------ scanning


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p was 0.0079 here", [0.0079]),
        ("a 6.54 percent gain", [6.54]),
        ("a gain of 6.54%", [6.54]),
        ("a change of -1.5 against +2", [-1.5, 2.0]),
        ("10,000 permutations", [10000.0]),
        ("no numbers at all", []),
    ],
)
def test_every_number_in_prose_is_found(text: str, expected: list[float]) -> None:
    assert [number.value for number in extract_numbers(text)] == expected


def test_a_number_carries_the_precision_it_was_written_at() -> None:
    numbers = {number.text: number for number in extract_numbers("0.0079 and 0.008 and 8")}

    assert numbers["0.0079"].decimals == 4
    assert numbers["0.008"].tolerance == pytest.approx(0.0005)
    assert numbers["8"].tolerance == pytest.approx(0.5)


def test_a_decimal_point_does_not_end_a_sentence() -> None:
    text = "The p value was 0.05 for the first. The second was worse."
    assert split_sentences(text) == [
        "The p value was 0.05 for the first.",
        "The second was worse.",
    ]


def test_an_empty_generation_splits_into_nothing() -> None:
    assert split_sentences("   ") == []


# ------------------------------------------------------------------ grounding


def table(**values: float) -> ContextTable:
    return ContextTable(
        entries=tuple(ContextEntry(label, value) for label, value in values.items())
    )


def test_a_number_written_to_fewer_decimals_is_supported() -> None:
    entries = table(p=0.007912)
    assert entries.supports(extract_numbers("0.0079")[0])
    assert entries.supports(extract_numbers("0.008")[0])


def test_a_number_that_is_merely_nearby_is_not_supported() -> None:
    """0.041 and 0.049 sit on either side of a gate and are not each other."""
    entries = table(p=0.049)
    assert not entries.supports(extract_numbers("0.041")[0])


def test_a_probability_may_be_written_as_a_percentage() -> None:
    entries = ContextTable(entries=(ContextEntry("p", 0.0079, KIND_PROBABILITY),))
    assert entries.supports(extract_numbers("0.79 percent")[0])


def test_a_percentage_may_be_written_as_a_fraction() -> None:
    entries = ContextTable(entries=(ContextEntry("change", 6.54, KIND_PERCENT),))
    assert entries.supports(extract_numbers("0.0654")[0])


def test_a_count_licenses_only_itself() -> None:
    """Guessing spellings from magnitude is how a count of 1 licenses 100."""
    entries = ContextTable(entries=(ContextEntry("regressions", 1.0, KIND_COUNT),))

    assert entries.supports(extract_numbers("1")[0])
    assert not entries.supports(extract_numbers("100")[0])


def test_a_sentence_with_an_unsupported_number_is_deleted_and_counted() -> None:
    text = (
        "The accuracy improved by 6.54 percent. "
        "The change was significant at p 0.0031. "
        "Two conditions were compared."
    )
    grounded = ground(text, table(change=6.5371, comparisons=2.0))

    assert grounded.n_dropped == 1
    assert "0.0031" not in grounded.text
    assert "6.54 percent" in grounded.text
    assert "Two conditions were compared." in grounded.text
    assert grounded.unsupported == ("0.0031",)


def test_a_sentence_with_no_numbers_at_all_always_survives() -> None:
    grounded = ground("The tool declined to compare two of the conditions.", table())

    assert grounded.n_dropped == 0
    assert grounded.text.startswith("The tool declined")


def test_the_drop_is_described_rather_than_hidden() -> None:
    grounded = ground("A gain of 99.9 percent was measured.", table(change=1.0))

    assert grounded.text == ""
    assert "1 sentence(s) dropped" in grounded.describe()
    assert "99.9" in grounded.describe()


def test_a_clean_pass_says_so() -> None:
    assert "every number checked" in ground("Nothing changed.", table()).describe()


# ----------------------------------------------------------------- the table


def test_the_table_holds_the_reports_own_numbers() -> None:
    entries = build_context_table(report())
    values = entries.values

    assert values["comparisons"] == 1.0
    assert values["improvements"] == 1.0
    assert values["condition_0 on val/accuracy: p"] == pytest.approx(0.0079)
    assert values["condition_0 on val/accuracy: adjusted p"] == pytest.approx(0.0158)
    assert values["condition_0 on val/accuracy: change percent"] == pytest.approx(6.5371)
    assert values["condition_0 on val/accuracy: runs compared"] == 10.0


def test_extra_context_joins_the_table() -> None:
    entries = build_context_table(report(), extra={"autofill macro F1": 0.9643})
    assert entries.values["autofill macro F1"] == pytest.approx(0.9643)


def test_the_prompt_carries_the_whole_report_the_table_and_the_length_rule() -> None:
    entries = build_context_table(report(2))
    prompt = build_prompt(report(2), entries, refusals=["short_run on val/loss: 3 points"])

    assert "condition_0 on val/accuracy" in prompt
    assert "condition_1 on val/accuracy" in prompt
    assert "short_run on val/loss" in prompt
    assert f"{MIN_WORDS} to {MAX_WORDS} word" in prompt
    assert "will be deleted" in prompt


def test_an_empty_report_still_renders_something_honest() -> None:
    rendered = render_report(TriageReport(findings=[], baseline="base"))
    assert "No comparison" in rendered


# ------------------------------------------------------------- the whole pass


HONEST = (
    "The sweep compared one condition against the baseline. "
    "Accuracy rose by 6.54 percent, from 0.8123 to 0.8654. "
    "The permutation test put the p value at 0.0079, and 0.0158 after correction. "
    "The tool therefore calls this an improvement."
)

POISONED = (
    "The sweep compared one condition against the baseline. "
    "Accuracy rose by 43.2 percent, which is a large effect. "
    "The p value was 0.0002 and the confidence interval excluded zero. "
    "The tool therefore calls this an improvement."
)


def summarise(store: Store, transport: FakeTransport) -> object:
    return summarise_report(OllamaClient(transport=transport), store, report())


def test_an_honest_summary_survives_the_grounding_pass(temp_database: Path) -> None:
    transport = FakeTransport({"/api/tags": TAGS, "/api/chat": fake_chat_says(HONEST)})
    with Store(temp_database) as store:
        summary = summarise(store, transport)

    assert summary.n_dropped == 0  # type: ignore[attr-defined]
    assert "6.54 percent" in summary.text  # type: ignore[attr-defined]
    assert summary.model == DEFAULT_CHAT_MODEL  # type: ignore[attr-defined]
    assert "No sentence was dropped" in summary.caption()  # type: ignore[attr-defined]


def test_a_poisoned_summary_loses_exactly_the_fabricated_sentences(
    temp_database: Path,
) -> None:
    """The test 6.4 is for: invented numbers do not reach the reader."""
    transport = FakeTransport({"/api/tags": TAGS, "/api/chat": fake_chat_says(POISONED)})
    with Store(temp_database) as store:
        summary = summarise(store, transport)

    assert summary.n_dropped == 2  # type: ignore[attr-defined]
    assert "43.2" not in summary.text  # type: ignore[attr-defined]
    assert "0.0002" not in summary.text  # type: ignore[attr-defined]
    assert "calls this an improvement" in summary.text  # type: ignore[attr-defined]
    assert "2 sentence(s) were deleted" in summary.caption()  # type: ignore[attr-defined]


def test_the_dropped_sentences_are_kept_for_the_reader_who_wants_them(
    temp_database: Path,
) -> None:
    transport = FakeTransport({"/api/tags": TAGS, "/api/chat": fake_chat_says(POISONED)})
    with Store(temp_database) as store:
        summary = summarise(store, transport)

    assert any("43.2" in sentence for sentence in summary.dropped_sentences)  # type: ignore[attr-defined]


def test_the_summary_is_cached_so_a_second_report_asks_nothing(
    temp_database: Path,
) -> None:
    transport = FakeTransport({"/api/tags": TAGS, "/api/chat": fake_chat_says(HONEST)})
    with Store(temp_database) as store:
        first = summarise(store, transport)
    with Store(temp_database) as store:
        second = summarise(store, transport)

    assert len(transport.bodies("/api/chat")) == 1
    assert second.cached is True  # type: ignore[attr-defined]
    assert first.text == second.text  # type: ignore[attr-defined]
    assert first.cached is False  # type: ignore[attr-defined]


def test_the_heading_names_the_model_that_wrote_it(temp_database: Path) -> None:
    transport = FakeTransport({"/api/tags": TAGS, "/api/chat": fake_chat_says(HONEST)})
    with Store(temp_database) as store:
        summary = summarise(store, transport)

    assert summary.heading == f"Automated summary (local LLM: {DEFAULT_CHAT_MODEL})"  # type: ignore[attr-defined]
