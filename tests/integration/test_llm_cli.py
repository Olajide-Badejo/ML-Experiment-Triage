"""The `triage llm` verbs, driven through `main` with a fake transport injected.

The transport is injected by monkeypatching `urllib_transport`, which is the
one seam that lets a whole command run end to end without a model: everything
above it, the parser, the handlers, the artifacts and the exit codes, is the
real thing. Nothing here opens a socket.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from llm_fakes import (
    TAGS,
    FakeTransport,
    fake_chat_copies_the_nearest_example,
    fake_chat_says,
    fake_embeddings,
)

from triage import cli
from triage.autofill.generator import GeneratorConfig, write_corpus
from triage.core.store import Store
from triage.llm import ollama_client


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    """A local Ollama that is up, with the Section 2 models pulled."""
    fake = FakeTransport(
        {
            "/api/tags": TAGS,
            "/api/embed": fake_embeddings(256),
            "/api/chat": fake_chat_copies_the_nearest_example(),
        }
    )
    monkeypatch.setattr(ollama_client, "urllib_transport", fake)
    return fake


@pytest.fixture
def down(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    """A machine with no Ollama on the port."""
    refused = OSError("connection refused")
    fake = FakeTransport({"/api/tags": refused, "/api/chat": refused, "/api/embed": refused})
    monkeypatch.setattr(ollama_client, "urllib_transport", fake)
    return fake


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    write_corpus(root, GeneratorConfig(n_fields=400, locales=("de_DE",)), seed=0)
    return root


def test_the_llm_verbs_are_in_the_one_help_document() -> None:
    text = cli.build_parser().format_help()

    assert " llm " in text or "llm," in text
    assert "ask" in text


def test_annotate_writes_both_artifact_shapes(
    corpus: Path, tmp_path: Path, service: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "llm_eval"
    code = cli.main(
        [
            "llm", "annotate",
            "--data", str(corpus),
            "--out", str(out),
            "--database", str(tmp_path / "cache.db"),
            "--locale", "de_DE",
            "--k", "4",
            "--limit", "40",
            "--bootstrap", "3",
        ]
    )  # fmt: skip

    assert code == cli.EXIT_OK
    printed = capsys.readouterr().out
    assert "llm (kNN k=4" in printed
    assert (out / "evaluation.json").is_file()
    rows = [
        json.loads(line)
        for line in (out / "outcomes" / "autofill_eval" / "run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["engine"] for row in rows} == {"llm", "rules"}
    report = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert report["engines"]["llm"]["n_rows"] == 40


def test_annotate_twice_asks_the_model_nothing_the_second_time(
    corpus: Path, tmp_path: Path, service: FakeTransport
) -> None:
    arguments = [
        "llm", "annotate",
        "--data", str(corpus),
        "--out", str(tmp_path / "llm_eval"),
        "--database", str(tmp_path / "cache.db"),
        "--k", "4", "--limit", "40", "--bootstrap", "2",
    ]  # fmt: skip

    assert cli.main(arguments) == cli.EXIT_OK
    after_first = len(service.bodies("/api/chat"))
    assert cli.main(arguments) == cli.EXIT_OK

    assert after_first == 40
    assert len(service.bodies("/api/chat")) == after_first


def test_the_ablation_prints_a_verdict_with_a_p_value(
    corpus: Path, tmp_path: Path, service: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "llm", "annotate",
            "--data", str(corpus),
            "--out", str(tmp_path / "ablation"),
            "--database", str(tmp_path / "cache.db"),
            "--ablation", "--k", "4", "--limit", "60", "--bootstrap", "3",
        ]
    )  # fmt: skip

    assert code == cli.EXIT_OK
    printed = capsys.readouterr().out
    assert "ablation:" in printed
    assert "zero shot" in printed
    assert " at p " in printed
    assert (tmp_path / "ablation" / "ablation.db").is_file()


def test_a_service_that_is_down_names_the_remedy_and_exits_two(
    corpus: Path, tmp_path: Path, down: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "llm", "annotate",
            "--data", str(corpus),
            "--out", str(tmp_path / "llm_eval"),
            "--database", str(tmp_path / "cache.db"),
        ]
    )  # fmt: skip

    assert code == cli.EXIT_USAGE
    assert "ollama serve" in capsys.readouterr().err


def test_the_autofill_llm_policy_scores_the_split_instead_of_refusing(
    corpus: Path, tmp_path: Path, service: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    """Part 09 left this verb declaring the policy and refusing it; it works now."""
    out = tmp_path / "policy_eval"
    code = cli.main(
        [
            "autofill", "evaluate",
            "--data", str(corpus),
            "--out", str(out),
            "--policy", "llm",
            "--database", str(tmp_path / "cache.db"),
            "--locale", "de_DE",
            "--llm-k", "4",
            "--llm-limit", "40",
            "--bootstrap", "3",
        ]
    )  # fmt: skip

    assert code == cli.EXIT_OK
    printed = capsys.readouterr().out
    assert "llm" in printed
    report = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert sorted(report["engines"]) == ["llm", "rules"]


def test_the_autofill_llm_policy_still_refuses_when_ollama_is_down(
    corpus: Path, tmp_path: Path, down: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "autofill", "evaluate",
            "--data", str(corpus),
            "--out", str(tmp_path / "policy_eval"),
            "--policy", "llm",
            "--database", str(tmp_path / "cache.db"),
        ]
    )  # fmt: skip

    assert code == cli.EXIT_USAGE
    assert "ollama serve" in capsys.readouterr().err


@pytest.fixture(scope="module")
def ingested(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real database, made the way a user would make one.

    Module scoped because synthesising and ingesting the demo sweep is by far
    the slowest thing in this file and none of the three verbs that read it
    write to it.
    """
    from examples.make_synthetic_runs import generate

    root = tmp_path_factory.mktemp("llm_cli")
    generate(root / "demo_sweep", show_progress=False)
    database = root / "triage.db"
    assert (
        cli.main(["ingest", str(root / "demo_sweep"), "--database", str(database), "--quiet"])
        == cli.EXIT_OK
    )
    return database


@pytest.fixture
def database(ingested: Path, tmp_path: Path) -> Path:
    """A private copy of the module database, per test.

    The response cache lives in the same file as the runs, which is the design;
    the consequence for a test file is that two tests sharing a database share a
    cache, and one of them then measures the other's HTTP calls. A copy per test
    keeps each one measuring itself.
    """
    private = tmp_path / "triage.db"
    for suffix in ("", "-wal", "-shm"):
        source = ingested.with_name(ingested.name + suffix)
        if source.is_file():
            shutil.copy(source, private.with_name(private.name + suffix))
    return private


def test_summarize_grounds_the_generated_text(
    database: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeTransport(
        {
            "/api/tags": TAGS,
            "/api/chat": fake_chat_says(
                "The sweep was compared against the baseline. "
                "One condition regressed by 999.9 percent, which is enormous. "
                "The tool reports the rest as unchanged."
            ),
        }
    )
    monkeypatch.setattr(ollama_client, "urllib_transport", fake)

    code = cli.main(
        ["llm", "summarize", "--database", str(database), "--baseline", "lr0.0010_bs32"]
    )

    assert code == cli.EXIT_OK
    printed = capsys.readouterr().out
    assert "Automated summary (local LLM:" in printed
    assert "999.9" not in printed
    assert "sentence(s) were deleted" in printed


def test_ask_refuses_a_question_it_cannot_ground(
    database: Path, service: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["llm", "ask", "what is the capital of Germany?", "--database", str(database)])

    assert code == cli.EXIT_USAGE
    assert "What I can answer" in capsys.readouterr().out


def test_the_bare_ask_verb_is_the_same_verb(
    database: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Section 6.5 spells it `triage ask`; Section 5.5 spells it `triage llm ask`."""
    fake = FakeTransport(
        {
            "/api/tags": TAGS,
            "/api/embed": fake_embeddings(256),
            "/api/chat": fake_chat_says("The database holds runs of a synthetic sweep."),
        }
    )
    monkeypatch.setattr(ollama_client, "urllib_transport", fake)

    code = cli.main(["ask", "what is in the database?", "--database", str(database)])

    assert code == cli.EXIT_OK
    assert "intent database" in capsys.readouterr().out


def test_ask_names_the_missing_database_rather_than_creating_one(
    tmp_path: Path, service: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "typo.db"
    code = cli.main(["llm", "ask", "how many runs?", "--database", str(missing)])

    assert code == cli.EXIT_USAGE
    assert "no such database" in capsys.readouterr().err
    assert not missing.exists()


def test_the_annotation_cache_lives_in_the_database_the_user_named(
    corpus: Path, tmp_path: Path, service: FakeTransport
) -> None:
    database = tmp_path / "cache.db"
    cli.main(
        [
            "llm", "annotate",
            "--data", str(corpus),
            "--out", str(tmp_path / "llm_eval"),
            "--database", str(database),
            "--k", "4", "--limit", "40", "--bootstrap", "2",
        ]
    )  # fmt: skip

    with Store(database) as store:
        counts = store.cache_counts()
    assert counts["responses"] == 40
    assert counts["embeddings"] > 0


# --------------------------------------------- the summary block in the report


SUMMARY_TEXT = (
    "The sweep was compared against the baseline condition. "
    "One condition improved by 999.9 percent, which no measurement supports. "
    "The tool reports the remaining comparisons with their gates."
)


def report_transport() -> FakeTransport:
    return FakeTransport({"/api/tags": TAGS, "/api/chat": fake_chat_says(SUMMARY_TEXT)})


def test_the_default_report_is_byte_identical_to_what_it_always_was(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The determinism invariant: no flag, no generated text, no difference."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1735689600")
    first = tmp_path / "a.html"
    second = tmp_path / "b.html"
    for output in (first, second):
        arguments = [
            "report", "--database", str(database),
            "--baseline", "lr0.0010_bs32",
            "--output", str(output), "--quiet",
        ]  # fmt: skip
        assert cli.main(arguments) == cli.EXIT_OK

    assert first.read_bytes() == second.read_bytes()
    assert b"Automated summary" not in first.read_bytes()


def test_the_flag_embeds_the_grounded_summary_under_the_specified_heading(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1735689600")
    fake = report_transport()
    monkeypatch.setattr(ollama_client, "urllib_transport", fake)
    output = tmp_path / "with_summary.html"

    code = cli.main(
        [
            "report", "--database", str(database),
            "--baseline", "lr0.0010_bs32",
            "--output", str(output), "--llm-summary", "--quiet",
        ]
    )  # fmt: skip

    assert code == cli.EXIT_OK
    page = output.read_text(encoding="utf-8")
    assert "Automated summary (local LLM: mistral-nemo:12b-instruct-2407-q4_K_M)" in page
    assert "999.9" not in page, "the grounding pass runs before the page is written"
    assert "compared against the baseline condition" in page
    assert "sentence(s) were deleted" in page


def test_the_flagged_report_is_deterministic_through_the_cache(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1735689600")
    fake = report_transport()
    monkeypatch.setattr(ollama_client, "urllib_transport", fake)
    first = tmp_path / "one.html"
    second = tmp_path / "two.html"
    for output in (first, second):
        cli.main(
            [
                "report", "--database", str(database),
                "--baseline", "lr0.0010_bs32",
                "--output", str(output), "--llm-summary", "--quiet",
            ]
        )  # fmt: skip

    assert first.read_bytes() == second.read_bytes()
    assert len(fake.bodies("/api/chat")) == 1, "the second build reads the cache"


def test_a_model_that_is_not_running_costs_the_summary_and_not_the_report(
    database: Path, tmp_path: Path, down: FakeTransport, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "no_summary.html"
    code = cli.main(
        [
            "report", "--database", str(database),
            "--baseline", "lr0.0010_bs32",
            "--output", str(output), "--llm-summary", "--quiet",
        ]
    )  # fmt: skip

    assert code == cli.EXIT_OK
    assert output.is_file()
    assert "no automated summary" in capsys.readouterr().err
    assert "Automated summary" not in output.read_text(encoding="utf-8")
