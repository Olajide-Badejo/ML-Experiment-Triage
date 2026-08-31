"""The half of the agentic demo that needs a real browser.

Marked `browser` and skipped when no Chromium is installed, which is the same
arrangement the Ollama tests use and for the same reason: a machine without the
program is not a machine with a defect. `pytest -m browser` runs these
deliberately.

What is FOR here is everything a dictionary cannot stand in for. The live DOM
has to report the same fields the parser reports, a value set through
`Runtime.evaluate` has to actually be in the element afterwards, and the score
the browser run produces has to be the score the CI run produces. The last one
is the contract that makes `--no-browser` worth having: if the two ever
disagree, the cheap mode is measuring something else.

Nothing here leaves localhost. The pages are `file://` URLs written by the
generator into a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage.autofill import agentic
from triage.autofill.evaluate import ScoredEngine, heuristic_predict
from triage.autofill.generator import FieldRecord, GeneratorConfig, write_corpus
from triage.autofill.policy import RewardModel, always_fill

pytestmark = pytest.mark.browser

CONFIG = GeneratorConfig(n_fields=400, locales=("en_US", "de_DE"))

#: The committed README screenshot is about this size; anything far above it is
#: the kind of binary churn D35a is about.
SCREENSHOT_CEILING_BYTES = 400_000


@pytest.fixture(scope="module", autouse=True)
def _needs_chrome() -> None:
    if not agentic.chrome_available():
        pytest.skip("no Chromium on this machine: `choreo_get_chrome` installs one")


@pytest.fixture(scope="module")
def pages(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = tmp_path_factory.mktemp("agentic_browser")
    paths = write_corpus(root / "data", CONFIG, seed=0).pages
    assert len(paths) == 6, "three templates in two locales"
    return paths


def heuristic(records: list[FieldRecord]) -> ScoredEngine:
    predicted, confidence, latency = heuristic_predict(records)
    return ScoredEngine(
        engine="rules", predicted=predicted, confidence=confidence, latency_us=latency
    )


def test_the_live_dom_reports_the_same_fields_as_the_parser(pages: list[Path]) -> None:
    """The two extractors, over every page, attribute by attribute."""
    with agentic.ChromeSession() as session:
        for path in pages:
            live = session.page(path).source
            static = agentic.StaticPage.from_path(path).source
            assert live.locale == static.locale
            assert live.template == static.template
            assert len(live.rows) == len(static.rows)
            for from_dom, from_markup in zip(live.rows, static.rows, strict=True):
                for key in agentic.ROW_KEYS:
                    assert from_dom[key] == from_markup[key], f"{path.stem}: {key}"


def test_a_value_typed_into_the_page_is_in_the_element_afterwards(pages: list[Path]) -> None:
    with agentic.ChromeSession() as session:
        page = session.page(pages[0])
        element = page.rows()[0]["id"]
        assert page.values()[element] == ""
        page.fill(element, "Avery Nakamura")
        assert page.values()[element] == "Avery Nakamura"


def test_a_page_named_by_a_relative_path_still_loads(
    pages: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--pages some/relative/dir` is the ordinary way to type it.

    A `file://` URI is absolute by definition, so a relative path has to be
    resolved before it becomes one. This failed on the first real run of the
    demo, with a message about URIs and no page opened.
    """
    monkeypatch.chdir(pages[0].parent)
    with agentic.ChromeSession() as session:
        assert session.page(Path(pages[0].name)).rows()


def test_a_tab_holding_another_document_is_not_read_as_the_page(
    pages: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page is read only once the tab actually holds it.

    `create_tab` returns before the navigation commits, and until it does the
    tab holds the initial `about:blank`, which reports `readyState` complete
    because it is a complete document. Waiting on `readyState` alone let the
    extraction run against that blank document and report a page with no
    fields, no form id and no language, with no error raised: on the first
    hosted CI run two tests here saw exactly that on windows-latest.

    Asking a tab that holds one page to prepare a different one is the same
    mismatch without the race, so it pins the fix deterministically.
    """
    monkeypatch.setattr(agentic, "READY_TIMEOUT_S", 0.5)
    with agentic.ChromeSession() as session:
        page = session.page(pages[0])
        with pytest.raises(agentic.BrowserError, match="rather than a complete"):
            page.prepare(pages[1].resolve().as_uri())


def test_a_page_answers_a_question_about_its_own_layout(pages: list[Path]) -> None:
    """`ChromePage.evaluate`, which is how the README GIF frames whole sections.

    A caller that has a page open has questions this module has no business
    enumerating. The one that exists asks the rendered report where its own
    sections are, so that a frame cropped out of a tall capture lands on a whole
    figure; the assertion here is the same shape on a page that is always there.
    """
    with agentic.ChromeSession() as session:
        page = session.page(pages[0])
        element = page.rows()[0]["id"]
        box = json.loads(
            page.evaluate(
                "JSON.stringify(Object.entries("
                f"document.getElementById({json.dumps(element)}).getBoundingClientRect()"
                ".toJSON()))"
            )
        )
        measured = dict(box)
        assert measured["width"] > 0 and measured["height"] > 0
        assert measured["bottom"] > measured["top"]


def test_filling_an_element_that_is_not_there_raises_rather_than_passing(
    pages: list[Path],
) -> None:
    with agentic.ChromeSession() as session:
        page = session.page(pages[0])
        with pytest.raises(agentic.BrowserError, match="no element"):
            page.fill("not_on_this_page", "Avery")


def test_the_browser_run_and_the_no_browser_run_agree_on_every_number(
    pages: list[Path],
) -> None:
    """The claim that makes `--no-browser` a test of the demo rather than of itself."""
    settings = {
        "classify": heuristic,
        "policy": always_fill(),
        "reward": RewardModel(),
    }
    driven = agentic.run_demo(
        pages, config=agentic.DemoConfig(engine="rules", browser=True), **settings
    )
    parsed = agentic.run_demo(
        pages, config=agentic.DemoConfig(engine="rules", browser=False), **settings
    )
    assert driven.n_fields == parsed.n_fields
    assert driven.n_filled == parsed.n_filled
    assert driven.n_landed == driven.n_filled, "every value the demo typed has to be in the DOM"
    assert driven.fill_accuracy == pytest.approx(parsed.fill_accuracy)
    assert driven.expected_reward == pytest.approx(parsed.expected_reward)
    assert [page.to_dict()["outcomes"] for page in driven.pages] == [
        page.to_dict()["outcomes"] for page in parsed.pages
    ]


def test_the_demo_captures_one_screenshot_at_scale_one(pages: list[Path], tmp_path: Path) -> None:
    shot = tmp_path / "agentic-demo.png"
    result = agentic.run_demo(
        pages,
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        config=agentic.DemoConfig(
            engine="rules", browser=True, screenshot=shot, screenshot_page=pages[0].stem
        ),
    )
    assert result.screenshot == shot
    data = shot.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) < SCREENSHOT_CEILING_BYTES
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert width == agentic.VIEWPORT_WIDTH * agentic.SCREENSHOT_SCALE
    assert height < agentic.VIEWPORT_HEIGHT, "the capture crops to the content, not to the window"
