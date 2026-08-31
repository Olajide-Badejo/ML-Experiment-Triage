"""The agentic demo's logic, with no browser anywhere near it.

This file is the `--no-browser` half of 5.4, and it is where the demo is
actually specified. Everything a browser adds (a process, a protocol, a paint)
is a transport; what has to be right is that the fields are recovered from the
page, that the policy decides over them, that the value typed is the one the
locale's profile holds, and that the reward is the reward `policy.py` defines.
All four are checked here, against pages the generator wrote, and they run in
CI on a machine with no Chrome on it.

The one property that ties the two halves together is asserted in
`tests/integration/test_agentic_browser.py`: the fields the parser recovers and
the fields the live DOM reports are the same fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage.autofill import agentic
from triage.autofill.evaluate import ScoredEngine, heuristic_predict
from triage.autofill.generator import (
    PAGE_SPAWN_BASE,
    GeneratorConfig,
    _build_form,
    generate_pages,
    write_corpus,
)
from triage.autofill.policy import RewardModel, always_fill, global_threshold, never_fill
from triage.autofill.taxonomy import FIELD_TYPES, FieldType, index_of

CONFIG = GeneratorConfig(n_fields=600, locales=("en_US", "de_DE"))


@pytest.fixture(scope="module")
def pages() -> dict[str, str]:
    return generate_pages(CONFIG, seed=0)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("agentic")
    write_corpus(root / "data", CONFIG, seed=0)
    return root / "data"


# ------------------------------------------------------------------ the parser


def test_the_parser_recovers_every_field_the_generator_wrote(pages: dict[str, str]) -> None:
    """The stdlib parse of a page equals the records the page was written from.

    This is the load bearing assertion of the whole no browser mode. If it
    holds, the classifier sees exactly what it was trained on and a fill
    accuracy measured here is a fill accuracy; if it drifts by one attribute,
    every number below is about a different corpus.
    """
    for index, template in enumerate(("checkout", "registration", "address_book")):
        for offset, locale in enumerate(CONFIG.locales):
            form_index = PAGE_SPAWN_BASE + index * len(CONFIG.locales) + offset
            written = _build_form(form_index, CONFIG, 0, template, locale)
            source = agentic.parse_page(pages[f"{template}_{locale}.html"], f"{template}_{locale}")
            recovered = source.records()
            assert len(recovered) == len(written)
            for got, expected in zip(recovered, written, strict=True):
                assert got.label == expected.label
                assert got.name == expected.name
                assert got.element_id == expected.element_id
                assert got.placeholder == expected.placeholder
                assert got.autocomplete == expected.autocomplete
                assert got.input_type == expected.input_type
                assert got.section == expected.section
                assert got.previous_label == expected.previous_label
                assert got.next_label == expected.next_label
                assert got.field_type is expected.field_type


def test_the_parser_reads_the_locale_and_the_template_off_the_page(pages: dict[str, str]) -> None:
    source = agentic.parse_page(pages["checkout_de_DE.html"], "checkout_de_DE")
    assert source.locale == "de_DE"
    assert source.template == "checkout"
    assert all(record.locale == "de_DE" for record in source.records())


def test_a_dropped_label_is_recovered_as_an_empty_string() -> None:
    html = (
        '<!doctype html>\n<html lang="en-US">\n<head><title>t</title></head>\n<body>\n'
        '<form id="checkout">\n<fieldset><legend>account</legend>\n'
        '<input type="text" id="input_1" name="input_1" data-truth="unknown">\n'
        '<label for="mail_2">Email</label>\n'
        '<input type="email" id="mail_2" name="mail" data-truth="email">\n'
        "</fieldset>\n</form>\n</body>\n</html>\n"
    )
    records = agentic.parse_page(html, "checkout_en_US").records()
    assert [record.label for record in records] == ["", "Email"]
    assert [record.next_label for record in records] == ["Email", ""]
    assert [record.previous_label for record in records] == ["", ""]


def test_the_submit_button_is_not_a_field(pages: dict[str, str]) -> None:
    """`<button type="submit">` is not an input, and nothing may invent it."""
    records = agentic.parse_page(pages["checkout_en_US.html"], "checkout_en_US").records()
    assert all(
        record.field_type is not FieldType.UNKNOWN or record.element_id for record in records
    )
    assert all("submit" not in record.input_type for record in records)


def test_an_escaped_label_comes_back_unescaped() -> None:
    html = (
        '<!doctype html>\n<html lang="de-DE">\n<body>\n<form id="registration">\n'
        "<fieldset><legend>account</legend>\n"
        '<label for="a_1">Stra&#39;sse &amp; Nr.</label>\n'
        '<input type="text" id="a_1" name="a" data-truth="address-line1">\n'
        "</fieldset>\n</form>\n</body>\n</html>\n"
    )
    assert agentic.parse_page(html, "registration_de_DE").records()[0].label == "Stra'sse & Nr."


def test_a_page_with_no_inputs_is_refused_rather_than_scored() -> None:
    html = '<!doctype html>\n<html lang="en-US">\n<body><form id="checkout"></form></body></html>\n'
    with pytest.raises(ValueError, match="no input"):
        agentic.parse_page(html, "checkout_en_US").records()


# ----------------------------------------------------------------- the profile


def test_the_profile_holds_a_value_for_every_token_in_both_locales() -> None:
    for locale in ("en_US", "de_DE"):
        values = agentic.profile_for(locale)
        assert set(values) == set(FIELD_TYPES)
        fillable = [member for member in FIELD_TYPES if member is not FieldType.UNKNOWN]
        assert all(values[member] for member in fillable)


def test_the_unknown_token_has_no_value_to_type() -> None:
    assert agentic.profile_for("en_US")[FieldType.UNKNOWN] == ""
    assert agentic.profile_for("de_DE")[FieldType.UNKNOWN] == ""


def test_the_card_number_is_the_documented_test_number() -> None:
    """Section 10: card fields use documented test numbers, in both locales."""
    for locale in ("en_US", "de_DE"):
        assert agentic.profile_for(locale)[FieldType.CC_NUMBER] == "4242 4242 4242 4242"


def test_an_unknown_locale_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="fr_FR"):
        agentic.profile_for("fr_FR")


# ---------------------------------------------------------- the missing extra


def test_a_missing_driver_is_reported_as_a_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one place a user meets the `agentic` extra is a message naming it."""
    import importlib

    import triage._extras as extras

    def refuse(name: str) -> None:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(extras.importlib, "import_module", refuse)
    reason = agentic.why_no_browser()
    assert reason is not None
    assert "choreographer" in reason
    assert 'pip install "ml-experiment-triage[agentic]"' in reason
    assert not agentic.chrome_available()
    # And the real one still imports, so the patch was the only reason.
    monkeypatch.undo()
    assert importlib.import_module("triage.autofill.agentic") is agentic


def test_a_driver_that_finds_no_browser_points_at_the_cheap_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import triage._extras as extras

    class Chromium:
        @staticmethod
        def find_browser(*, skip_local: bool) -> str | None:
            return None

    monkeypatch.setattr(
        extras.importlib, "import_module", lambda name: type("m", (), {"Chromium": Chromium})
    )
    reason = agentic.why_no_browser()
    assert reason is not None
    assert "--no-browser" in reason


# ------------------------------------------------------------- the static page


def test_the_static_page_takes_a_value_and_reads_it_back(pages: dict[str, str]) -> None:
    page = agentic.StaticPage(pages["checkout_en_US.html"], "checkout_en_US")
    first = page.rows()[0]["id"]
    assert page.values()[first] == ""
    page.fill(first, "Avery")
    assert page.values()[first] == "Avery"


def test_the_static_page_refuses_an_element_that_is_not_there(pages: dict[str, str]) -> None:
    page = agentic.StaticPage(pages["checkout_en_US.html"], "checkout_en_US")
    with pytest.raises(KeyError, match="nope"):
        page.fill("nope", "Avery")


# ------------------------------------------------------------------ the scoring


def heuristic(records: list) -> ScoredEngine:
    predicted, confidence, latency = heuristic_predict(records)
    return ScoredEngine(
        engine="rules", predicted=predicted, confidence=confidence, latency_us=latency
    )


def test_never_fill_types_nothing_and_earns_nothing(pages: dict[str, str]) -> None:
    result = agentic.run_page(
        agentic.StaticPage(pages["checkout_en_US.html"], "checkout_en_US"),
        classify=heuristic,
        policy=never_fill(),
        reward=RewardModel(),
        engine="rules",
    )
    assert result.n_filled == 0
    assert result.total_reward == 0.0
    assert result.fill_accuracy is None
    assert all(outcome.value == "" for outcome in result.outcomes)


def test_always_fill_earns_exactly_what_the_reward_model_says(pages: dict[str, str]) -> None:
    """The reward is recomputed here from the truth, one field at a time."""
    reward = RewardModel()
    result = agentic.run_page(
        agentic.StaticPage(pages["checkout_de_DE.html"], "checkout_de_DE"),
        classify=heuristic,
        policy=always_fill(),
        reward=reward,
        engine="rules",
    )
    expected = 0.0
    for outcome in result.outcomes:
        if not outcome.filled:
            continue
        expected += 1.0 if outcome.correct else -reward.penalty(FieldType(outcome.truth))
    assert result.total_reward == pytest.approx(expected)
    assert result.expected_reward == pytest.approx(expected / result.n_fields)


def test_a_field_predicted_unknown_is_never_filled(pages: dict[str, str]) -> None:
    """Always fill still types nothing where the profile has nothing to type."""
    result = agentic.run_page(
        agentic.StaticPage(pages["registration_en_US.html"], "registration_en_US"),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        engine="rules",
    )
    unknowns = [item for item in result.outcomes if item.predicted == FieldType.UNKNOWN.value]
    assert unknowns, "the corpus puts adversarial fields on every page"
    assert all(not item.filled and item.value == "" for item in unknowns)


def test_a_correct_fill_types_the_locale_profile_value(pages: dict[str, str]) -> None:
    profile = agentic.profile_for("de_DE")
    result = agentic.run_page(
        agentic.StaticPage(pages["checkout_de_DE.html"], "checkout_de_DE"),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        engine="rules",
    )
    filled = [item for item in result.outcomes if item.filled]
    assert filled
    for item in filled:
        assert item.value == profile[FieldType(item.predicted)]
        assert item.landed, "a value that did not read back is a defect, not a wrong answer"


def test_a_higher_threshold_fills_fewer_fields(pages: dict[str, str]) -> None:
    page = "address_book_en_US"
    filled = []
    for threshold in (0.0, 0.7, 0.9, 2.0):
        result = agentic.run_page(
            agentic.StaticPage(pages[f"{page}.html"], page),
            classify=heuristic,
            policy=global_threshold(threshold),
            reward=RewardModel(),
            engine="rules",
        )
        filled.append(result.n_filled)
    assert filled == sorted(filled, reverse=True)
    assert filled[0] > filled[-1] == 0


def test_the_wrong_fill_penalty_follows_the_cost_tier(pages: dict[str, str]) -> None:
    """A payment mistake costs four correct fills; the same mistake as `other` costs one."""
    cheap = agentic.run_page(
        agentic.StaticPage(pages["checkout_en_US.html"], "checkout_en_US"),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel.parse("payment=0,identity=0,address=0,other=0"),
        engine="rules",
    )
    dear = agentic.run_page(
        agentic.StaticPage(pages["checkout_en_US.html"], "checkout_en_US"),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel.parse("payment=40,identity=40,address=20,other=10"),
        engine="rules",
    )
    assert cheap.n_filled == dear.n_filled
    assert cheap.total_reward > dear.total_reward


# --------------------------------------------------------------- the whole run


def test_the_demo_covers_three_pages_in_two_locales(corpus: Path) -> None:
    """Acceptance criterion 5's table, measured with no browser."""
    result = agentic.run_demo(
        sorted((corpus / "pages").glob("*.html")),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        config=agentic.DemoConfig(engine="rules", browser=False),
    )
    assert len(result.pages) == 6
    assert {page.locale for page in result.pages} == {"en_US", "de_DE"}
    assert {page.template for page in result.pages} == {
        "checkout",
        "registration",
        "address_book",
    }
    assert 0.0 <= result.fill_accuracy <= 1.0
    assert result.n_fields == sum(page.n_fields for page in result.pages)


def test_the_table_names_every_page_and_the_total(corpus: Path) -> None:
    result = agentic.run_demo(
        sorted((corpus / "pages").glob("*.html")),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        config=agentic.DemoConfig(engine="rules", browser=False),
    )
    table = agentic.format_table(result)
    for page in result.pages:
        assert page.template in table
        assert page.locale in table
    assert "all pages" in table


def test_the_run_directories_are_the_shape_ingest_already_reads(
    corpus: Path, tmp_path: Path
) -> None:
    result = agentic.run_demo(
        sorted((corpus / "pages").glob("*.html")),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        config=agentic.DemoConfig(engine="rules", browser=False, bootstrap=3),
    )
    paths = agentic.write_demo(tmp_path / "out", result)
    assert len(paths.runs) == 6 * 3
    for run in paths.runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        assert config["variant"] == "rules"
        rows = [
            json.loads(line)
            for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert rows
        assert set(rows[0]) == {"step", *agentic.TASK_TAGS}
    seeds = [
        json.loads((run / "config.json").read_text(encoding="utf-8"))["seed"] for run in paths.runs
    ]
    assert len(set(seeds)) == len(seeds), "one condition cannot hold two runs of one seed"


def test_the_summary_carries_the_measured_table(corpus: Path, tmp_path: Path) -> None:
    result = agentic.run_demo(
        sorted((corpus / "pages").glob("*.html")),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        config=agentic.DemoConfig(engine="rules", browser=False, bootstrap=2),
    )
    paths = agentic.write_demo(tmp_path / "out", result)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert summary["engine"] == "rules"
    assert summary["browser"] is False
    assert len(summary["pages"]) == 6
    assert summary["totals"]["fill_accuracy"] == pytest.approx(result.fill_accuracy)


def test_two_engines_into_one_directory_keep_two_summaries(corpus: Path, tmp_path: Path) -> None:
    """Ranking the engines means running the demo three times into one place."""
    out = tmp_path / "shared"
    for engine in ("rules", "ngram"):
        result = agentic.run_demo(
            sorted((corpus / "pages").glob("*.html")),
            classify=heuristic,
            policy=always_fill(),
            reward=RewardModel(),
            config=agentic.DemoConfig(engine=engine, browser=False, bootstrap=2),
        )
        agentic.write_demo(out, result)
    assert sorted(path.name for path in out.glob("agentic_*.json")) == [
        "agentic_ngram.json",
        "agentic_rules.json",
    ]
    assert len(sorted((out / "runs").iterdir())) == 2 * 6 * 2


def test_two_runs_of_the_demo_write_the_same_bytes(corpus: Path, tmp_path: Path) -> None:
    """Nothing here is a function of the clock except the timings, which are not written."""
    written = []
    for name in ("first", "second"):
        result = agentic.run_demo(
            sorted((corpus / "pages").glob("*.html")),
            classify=heuristic,
            policy=always_fill(),
            reward=RewardModel(),
            config=agentic.DemoConfig(engine="rules", browser=False, bootstrap=2),
        )
        paths = agentic.write_demo(tmp_path / name, result)
        written.append(paths.summary.read_bytes())
    assert written[0] == written[1]


def test_a_shorter_prediction_vector_is_refused(pages: dict[str, str]) -> None:
    def truncated(records: list) -> ScoredEngine:
        engine = heuristic(records)
        return ScoredEngine(
            engine=engine.engine,
            predicted=engine.predicted[:-1],
            confidence=engine.confidence[:-1],
            latency_us=engine.latency_us,
        )

    with pytest.raises(ValueError, match="one prediction per field"):
        agentic.run_page(
            agentic.StaticPage(pages["checkout_en_US.html"], "checkout_en_US"),
            classify=truncated,
            policy=always_fill(),
            reward=RewardModel(),
            engine="rules",
        )


def test_a_value_that_does_not_land_is_counted_as_a_wrong_fill(pages: dict[str, str]) -> None:
    """The read back is a check, not a formality: a page that swallows the value loses the reward."""

    class Swallowing(agentic.StaticPage):
        def fill(self, element_id: str, value: str) -> None:
            super().fill(element_id, "")

    result = agentic.run_page(
        Swallowing(pages["checkout_en_US.html"], "checkout_en_US"),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        engine="rules",
    )
    assert result.n_filled > 0
    assert result.n_landed == 0
    assert result.fill_accuracy == 0.0
    assert result.total_reward < 0


def test_the_cumulative_curve_converges_on_the_page_score(pages: dict[str, str]) -> None:
    result = agentic.run_page(
        agentic.StaticPage(pages["checkout_de_DE.html"], "checkout_de_DE"),
        classify=heuristic,
        policy=always_fill(),
        reward=RewardModel(),
        engine="rules",
    )
    curve = agentic.cumulative_curve(result.outcomes, np.arange(result.n_fields))
    assert len(curve) == result.n_fields
    assert curve[-1]["task/reward"] == pytest.approx(result.expected_reward)


def test_every_taxonomy_token_has_a_head_index_the_policy_can_index() -> None:
    """A guard on the coupling the demo relies on rather than on the demo itself."""
    assert [index_of(member) for member in FIELD_TYPES] == list(range(len(FIELD_TYPES)))
