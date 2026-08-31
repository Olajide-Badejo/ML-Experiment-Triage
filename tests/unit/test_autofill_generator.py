"""The synthetic form corpus: determinism, coverage, and the German lexicon.

The generator is the ground truth of this whole vertical, so the properties
tested here are the ones every downstream number rests on: the same seed gives
the same bytes, the noise knobs actually bite, every locale named in the
specification is present with the vocabulary it names, and the ground truth in
the HTML pages agrees with the ground truth in the JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triage.autofill.generator import (
    GERMAN_LEXICON_FLOOR,
    LEXICON,
    LOCALES,
    TEMPLATES,
    FieldRecord,
    GeneratorConfig,
    generate_fields,
    generate_pages,
    load_split,
    write_corpus,
)
from triage.autofill.taxonomy import FIELD_TYPES, FieldType


@pytest.fixture(scope="module")
def small() -> GeneratorConfig:
    return GeneratorConfig(n_fields=600)


@pytest.fixture(scope="module")
def records(small: GeneratorConfig) -> list[FieldRecord]:
    return generate_fields(small, seed=0)


def test_the_corpus_is_exactly_the_requested_size(records: list[FieldRecord]) -> None:
    assert len(records) == 600


def test_the_same_seed_gives_the_same_records(small: GeneratorConfig) -> None:
    first = generate_fields(small, seed=7)
    second = generate_fields(small, seed=7)
    assert first == second


def test_a_different_seed_gives_a_different_corpus(small: GeneratorConfig) -> None:
    assert generate_fields(small, seed=7) != generate_fields(small, seed=8)


def test_the_written_corpus_is_byte_identical_across_two_runs(
    tmp_path: Path, small: GeneratorConfig
) -> None:
    """Byte determinism, which is the property the specification asks for.

    Equal records are not enough: the numbers in this repository are reproduced
    from files, so what has to match is what lands on disk, including the HTML
    pages and the metadata.
    """
    left = write_corpus(tmp_path / "a", small, seed=3)
    right = write_corpus(tmp_path / "b", small, seed=3)
    assert sorted(p.name for p in left.every_file()) == sorted(p.name for p in right.every_file())
    for one, other in zip(sorted(left.every_file()), sorted(right.every_file()), strict=True):
        assert one.read_bytes() == other.read_bytes(), one.name


def test_the_corpus_files_are_ascii_so_the_bytes_do_not_depend_on_a_codec(
    tmp_path: Path, small: GeneratorConfig
) -> None:
    """The German lexicon is written in the transliteration the spec uses."""
    paths = write_corpus(tmp_path / "corpus", small, seed=1)
    for jsonl in paths.splits.values():
        jsonl.read_bytes().decode("ascii")


def test_every_split_is_present_and_disjoint(tmp_path: Path, small: GeneratorConfig) -> None:
    paths = write_corpus(tmp_path / "corpus", small, seed=0)
    loaded = {split: load_split(paths.root, split) for split in ("train", "val", "test")}
    assert sum(len(rows) for rows in loaded.values()) == 600
    ids = [row.form_id for rows in loaded.values() for row in rows]
    assert len(ids) == len(set(ids))
    for rows in loaded.values():
        assert rows, "every split has to hold rows for a comparison to be possible"


def test_a_form_never_straddles_two_splits(records: list[FieldRecord]) -> None:
    """Fields on one form share markup, so splitting inside one leaks it."""
    by_form: dict[str, set[str]] = {}
    for record in records:
        by_form.setdefault(record.form_key, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_form.values())


def test_both_locales_appear_and_are_labelled(records: list[FieldRecord]) -> None:
    assert {record.locale for record in records} == set(LOCALES)


def test_every_template_appears(records: list[FieldRecord]) -> None:
    assert {record.template for record in records} == set(TEMPLATES)


def test_the_template_id_names_the_locale_and_the_markup_variant(
    records: list[FieldRecord],
) -> None:
    for record in records:
        assert record.template in record.template_id
        assert record.locale in record.template_id
    # Enough clusters that a clustered permutation has resolution to work with.
    assert len({record.template_id for record in records}) >= 12


def test_the_form_id_is_the_per_field_unit_the_consumer_schema_uses(
    records: list[FieldRecord],
) -> None:
    """E5: the fixture pairs on `form_id` and clusters on `template_id`."""
    assert len({record.form_id for record in records}) == len(records)


def test_every_field_type_in_the_taxonomy_is_generated() -> None:
    records = generate_fields(GeneratorConfig(n_fields=4000), seed=0)
    assert {record.field_type for record in records} == set(FIELD_TYPES)


def test_every_record_carries_every_attribute_the_specification_lists(
    records: list[FieldRecord],
) -> None:
    row = records[0].to_json_dict()
    for key in (
        "label",
        "name",
        "element_id",
        "placeholder",
        "autocomplete",
        "input_type",
        "section",
        "previous_label",
        "next_label",
        "locale",
        "field_type",
    ):
        assert key in row, key


def test_the_sections_are_the_four_the_specification_names(records: list[FieldRecord]) -> None:
    assert {record.section for record in records} <= {"billing", "shipping", "account", "payment"}


def test_label_dropout_is_a_knob_that_bites(small: GeneratorConfig) -> None:
    none = generate_fields(GeneratorConfig(n_fields=small.n_fields, label_dropout=0.0), seed=0)
    lots = generate_fields(GeneratorConfig(n_fields=small.n_fields, label_dropout=0.9), seed=0)
    assert not any(record.label == "" for record in none)
    assert sum(record.label == "" for record in lots) > 0.7 * len(lots)


def test_the_adversarial_generic_attribute_knob_bites(small: GeneratorConfig) -> None:
    lots = generate_fields(
        GeneratorConfig(n_fields=small.n_fields, generic_attribute_prob=1.0), seed=0
    )
    assert all(record.name.startswith("input_") for record in lots)


def test_the_abbreviation_knob_bites(small: GeneratorConfig) -> None:
    never = generate_fields(
        GeneratorConfig(n_fields=small.n_fields, label_dropout=0.0, abbreviation_prob=0.0), seed=0
    )
    always = generate_fields(
        GeneratorConfig(n_fields=small.n_fields, label_dropout=0.0, abbreviation_prob=1.0), seed=0
    )
    assert sum(len(r.label) for r in always) < sum(len(r.label) for r in never)


def test_the_autocomplete_attribute_is_present_and_sometimes_wrong(
    small: GeneratorConfig,
) -> None:
    records = generate_fields(
        GeneratorConfig(n_fields=2000, p_autocomplete=1.0, p_autocomplete_correct=0.6),
        seed=0,
    )
    present = [r for r in records if r.autocomplete]
    assert len(present) == len(records)
    wrong = [r for r in present if r.autocomplete != r.field_type.value]
    assert 0.2 * len(present) < len(wrong) < 0.6 * len(present)


def test_no_autocomplete_attribute_when_the_probability_is_zero() -> None:
    records = generate_fields(GeneratorConfig(n_fields=200, p_autocomplete=0.0), seed=0)
    assert all(record.autocomplete == "" for record in records)


def test_the_german_lexicon_holds_every_term_the_specification_names() -> None:
    """5.2 names a floor for the de_DE vocabulary. This is that floor."""
    written = {
        text.lower()
        for entry in LEXICON["de_DE"].values()
        for text in (*entry.labels, *entry.abbreviations)
    }
    missing = [term for term in GERMAN_LEXICON_FLOOR if term.lower() not in written]
    assert missing == [], missing


def test_the_german_lexicon_has_a_synonym_tail_beyond_the_canonical_terms() -> None:
    """The margin in acceptance criterion 2 has to come from somewhere real."""
    canonical = {term.lower() for term in GERMAN_LEXICON_FLOOR}
    tail = [
        text
        for entry in LEXICON["de_DE"].values()
        for text in entry.labels
        if text.lower() not in canonical
    ]
    assert len(tail) >= 20


def test_the_card_numbers_are_the_documented_test_numbers() -> None:
    """Section 10: synthetic identity data only, card fields use test numbers."""
    placeholders = {
        text for locale in LOCALES for text in LEXICON[locale][FieldType.CC_NUMBER].placeholders
    }
    assert placeholders == {"4242 4242 4242 4242"}


def test_the_pages_are_one_per_template_per_locale(small: GeneratorConfig) -> None:
    pages = generate_pages(small, seed=0)
    assert set(pages) == {
        f"{template}_{locale}.html" for template in TEMPLATES for locale in LOCALES
    }


def test_every_input_on_a_page_carries_its_ground_truth(small: GeneratorConfig) -> None:
    pages = generate_pages(small, seed=0)
    for name, html in pages.items():
        assert html.count("<input") == html.count("data-truth="), name
        assert html.count("<input") >= 4, name
        assert "<!doctype html>" in html


def test_the_page_ground_truth_values_are_taxonomy_tokens(small: GeneratorConfig) -> None:
    import re

    tokens = {
        match
        for html in generate_pages(small, seed=0).values()
        for match in re.findall(r'data-truth="([^"]*)"', html)
    }
    assert tokens <= {member.value for member in FIELD_TYPES}


def test_the_metadata_records_the_knobs_the_corpus_was_built_with(
    tmp_path: Path, small: GeneratorConfig
) -> None:
    """A corpus whose noise settings are not written down is not reproducible."""
    paths = write_corpus(tmp_path / "corpus", small, seed=11)
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    assert meta["seed"] == 11
    assert meta["config"]["label_dropout"] == small.label_dropout
    assert meta["config"]["abbreviation_prob"] == small.abbreviation_prob
    assert meta["counts"]["train"] + meta["counts"]["val"] + meta["counts"]["test"] == 600


def test_a_config_with_an_impossible_knob_is_refused() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        GeneratorConfig(label_dropout=1.5)
    with pytest.raises(ValueError, match="at least one field"):
        GeneratorConfig(n_fields=0)


def test_loading_a_split_that_was_never_written_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no fields\.val\.jsonl"):
        load_split(tmp_path, "val")
