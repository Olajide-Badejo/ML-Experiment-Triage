"""The field taxonomy: WHATWG values, the TPT mapping, and the model head index.

The value space is a cross repository contract (E7b), so these tests are written
as literals rather than as properties of whatever the enum happens to contain.
A token spelled differently here is a token that stops interoperating with
`autofill_audit.taxonomy.Label`, and nothing inside this repository would
notice.
"""

from __future__ import annotations

import pytest

from triage.autofill.taxonomy import (
    FIELD_TYPES,
    TPT_LABELS,
    FieldType,
    from_tpt_label,
    index_of,
    type_at,
)

#: The value space of 5.2, written out. Order is the model head order.
EXPECTED_VALUES = (
    "given-name",
    "family-name",
    "name",
    "email",
    "tel",
    "address-line1",
    "address-line2",
    "address-level2",
    "address-level1",
    "postal-code",
    "country-name",
    "organization",
    "cc-name",
    "cc-number",
    "cc-exp-month",
    "cc-exp-year",
    "cc-csc",
    "username",
    "unknown",
)


def test_the_value_space_is_the_whatwg_one_verbatim() -> None:
    assert tuple(member.value for member in FIELD_TYPES) == EXPECTED_VALUES


def test_the_enum_is_a_string_enum_so_the_values_travel_as_strings() -> None:
    """`json.dumps` of a member has to be the token, not an enum repr."""
    import json

    assert FieldType.POSTAL_CODE == "postal-code"
    assert json.dumps({"label": FieldType.POSTAL_CODE}) == '{"label": "postal-code"}'


def test_the_head_index_is_stable_and_dense() -> None:
    indices = [index_of(member) for member in FIELD_TYPES]
    assert indices == list(range(len(FIELD_TYPES)))
    assert index_of(FieldType.GIVEN_NAME) == 0
    assert index_of(FieldType.UNKNOWN) == len(FIELD_TYPES) - 1


def test_type_at_inverts_index_of() -> None:
    for member in FIELD_TYPES:
        assert type_at(index_of(member)) is member


def test_type_at_refuses_an_index_outside_the_head() -> None:
    with pytest.raises(ValueError, match="head has"):
        type_at(len(FIELD_TYPES))


def test_tpt_snake_case_labels_map_to_the_whatwg_values() -> None:
    """E7b: TPT's 17 snake_case labels arrive as this repo's token values."""
    assert len(TPT_LABELS) == 17
    assert from_tpt_label("given_name") is FieldType.GIVEN_NAME
    assert from_tpt_label("family_name") is FieldType.FAMILY_NAME
    assert from_tpt_label("postal_code") is FieldType.POSTAL_CODE
    assert from_tpt_label("cc_exp_month") is FieldType.CC_EXP_MONTH


def test_the_tpt_labels_that_are_not_a_hyphen_swap_are_mapped_by_hand() -> None:
    """The interesting half: names that differ by more than the separator."""
    assert from_tpt_label("city") is FieldType.ADDRESS_LEVEL2
    assert from_tpt_label("state") is FieldType.ADDRESS_LEVEL1
    assert from_tpt_label("country") is FieldType.COUNTRY_NAME
    assert from_tpt_label("phone") is FieldType.TEL


def test_every_tpt_label_resolves() -> None:
    for label in TPT_LABELS:
        assert isinstance(from_tpt_label(label), FieldType)


def test_an_unmapped_tpt_label_is_refused_rather_than_guessed() -> None:
    with pytest.raises(KeyError, match="not a TPT field label"):
        from_tpt_label("sortiness")


def test_a_whatwg_token_is_accepted_as_itself() -> None:
    """A caller holding one of our own values should not have to know which."""
    assert from_tpt_label("postal-code") is FieldType.POSTAL_CODE


def test_the_taxonomy_carries_its_provenance_in_the_docstring() -> None:
    """Ground rule: a value space copied from somewhere says where from."""
    import triage.autofill.taxonomy as module

    assert module.__doc__ is not None
    assert "WHATWG" in module.__doc__
    assert "autofill_audit" in module.__doc__
