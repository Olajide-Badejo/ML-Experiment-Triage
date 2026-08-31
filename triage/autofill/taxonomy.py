"""The field label space, and where its spelling comes from.

**Provenance.** The VALUES are WHATWG HTML autocomplete tokens, taken verbatim
for every token included. That is not a stylistic preference: `Autofill_audit`
ships a 37 token `autofill_audit.taxonomy.Label` StrEnum validated against a 960
form corpus and built on the same value space, and the PyTorch Performance
Toolkit ships a 17 label snake_case taxonomy for the same job. Three projects by
one author that disagree about how to spell a postal code cannot exchange a
single evaluation file. Adopting the standard's spelling rather than inventing a
fourth one makes the strings interoperate by construction (E7b), and
`from_tpt_label` is the documented bridge for the one sibling that uses the
snake_case form.

This taxonomy is a SUBSET of `Autofill_audit`'s: eighteen tokens plus
`unknown`, chosen to cover the identity, address and payment groups a checkout
or registration form actually contains, which is what this repository's demo
workload generates. Every token here is spelled as it is spelled there.

**`unknown` is a real class, not a null.** A form holds newsletter checkboxes,
free text comments and honeypots, and a classifier whose head cannot say "none
of these" has to spend its probability mass on a wrong answer instead. It is the
last index by convention, so a caller slicing the head down to the tokens it
fills can take a prefix.
"""

from __future__ import annotations

from enum import StrEnum


class FieldType(StrEnum):
    """One form field's ground truth type, valued as a WHATWG autofill token.

    `StrEnum` rather than `Enum` so that a member IS its token: it serialises to
    JSON as the string, compares equal to the string a sibling repository wrote,
    and needs no `.value` at any boundary. The declaration order is the model
    head order and is therefore part of the on disk format of a `.npz` of
    weights; append rather than reorder.
    """

    GIVEN_NAME = "given-name"
    FAMILY_NAME = "family-name"
    NAME = "name"
    EMAIL = "email"
    TEL = "tel"
    ADDRESS_LINE1 = "address-line1"
    ADDRESS_LINE2 = "address-line2"
    ADDRESS_LEVEL2 = "address-level2"
    ADDRESS_LEVEL1 = "address-level1"
    POSTAL_CODE = "postal-code"
    COUNTRY_NAME = "country-name"
    ORGANIZATION = "organization"
    CC_NAME = "cc-name"
    CC_NUMBER = "cc-number"
    CC_EXP_MONTH = "cc-exp-month"
    CC_EXP_YEAR = "cc-exp-year"
    CC_CSC = "cc-csc"
    USERNAME = "username"
    UNKNOWN = "unknown"


#: The head, in order. A tuple rather than a call to `list(FieldType)` at every
#: use site, so that the order is written once and read everywhere.
FIELD_TYPES: tuple[FieldType, ...] = tuple(FieldType)

#: `token -> head index`, built once. Enum lookup is a dictionary lookup already,
#: but this is on the inner loop of every featurisation and every score.
_INDEX: dict[FieldType, int] = {member: position for position, member in enumerate(FIELD_TYPES)}

#: The PyTorch Performance Toolkit's snake_case labels, mapped to this value
#: space. Thirteen of the seventeen are a separator swap; the four that are not
#: are the ones worth writing down, because `city` and `address-level2` are the
#: same concept under two names and no automatic rule connects them.
TPT_LABELS: dict[str, FieldType] = {
    "given_name": FieldType.GIVEN_NAME,
    "family_name": FieldType.FAMILY_NAME,
    "full_name": FieldType.NAME,
    "email": FieldType.EMAIL,
    "phone": FieldType.TEL,
    "address_line1": FieldType.ADDRESS_LINE1,
    "address_line2": FieldType.ADDRESS_LINE2,
    "city": FieldType.ADDRESS_LEVEL2,
    "state": FieldType.ADDRESS_LEVEL1,
    "postal_code": FieldType.POSTAL_CODE,
    "country": FieldType.COUNTRY_NAME,
    "organization": FieldType.ORGANIZATION,
    "cc_name": FieldType.CC_NAME,
    "cc_number": FieldType.CC_NUMBER,
    "cc_exp_month": FieldType.CC_EXP_MONTH,
    "cc_exp_year": FieldType.CC_EXP_YEAR,
    "cc_csc": FieldType.CC_CSC,
}


def index_of(field_type: FieldType) -> int:
    """The stable head index of a token."""
    return _INDEX[field_type]


def type_at(index: int) -> FieldType:
    """The token at a head index, refusing one the head does not have.

    An out of range class index is the shape of a weights file trained against a
    different taxonomy, and returning `unknown` for it would turn that into a
    silently wrong evaluation rather than a loud one.
    """
    if not 0 <= index < len(FIELD_TYPES):
        raise ValueError(
            f"class index {index} is outside the taxonomy: the head has "
            f"{len(FIELD_TYPES)} classes, indices 0 to {len(FIELD_TYPES) - 1}"
        )
    return FIELD_TYPES[index]


def from_tpt_label(label: str) -> FieldType:
    """A TPT snake_case label, or one of our own tokens, as a `FieldType`.

    Accepting our own value space too is not laxity: a bridge is called from
    code holding a label whose origin it may not know, and a token that is
    already correct should not have to be recognised as wrong first. Anything
    else raises, naming the label, because the alternative is mapping an
    unrecognised class to `unknown` and reporting an accuracy that counts it.
    """
    if label in TPT_LABELS:
        return TPT_LABELS[label]
    try:
        return FieldType(label)
    except ValueError:
        raise KeyError(
            f"{label!r} is not a TPT field label and not one of this taxonomy's tokens; "
            f"the mapping covers {', '.join(sorted(TPT_LABELS))}"
        ) from None
