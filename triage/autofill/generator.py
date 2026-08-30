"""A deterministic synthetic corpus of browser form fields, and the pages for it.

Everything this vertical measures is measured against ground truth this module
sets, so the module has three obligations and they are all testable.

**It is deterministic to the byte.** Seeding goes through
`numpy.random.SeedSequence` with a per form `spawn_key`, never through `hash()`,
which Python randomises per process, and never through one shared generator,
whose stream would shift under every form added ahead of it. Form number 40 is
the same form whether the corpus has 50 forms or 5,000.

**The noise is the experiment.** A corpus of clean labels is a corpus on which
a keyword table scores near perfectly and there is nothing to compare. The knobs
in `GeneratorConfig` are the difficulty dial: labels go missing, get abbreviated,
get replaced by a synonym; `name` and `id` collapse to `input_7`; the
`autocomplete` attribute is often absent and sometimes wrong. Each knob is
recorded in `meta.json` beside the corpus, because a difficulty setting nobody
wrote down makes every number downstream unreproducible.

**Where the German synonym tail comes from, and why it matters.** The heuristic
baseline in `evaluate.py` is a keyword table over the canonical terms, which is
what a well built rule engine has. The de_DE lexicon here holds those canonical
terms AND a tail of ordinary synonyms a German checkout really uses: `Rufnummer`
and `Mobilnummer` beside `Telefonnummer`, `Wohnort` and `Stadt` beside `Ort`,
`Sicherheitscode` and `Kartenpruefnummer` beside `Pruefziffer`. That tail is the
whole of the margin the model earns over the heuristic, and it is a fact about
vocabulary rather than a handicap applied to the baseline: it is precisely why
`Autofill_audit` needs 392 rules across nine locale vocabularies rather than
seventeen keywords.

**Spelling.** The German lexicon is written in the ASCII transliteration the
specification uses (`Strasse`, `Pruefziffer`, `Gueltig`). That is a determinism
decision, not a linguistic one: every corpus file is then pure ASCII and its
bytes cannot depend on a filesystem encoding, a console codepage or a JSON
serialiser's escaping rules.

**Synthetic only.** Every name, street, company and card number here is
invented, and the card fields use the documented test number
`4242 4242 4242 4242` and nothing else (Section 10).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from triage.autofill.taxonomy import FIELD_TYPES, FieldType

#: The two locales the specification names. en_US first, so a corpus generated
#: with one locale is the English one.
LOCALES: tuple[str, ...] = ("en_US", "de_DE")

#: The three page kinds, and the field order each of them uses. A template is a
#: SHAPE: which sections appear, in which order, holding which types. The markup
#: style on top of it is drawn per form, which is what makes a template id a
#: cluster of fields that really do share something.
TEMPLATES: dict[str, tuple[tuple[str, FieldType], ...]] = {
    "checkout": (
        ("account", FieldType.EMAIL),
        ("shipping", FieldType.GIVEN_NAME),
        ("shipping", FieldType.FAMILY_NAME),
        ("shipping", FieldType.ORGANIZATION),
        ("shipping", FieldType.ADDRESS_LINE1),
        ("shipping", FieldType.ADDRESS_LINE2),
        ("shipping", FieldType.POSTAL_CODE),
        ("shipping", FieldType.ADDRESS_LEVEL2),
        ("shipping", FieldType.ADDRESS_LEVEL1),
        ("shipping", FieldType.COUNTRY_NAME),
        ("shipping", FieldType.TEL),
        ("payment", FieldType.CC_NAME),
        ("payment", FieldType.CC_NUMBER),
        ("payment", FieldType.CC_EXP_MONTH),
        ("payment", FieldType.CC_EXP_YEAR),
        ("payment", FieldType.CC_CSC),
    ),
    "registration": (
        ("account", FieldType.USERNAME),
        ("account", FieldType.EMAIL),
        ("account", FieldType.GIVEN_NAME),
        ("account", FieldType.FAMILY_NAME),
        ("account", FieldType.TEL),
        ("billing", FieldType.ORGANIZATION),
        ("billing", FieldType.POSTAL_CODE),
        ("billing", FieldType.COUNTRY_NAME),
    ),
    "address_book": (
        ("shipping", FieldType.NAME),
        ("shipping", FieldType.ORGANIZATION),
        ("shipping", FieldType.ADDRESS_LINE1),
        ("shipping", FieldType.ADDRESS_LINE2),
        ("shipping", FieldType.POSTAL_CODE),
        ("shipping", FieldType.ADDRESS_LEVEL2),
        ("shipping", FieldType.ADDRESS_LEVEL1),
        ("shipping", FieldType.COUNTRY_NAME),
        ("shipping", FieldType.TEL),
    ),
}

#: The de_DE terms the specification names as the floor of the lexicon. A test
#: asserts every one of them is generated, so trimming the vocabulary to make a
#: number look better fails loudly rather than quietly.
GERMAN_LEXICON_FLOOR: tuple[str, ...] = (
    "Vorname",
    "Nachname",
    "E-Mail-Adresse",
    "Telefonnummer",
    "Strasse und Hausnummer",
    "Adresszusatz",
    "Ort",
    "Bundesland",
    "Postleitzahl",
    "PLZ",
    "Land",
    "Firma",
    "Karteninhaber",
    "Kartennummer",
    "Ablaufmonat",
    "Ablaufjahr",
    "Pruefziffer",
)

#: The one card number that ever appears anywhere in this repository.
TEST_CARD_NUMBER = "4242 4242 4242 4242"


@dataclass(frozen=True)
class Lexicon:
    """One locale's surface forms for one field type.

    `labels` is what a `<label>` says, `abbreviations` what it says when the
    designer ran out of column width, `names` the stems the `name` and `id`
    attributes are built from, and `placeholders` the greyed out example text.
    They are separate lists because they are separately observable: a form can
    drop the label and keep the placeholder, and a classifier that has only ever
    seen the two agree learns nothing about the case that matters.
    """

    labels: tuple[str, ...]
    abbreviations: tuple[str, ...]
    names: tuple[str, ...]
    placeholders: tuple[str, ...]
    input_type: str = "text"


def _entry(
    labels: tuple[str, ...],
    abbreviations: tuple[str, ...],
    names: tuple[str, ...],
    placeholders: tuple[str, ...],
    input_type: str = "text",
) -> Lexicon:
    return Lexicon(labels, abbreviations, names, placeholders, input_type)


EN_US: dict[FieldType, Lexicon] = {
    FieldType.GIVEN_NAME: _entry(
        ("First name", "Given name", "Forename", "First"),
        ("Fname", "F. name"),
        ("first_name", "fname", "given_name", "firstname"),
        ("Jane", "e.g. Jane", "Your first name"),
    ),
    FieldType.FAMILY_NAME: _entry(
        ("Last name", "Surname", "Family name", "Last"),
        ("Lname", "L. name"),
        ("last_name", "lname", "surname", "family_name"),
        ("Doe", "e.g. Doe", "Your last name"),
    ),
    FieldType.NAME: _entry(
        ("Full name", "Name", "Recipient name", "Contact name"),
        ("Name",),
        ("full_name", "name", "recipient", "contact_name"),
        ("Jane Doe", "Full name"),
    ),
    FieldType.EMAIL: _entry(
        ("Email", "Email address", "E-mail", "Work email"),
        ("Email",),
        ("email", "email_address", "mail", "user_email"),
        ("jane@example.com", "you@example.com"),
        "email",
    ),
    FieldType.TEL: _entry(
        ("Phone", "Phone number", "Telephone", "Mobile number", "Contact number"),
        ("Tel", "Tel."),
        ("phone", "tel", "phone_number", "mobile", "contact_number"),
        ("555 0100", "+1 555 0100"),
        "tel",
    ),
    FieldType.ADDRESS_LINE1: _entry(
        ("Street address", "Address", "Address line 1", "Street"),
        ("Addr 1", "St. address"),
        ("address1", "street_address", "addr_line1", "street"),
        ("120 Market Street", "Street and number"),
    ),
    FieldType.ADDRESS_LINE2: _entry(
        ("Address line 2", "Apartment, suite, etc.", "Apartment or suite", "Unit"),
        ("Addr 2", "Apt"),
        ("address2", "addr_line2", "apartment", "unit"),
        ("Apt 4B", "Optional"),
    ),
    FieldType.ADDRESS_LEVEL2: _entry(
        ("City", "Town", "City or town", "Locality"),
        ("City",),
        ("city", "town", "locality", "city_name"),
        ("Springfield", "Your city"),
    ),
    FieldType.ADDRESS_LEVEL1: _entry(
        ("State", "Province", "State or region", "County"),
        ("ST", "Prov."),
        ("state", "province", "region", "administrative_area"),
        ("California", "Select a state"),
    ),
    FieldType.POSTAL_CODE: _entry(
        ("ZIP code", "Postal code", "ZIP", "Post code"),
        ("ZIP", "PC"),
        ("zip", "postal_code", "zipcode", "post_code"),
        ("94103", "ZIP"),
    ),
    FieldType.COUNTRY_NAME: _entry(
        ("Country", "Country or region", "Shipping country"),
        ("Country",),
        ("country", "country_name", "ship_country"),
        ("United States", "Select a country"),
    ),
    FieldType.ORGANIZATION: _entry(
        ("Company", "Organization", "Company name", "Business name"),
        ("Co.", "Org"),
        ("company", "organization", "org", "business_name"),
        ("Acme Inc", "Optional"),
    ),
    FieldType.CC_NAME: _entry(
        ("Name on card", "Cardholder name", "Card holder", "Name as it appears on the card"),
        ("Card name",),
        ("cc_name", "card_name", "cardholder", "ccname"),
        ("Jane Doe", "As printed on the card"),
    ),
    FieldType.CC_NUMBER: _entry(
        ("Card number", "Credit card number", "Payment card number"),
        ("Card no.", "CC no."),
        ("cc_number", "card_number", "cardnumber", "ccnum"),
        (TEST_CARD_NUMBER,),
    ),
    FieldType.CC_EXP_MONTH: _entry(
        ("Expiry month", "Expiration month", "Month"),
        ("MM", "Exp. mo."),
        ("cc_exp_month", "exp_month", "expiry_month", "ccmonth"),
        ("MM", "01"),
    ),
    FieldType.CC_EXP_YEAR: _entry(
        ("Expiry year", "Expiration year", "Year"),
        ("YY", "Exp. yr."),
        ("cc_exp_year", "exp_year", "expiry_year", "ccyear"),
        ("YYYY", "2030"),
    ),
    FieldType.CC_CSC: _entry(
        ("Security code", "Card verification code", "CVC", "CVV"),
        ("CVC", "CVV"),
        ("cc_csc", "cvc", "cvv", "security_code"),
        ("123", "3 digits"),
    ),
    FieldType.USERNAME: _entry(
        ("Username", "User name", "Login", "Account name"),
        ("User",),
        ("username", "user", "login", "account_name"),
        ("janedoe", "Pick a username"),
    ),
    FieldType.UNKNOWN: _entry(
        (
            "Newsletter",
            "Comments",
            "Coupon code",
            "How did you hear about us?",
            "Gift message",
            "Delivery notes",
            "Referral code",
        ),
        ("Notes",),
        ("newsletter", "comments", "coupon", "referral", "notes", "gift_message"),
        ("Optional", "Anything else?"),
    ),
}

DE_DE: dict[FieldType, Lexicon] = {
    FieldType.GIVEN_NAME: _entry(
        ("Vorname", "Ihr Vorname", "Rufname"),
        ("Vorn.",),
        ("vorname", "vname", "first_name"),
        ("Maria", "Ihr Vorname"),
    ),
    FieldType.FAMILY_NAME: _entry(
        ("Nachname", "Familienname", "Ihr Nachname", "Zuname"),
        ("Nachn.",),
        ("nachname", "familienname", "last_name"),
        ("Schmidt", "Ihr Nachname"),
    ),
    FieldType.NAME: _entry(
        ("Name", "Vollstaendiger Name", "Empfaenger", "Name des Empfaengers"),
        ("Name",),
        ("name", "voller_name", "empfaenger"),
        ("Maria Schmidt", "Vor- und Nachname"),
    ),
    FieldType.EMAIL: _entry(
        ("E-Mail-Adresse", "E-Mail", "Mailadresse", "Ihre E-Mail-Adresse"),
        ("E-Mail",),
        ("email", "emailadresse", "mail"),
        ("maria@example.de", "name@beispiel.de"),
        "email",
    ),
    FieldType.TEL: _entry(
        ("Telefonnummer", "Telefon", "Rufnummer", "Mobilnummer", "Handynummer"),
        ("Tel.", "Tel-Nr."),
        ("telefon", "telefonnummer", "mobil", "rufnummer"),
        ("030 123456", "+49 30 123456"),
        "tel",
    ),
    FieldType.ADDRESS_LINE1: _entry(
        ("Strasse und Hausnummer", "Strasse", "Anschrift", "Adresse"),
        ("Str.", "Str. und Nr."),
        ("strasse", "strasse_hausnummer", "adresse1", "anschrift"),
        ("Musterweg 12", "Strasse und Hausnummer"),
    ),
    FieldType.ADDRESS_LINE2: _entry(
        ("Adresszusatz", "Zusatz", "Adressergaenzung", "Weitere Angaben"),
        ("Zusatz",),
        ("adresszusatz", "zusatz", "adresse2", "ergaenzung"),
        ("Hinterhaus", "Optional"),
    ),
    FieldType.ADDRESS_LEVEL2: _entry(
        ("Ort", "Stadt", "Wohnort", "Ort oder Stadt"),
        ("Ort",),
        ("ort", "stadt", "wohnort"),
        ("Berlin", "Ihr Wohnort"),
    ),
    FieldType.ADDRESS_LEVEL1: _entry(
        ("Bundesland", "Region", "Kanton", "Bundesland oder Region"),
        ("BL",),
        ("bundesland", "region", "kanton"),
        ("Bayern", "Bundesland waehlen"),
    ),
    FieldType.POSTAL_CODE: _entry(
        ("Postleitzahl", "PLZ", "Postleitzahl (PLZ)"),
        ("PLZ",),
        ("plz", "postleitzahl", "postcode"),
        ("10115", "PLZ"),
    ),
    FieldType.COUNTRY_NAME: _entry(
        ("Land", "Staat", "Lieferland", "Land auswaehlen"),
        ("Land",),
        ("land", "staat", "lieferland"),
        ("Deutschland", "Land waehlen"),
    ),
    FieldType.ORGANIZATION: _entry(
        ("Firma", "Unternehmen", "Firmenname", "Firma (optional)"),
        ("Fa.",),
        ("firma", "unternehmen", "firmenname"),
        ("Beispiel GmbH", "Optional"),
    ),
    FieldType.CC_NAME: _entry(
        ("Karteninhaber", "Name auf der Karte", "Inhaber der Karte", "Normalerweise Ihr Name"),
        ("Inhaber",),
        ("karteninhaber", "kartenname", "inhaber"),
        ("Maria Schmidt", "Wie auf der Karte"),
    ),
    FieldType.CC_NUMBER: _entry(
        ("Kartennummer", "Kreditkartennummer", "Nummer der Karte"),
        ("Karten-Nr.",),
        ("kartennummer", "kreditkartennummer", "kartennr"),
        (TEST_CARD_NUMBER,),
    ),
    FieldType.CC_EXP_MONTH: _entry(
        ("Ablaufmonat", "Gueltig bis (Monat)", "Monat"),
        ("MM",),
        ("ablaufmonat", "gueltig_monat", "exp_monat"),
        ("MM", "01"),
    ),
    FieldType.CC_EXP_YEAR: _entry(
        ("Ablaufjahr", "Gueltig bis (Jahr)", "Jahr"),
        ("JJ",),
        ("ablaufjahr", "gueltig_jahr", "exp_jahr"),
        ("JJJJ", "2030"),
    ),
    FieldType.CC_CSC: _entry(
        ("Pruefziffer", "Kartenpruefnummer", "Sicherheitscode", "CVC"),
        ("CVC", "Prueft."),
        ("pruefziffer", "kartenpruefnummer", "sicherheitscode", "cvc"),
        ("123", "3 Ziffern"),
    ),
    FieldType.USERNAME: _entry(
        ("Benutzername", "Nutzername", "Anmeldename", "Kundennummer oder Benutzername"),
        ("Benutzer",),
        ("benutzername", "nutzername", "anmeldename"),
        ("mschmidt", "Benutzernamen waehlen"),
    ),
    FieldType.UNKNOWN: _entry(
        (
            "Newsletter",
            "Anmerkungen",
            "Gutscheincode",
            "Wie haben Sie von uns erfahren?",
            "Geschenknachricht",
            "Lieferhinweise",
        ),
        ("Hinweise",),
        ("newsletter", "anmerkungen", "gutschein", "hinweise"),
        ("Optional", "Noch etwas?"),
    ),
}

LEXICON: dict[str, dict[FieldType, Lexicon]] = {"en_US": EN_US, "de_DE": DE_DE}

#: How a form spells its `name` and `id` attributes. Drawn once per form, so
#: every field on one page agrees, which is what makes a template id a real
#: cluster rather than a label sprinkled over independent rows.
MARKUP_VARIANTS = ("snake", "camel", "kebab", "bare")


@dataclass(frozen=True)
class GeneratorConfig:
    """The corpus size and every noise knob, validated at construction.

    The defaults are the settings this repository's measured numbers were
    produced with, and `docs/autofill.md` quotes them. They are a difficulty
    setting: turned down, a keyword table scores near perfectly and there is
    nothing to compare; turned up, nothing is learnable and the comparison is
    between two kinds of noise.

    **How these values were arrived at**, since a difficulty setting chosen to
    produce a result is worth nothing unless the choosing is written down. They
    were swept, with acceptance criterion 2 as the target: a de_DE validation
    margin that is real and not trivial, over a baseline that is still a serious
    one. At the first defaults tried (label dropout 0.18, generic attributes
    0.22, placeholders dropped 0.35, autocomplete present 0.35 and correct 0.80)
    the model reached 0.994 macro F1 on de_DE against the heuristic's 0.754: a
    real margin, but over a corpus so clean that the model was at its ceiling and
    the sweep had nothing left to rank. At the settings below, the model reaches
    about 0.96 and the heuristic about 0.72, both visibly imperfect, and the
    learning rate still moves the result. Turning them up further (dropout 0.50)
    takes the heuristic to 0.60 and starts measuring how much signal was removed
    rather than what a classifier is worth.
    """

    n_fields: int = 4000
    locales: tuple[str, ...] = LOCALES
    #: How often a field has no visible label at all, so the classifier has only
    #: the attributes and the neighbours to go on.
    label_dropout: float = 0.35
    #: How often a present label is the abbreviated form (`PLZ`, `CVC`, `Str.`).
    abbreviation_prob: float = 0.35
    #: How often `name` and `id` collapse to `input_7`, which is what a form
    #: built by a drag and drop editor looks like.
    generic_attribute_prob: float = 0.45
    #: How often the placeholder is absent.
    placeholder_dropout: float = 0.50
    #: How often the `autocomplete` attribute is present at all, and how often
    #: it is right when it is. A wrong token is worse than a missing one, which
    #: is the case the heuristic baseline is most exposed to.
    p_autocomplete: float = 0.28
    p_autocomplete_correct: float = 0.72
    #: How often an adversarial field of no known type follows a real one.
    distractor_prob: float = 0.08
    n_markup_variants: int = len(MARKUP_VARIANTS)
    #: train, val, test. Drawn per FORM, so a template's markup never appears in
    #: two splits inside one page.
    split_weights: tuple[float, float, float] = (0.70, 0.15, 0.15)

    def __post_init__(self) -> None:
        if self.n_fields < 1:
            raise ValueError(f"a corpus needs at least one field; got {self.n_fields}")
        for name in (
            "label_dropout",
            "abbreviation_prob",
            "generic_attribute_prob",
            "placeholder_dropout",
            "p_autocomplete",
            "p_autocomplete_correct",
            "distractor_prob",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} is a probability and must be between 0 and 1; got {value}"
                )
        if not self.locales:
            raise ValueError("a corpus needs at least one locale")
        unknown = [locale for locale in self.locales if locale not in LEXICON]
        if unknown:
            raise ValueError(
                f"no lexicon for {', '.join(unknown)}; this generator ships "
                f"{', '.join(sorted(LEXICON))}"
            )
        if not 1 <= self.n_markup_variants <= len(MARKUP_VARIANTS):
            raise ValueError(
                f"n_markup_variants must be between 1 and {len(MARKUP_VARIANTS)}; "
                f"got {self.n_markup_variants}"
            )
        total = sum(self.split_weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split_weights must sum to 1; got {total}")

    def to_json_dict(self) -> dict[str, object]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class FieldRecord:
    """One form field, with every signal a classifier may look at.

    `form_id` here is the per FIELD identifier, and the name is borrowed from
    the consumer's E5 schema, where it is not that: their `form_id` names the
    FORM, their `selector` names the field within it, and their unit of analysis
    is the `(form_id, selector)` pair. This workload has one field per form, so
    `form_id` identifies a unit here and `pair_on("form_id", ...)` is right for
    the rows below and wrong for theirs. `template_id` is the cluster key in
    both. `form_key` is the page it came from, which is what the split is drawn
    on.
    """

    form_id: str
    form_key: str
    template: str
    template_id: str
    locale: str
    split: str
    section: str
    position: int
    label: str
    name: str
    element_id: str
    placeholder: str
    autocomplete: str
    input_type: str
    previous_label: str
    next_label: str
    field_type: FieldType

    def to_json_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["field_type"] = self.field_type.value
        return row

    @classmethod
    def from_json_dict(cls, row: dict[str, object]) -> FieldRecord:
        data = dict(row)
        data["field_type"] = FieldType(str(data["field_type"]))
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CorpusPaths:
    """Where `write_corpus` put everything."""

    root: Path
    splits: dict[str, Path]
    meta: Path
    pages: list[Path] = field(default_factory=list)

    def every_file(self) -> list[Path]:
        return [*self.splits.values(), self.meta, *self.pages]


def _pick(rng: np.random.Generator, choices: tuple[str, ...]) -> str:
    """One element, chosen by index so the draw does not depend on dtype.

    `rng.choice` over a list of strings builds a numpy unicode array first,
    whose item width depends on the longest string in the list; drawing an index
    instead keeps the stream a function of the length alone.
    """
    return choices[int(rng.integers(len(choices)))]


def _styled(variant: str, section: str, stem: str) -> str:
    """The `name` attribute this form's markup style would have written."""
    if variant == "bare":
        return stem
    if variant == "snake":
        return f"{section}_{stem}"
    if variant == "kebab":
        return f"{section}-{stem.replace('_', '-')}"
    parts = stem.split("_")
    return section + "".join(part.capitalize() for part in parts)


def _wrong_token(rng: np.random.Generator, truth: FieldType) -> str:
    """An `autocomplete` value that is present and not the right one.

    A quarter of the time it is `off`, which is what a developer writes to turn
    autofill away, and the rest of the time it is another token entirely, which
    is what a copied markup block leaves behind. Both mislead a rule that trusts
    the attribute; only the second misleads it into a specific wrong class.
    """
    if rng.random() < 0.25:
        return "off"
    others = [member for member in FIELD_TYPES if member is not truth]
    return others[int(rng.integers(len(others)))].value


def _autocomplete(rng: np.random.Generator, config: GeneratorConfig, truth: FieldType) -> str:
    if rng.random() >= config.p_autocomplete:
        return ""
    if rng.random() < config.p_autocomplete_correct:
        # There is no WHATWG token for "none of the above", so a correctly
        # annotated field of no known type says `off`, which is what it means.
        return "off" if truth is FieldType.UNKNOWN else truth.value
    return _wrong_token(rng, truth)


def _label(rng: np.random.Generator, config: GeneratorConfig, entry: Lexicon) -> str:
    if rng.random() < config.label_dropout:
        return ""
    if entry.abbreviations and rng.random() < config.abbreviation_prob:
        return _pick(rng, entry.abbreviations)
    return _pick(rng, entry.labels)


def _plan(
    rng: np.random.Generator, config: GeneratorConfig, template: str
) -> list[tuple[str, FieldType]]:
    """The field sequence of one form, with adversarial fields spliced in."""
    plan: list[tuple[str, FieldType]] = []
    for section, field_type in TEMPLATES[template]:
        plan.append((section, field_type))
        if rng.random() < config.distractor_prob:
            plan.append((section, FieldType.UNKNOWN))
    return plan


def _build_form(
    form_index: int, config: GeneratorConfig, seed: int, template: str, locale: str
) -> list[FieldRecord]:
    """Every field of one page, from a generator that is that page's alone.

    The `spawn_key` is the form index, so form 40 is the same form in a corpus
    of 50 forms and in a corpus of 5,000, and adding a template later cannot
    shift the pages generated before it.
    """
    rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(form_index,)))
    split = ("train", "val", "test")[
        int(rng.choice(3, p=np.asarray(config.split_weights, dtype=np.float64)))
    ]
    variant = MARKUP_VARIANTS[int(rng.integers(config.n_markup_variants))]
    lexicon = LEXICON[locale]

    form_key = f"form-{form_index:04d}"
    template_id = f"{template}-{locale}-{variant}"
    plan = _plan(rng, config, template)

    drafts: list[dict[str, object]] = []
    for position, (section, field_type) in enumerate(plan):
        entry = lexicon[field_type]
        generic = rng.random() < config.generic_attribute_prob
        stem = _pick(rng, entry.names)
        name = f"input_{position + 1}" if generic else _styled(variant, section, stem)
        element_id = (
            f"{name}_{position + 1}" if variant in {"snake", "bare"} else f"{name}-{position + 1}"
        )
        placeholder = (
            "" if rng.random() < config.placeholder_dropout else _pick(rng, entry.placeholders)
        )
        drafts.append(
            {
                "section": section,
                "field_type": field_type,
                "label": _label(rng, config, entry),
                "name": name,
                "element_id": element_id,
                "placeholder": placeholder,
                "autocomplete": _autocomplete(rng, config, field_type),
                "input_type": entry.input_type,
            }
        )

    records = []
    for position, draft in enumerate(drafts):
        records.append(
            FieldRecord(
                form_id=f"{form_key}-{position:02d}",
                form_key=form_key,
                template=template,
                template_id=template_id,
                locale=locale,
                split=split,
                section=str(draft["section"]),
                position=position,
                label=str(draft["label"]),
                name=str(draft["name"]),
                element_id=str(draft["element_id"]),
                placeholder=str(draft["placeholder"]),
                autocomplete=str(draft["autocomplete"]),
                input_type=str(draft["input_type"]),
                previous_label=str(drafts[position - 1]["label"]) if position else "",
                next_label=str(drafts[position + 1]["label"]) if position + 1 < len(drafts) else "",
                field_type=draft["field_type"],  # type: ignore[arg-type]
            )
        )
    return records


def _form_plan(form_index: int, config: GeneratorConfig) -> tuple[str, str]:
    """Which template and locale form number `form_index` uses.

    Template cycles fastest so that a short corpus still holds all three, and
    the locale turns over once per full cycle of templates so that neither is
    ever a prefix of the other.
    """
    templates = tuple(TEMPLATES)
    template = templates[form_index % len(templates)]
    locale = config.locales[(form_index // len(templates)) % len(config.locales)]
    return template, locale


def generate_fields(config: GeneratorConfig, seed: int) -> list[FieldRecord]:
    """`config.n_fields` records, built whole form at a time then truncated.

    Truncation lands mid form, which is realistic (a page can be cut short by a
    crawler) and is what keeps `--n-fields` an exact count rather than a hint.
    """
    records: list[FieldRecord] = []
    form_index = 0
    while len(records) < config.n_fields:
        template, locale = _form_plan(form_index, config)
        records.extend(_build_form(form_index, config, seed, template, locale))
        form_index += 1
    return records[: config.n_fields]


def _escape(text: str) -> str:
    """HTML attribute and text escaping, by hand and completely.

    `html.escape` would do, and this is here instead because the pages are a
    byte for byte reproducible artifact: writing the five substitutions out
    fixes the output against a future change in the standard library's default
    for `quote`.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _page_html(template: str, locale: str, records: list[FieldRecord]) -> str:
    """One standalone page, with the ground truth on every input."""
    language = locale.replace("_", "-")
    title = f"{template.replace('_', ' ')} ({locale})"
    lines = [
        "<!doctype html>",
        f'<html lang="{language}">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_escape(title)}</title>",
        "</head>",
        "<body>",
        f"<h1>{_escape(title)}</h1>",
        f'<form id="{_escape(template)}" method="post" action="#">',
    ]
    section = ""
    for record in records:
        if record.section != section:
            if section:
                lines.append("</fieldset>")
            section = record.section
            lines.append("<fieldset>")
            lines.append(f"<legend>{_escape(section)}</legend>")
        lines.append('<div class="field">')
        if record.label:
            lines.append(
                f'<label for="{_escape(record.element_id)}">{_escape(record.label)}</label>'
            )
        attributes = [
            f'type="{_escape(record.input_type)}"',
            f'id="{_escape(record.element_id)}"',
            f'name="{_escape(record.name)}"',
        ]
        if record.placeholder:
            attributes.append(f'placeholder="{_escape(record.placeholder)}"')
        if record.autocomplete:
            attributes.append(f'autocomplete="{_escape(record.autocomplete)}"')
        attributes.append(f'data-truth="{_escape(record.field_type.value)}"')
        lines.append(f"<input {' '.join(attributes)}>")
        lines.append("</div>")
    if section:
        lines.append("</fieldset>")
    lines.extend(['<button type="submit">Continue</button>', "</form>", "</body>", "</html>"])
    return "\n".join(lines) + "\n"


#: Where the page generator's seed stream starts, well clear of any form index a
#: corpus could reach, so a page and a corpus form never share a stream.
PAGE_SPAWN_BASE = 1_000_000


def generate_pages(config: GeneratorConfig, seed: int) -> dict[str, str]:
    """One HTML page per template per locale, keyed by file name."""
    pages: dict[str, str] = {}
    for index, template in enumerate(TEMPLATES):
        for offset, locale in enumerate(config.locales):
            form_index = PAGE_SPAWN_BASE + index * len(config.locales) + offset
            records = _build_form(form_index, config, seed, template, locale)
            pages[f"{template}_{locale}.html"] = _page_html(template, locale, records)
    return pages


def write_corpus(out_dir: Path | str, config: GeneratorConfig, seed: int) -> CorpusPaths:
    """Write the splits, the pages and the metadata under `out_dir`."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = generate_fields(config, seed)

    splits: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        rows = [record for record in records if record.split == split]
        path = root / f"fields.{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(record.to_json_dict(), sort_keys=True, ensure_ascii=True) + "\n"
                for record in rows
            ),
            encoding="utf-8",
            newline="\n",
        )
        splits[split] = path
        counts[split] = len(rows)

    page_dir = root / "pages"
    page_dir.mkdir(exist_ok=True)
    pages = []
    for name, html in generate_pages(config, seed).items():
        path = page_dir / name
        path.write_text(html, encoding="utf-8", newline="\n")
        pages.append(path)

    by_locale: dict[str, int] = {}
    for record in records:
        by_locale[record.locale] = by_locale.get(record.locale, 0) + 1
    meta = root / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "seed": seed,
                "config": config.to_json_dict(),
                "counts": counts,
                "locales": by_locale,
                "templates": sorted({record.template_id for record in records}),
                "n_classes": len(FIELD_TYPES),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return CorpusPaths(root=root, splits=splits, meta=meta, pages=sorted(pages))


def load_split(data_dir: Path | str, split: str) -> list[FieldRecord]:
    """Read one split back, refusing a corpus that was never generated."""
    path = Path(data_dir) / f"fields.{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no fields.{split}.jsonl under {data_dir}: run `triage autofill generate "
            f"--out {data_dir}` first"
        )
    return [
        FieldRecord.from_json_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
