"""Hashed character n grams over the signals a form field actually exposes.

**Why character n grams rather than words.** The vocabulary of a form is open,
multilingual, and full of things that are not words: `plz`, `cc_exp_month`,
`billingPostalCode`, `Str.`. German compounds mean the useful unit is a
substring rather than a token, which is exactly what `postleitzahl` and
`plz` sharing nothing at the word level and `strasse` inside
`strasse_hausnummer` sharing everything at the character level looks like. Three
to five characters is the range the specification names and the range that
covers both a short abbreviation and a compound stem.

**Why hashing rather than a vocabulary.** A learned vocabulary is a second
artifact to persist, to version, and to get out of step with a weights file. A
hash is a pure function: the weights file plus this module's constants is the
whole model, and a feature never seen in training still lands somewhere
sensible at evaluation time. The cost is collisions, which at 2^16 slots and a
few thousand live features is small and, more importantly, is measured rather
than assumed (a test asserts the live count).

**The hash is `zlib.crc32`, never `hash()`.** Python randomises string hashing
per process, so a model trained in one process would score differently in the
next, and nothing would say so. crc32 is fixed by its polynomial, is in the
standard library, and is fast. It is not a cryptographic hash and does not need
to be: nothing here is a security boundary.

**Namespacing.** Every signal is prefixed before hashing, so `name` the
attribute value and `name` the field type token cannot collide by construction,
and the model can learn that a string in the `autocomplete` attribute means
something different from the same string in a placeholder.

**Two features that are not n grams.** An absent signal emits one explicit
marker (`label:<absent>`), because a missing label is evidence rather than the
absence of evidence, and a row of no features at all is not something a linear
model can say anything about. A constant `bias:1` token gives the model its
intercept without a separate dense column. Both are hashed into the same space,
so the dimension stays exactly 2^16 as specified.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from triage.autofill.generator import FieldRecord
from triage.autofill.taxonomy import index_of

#: The hash space. A power of two so the fold is a mask rather than a modulo.
FEATURE_DIM = 2**16

#: Character n gram lengths, per 5.2.
NGRAM_SIZES: tuple[int, ...] = (3, 4, 5)

#: The six signal namespaces, and what each one is. The prefix is written into
#: the token before hashing, so these strings are part of the model format.
PREFIXES: dict[str, str] = {
    "label": "the visible <label> text, when there is one",
    "name": "the name and id attributes",
    "ph": "the placeholder text",
    "ac": "the autocomplete attribute",
    "type": "the input type attribute",
    "ctx": "the section and the two neighbouring labels",
}

#: What an empty signal emits instead of nothing.
ABSENT = "<absent>"

#: The intercept, as a token so it costs no special case anywhere else.
BIAS_TOKEN = "bias:1"


def hash_token(token: str) -> int:
    """A stable slot for one namespaced token.

    `zlib.crc32` rather than `hash()`: see the module docstring. The mask rather
    than a modulo because the dimension is a power of two, which `featurise`
    enforces.
    """
    return zlib.crc32(token.encode("utf-8")) & (FEATURE_DIM - 1)


def tokens_of(text: str, prefix: str) -> list[str]:
    """The namespaced character n grams of one signal, in a stable order.

    Short text is emitted whole rather than dropped: `plz`, `cvc` and `mm` are
    among the most informative signals in the corpus and all three are shorter
    than the largest n gram.
    """
    lowered = text.strip().lower()
    if not lowered:
        return [f"{prefix}:{ABSENT}"]
    grams: list[str] = []
    for size in NGRAM_SIZES:
        if len(lowered) < size:
            continue
        for start in range(len(lowered) - size + 1):
            grams.append(f"{prefix}:{lowered[start : start + size]}")
    if not grams:
        # Shorter than the smallest n gram: the whole string is the feature.
        grams.append(f"{prefix}:{lowered}")
    return grams


def signals(record: FieldRecord) -> list[tuple[str, str]]:
    """`(prefix, text)` for everything the classifier is allowed to look at.

    `field_type` is deliberately absent, and a test asserts that changing it
    changes nothing here. The `autocomplete` attribute IS here, and its being
    right most of the time is a property of the corpus rather than a leak: it is
    in the DOM, the heuristic baseline reads it, and it is wrong often enough
    that trusting it alone is a losing strategy.
    """
    return [
        ("label", record.label),
        ("name", record.name),
        ("name", record.element_id),
        ("ph", record.placeholder),
        ("ac", record.autocomplete),
        ("type", record.input_type),
        ("ctx", record.section),
        ("ctx", record.previous_label),
        ("ctx", record.next_label),
    ]


def feature_indices(record: FieldRecord) -> np.ndarray:
    """The sorted, unique slots this field switches on.

    Unique rather than counted: a repeated n gram in a long placeholder is not
    three times the evidence, and a binary indicator keeps the forward pass a
    gather and a sum rather than a weighted one.
    """
    codes = [hash_token(BIAS_TOKEN)]
    for prefix, text in signals(record):
        codes.extend(hash_token(token) for token in tokens_of(text, prefix))
    return np.unique(np.asarray(codes, dtype=np.int32))


@dataclass(frozen=True)
class FeatureMatrix:
    """A compressed sparse row layout, and the class each row belongs to.

    `indices[indptr[i]:indptr[i + 1]]` are row `i`'s active slots. This is the
    shape scipy's CSR uses, written out by hand so that the core install does
    not have to carry `scipy.sparse` for a matrix of ones: there are no values
    to store, because every feature is present or absent.
    """

    indptr: np.ndarray
    indices: np.ndarray
    classes: np.ndarray
    n_features: int

    def __len__(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def n_rows(self) -> int:
        return len(self)

    def row(self, position: int) -> np.ndarray:
        start, stop = int(self.indptr[position]), int(self.indptr[position + 1])
        return self.indices[start:stop]

    def take(self, rows: np.ndarray) -> FeatureMatrix:
        """The same matrix restricted to `rows`, in the order given.

        Two callers need this on a hot path: the trainer reshuffles the whole
        training split once per epoch, and the bootstrap in `evaluate.py` draws
        a resample with replacement. Both would be a Python loop over rows if
        this were written the obvious way, so the gather is built vectorised: the
        destination row of every nonzero, then its offset inside that row, then
        one fancy index into the source. Rows may repeat, which is what makes it
        usable for a resample rather than only for a permutation.
        """
        rows = np.asarray(rows, dtype=np.int64)
        lengths = self.indptr[rows + 1] - self.indptr[rows]
        indptr = np.zeros(rows.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=indptr[1:])
        destination_row = np.repeat(np.arange(rows.size, dtype=np.int64), lengths)
        within_row = np.arange(int(indptr[-1]), dtype=np.int64) - indptr[destination_row]
        source = self.indptr[rows][destination_row] + within_row
        return FeatureMatrix(
            indptr=indptr,
            indices=self.indices[source],
            classes=self.classes[rows],
            n_features=self.n_features,
        )


def featurise(records: Sequence[FieldRecord], dim: int = FEATURE_DIM) -> FeatureMatrix:
    """Hash a whole split into one sparse matrix and its class vector."""
    if dim != FEATURE_DIM:
        # The mask in `hash_token` is fixed to the module constant, so a caller
        # asking for another dimension would get features folded into the wrong
        # space and silently wrong numbers. One dimension, checked loudly.
        raise ValueError(
            f"the feature space is fixed at {FEATURE_DIM} (a power of two, folded with a "
            f"mask); got {dim}"
        )
    rows = [feature_indices(record) for record in records]
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([row.size for row in rows], out=indptr[1:], dtype=np.int64)
    indices = np.concatenate(rows).astype(np.int32) if rows else np.zeros(0, dtype=np.int32)
    classes = np.asarray([index_of(record.field_type) for record in records], dtype=np.int64)
    return FeatureMatrix(indptr=indptr, indices=indices, classes=classes, n_features=dim)
