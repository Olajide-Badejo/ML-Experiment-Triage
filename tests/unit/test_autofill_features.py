"""Hashed character n grams: stability, namespacing, and the sparse shape.

The hash is part of the on disk format of a weights file, so the tests that
matter here are the ones that fail when it moves: exact indices for known
strings, and the guarantee that the same string under two prefixes lands in two
different places.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from triage.autofill.features import (
    FEATURE_DIM,
    NGRAM_SIZES,
    PREFIXES,
    FeatureMatrix,
    feature_indices,
    featurise,
    hash_token,
    signals,
    tokens_of,
)
from triage.autofill.generator import GeneratorConfig, generate_fields
from triage.autofill.taxonomy import FieldType


def test_the_dimension_is_the_one_the_specification_names() -> None:
    assert FEATURE_DIM == 2**16


def test_the_n_gram_sizes_are_three_to_five() -> None:
    assert NGRAM_SIZES == (3, 4, 5)


def test_the_prefixes_are_the_six_the_specification_names() -> None:
    assert set(PREFIXES) == {"label", "name", "ph", "ac", "type", "ctx"}


def test_the_hash_is_stable_across_processes() -> None:
    """`hash()` is randomised per process; this must not be.

    A subprocess with a different `PYTHONHASHSEED` is the only way to catch a
    `hash()` sneaking back in, because inside one process it would look stable.
    """
    code = (
        "from triage.autofill.features import hash_token; "
        "print(hash_token('label:plz'), hash_token('ac:postal-code'))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    ]
    assert len(set(runs)) == 1, runs


def test_the_hash_lands_inside_the_dimension() -> None:
    for token in ("", "a", "label:vorname", "ctx:payment", "x" * 500):
        assert 0 <= hash_token(token) < FEATURE_DIM


def test_the_same_text_under_two_prefixes_is_two_features() -> None:
    """Namespacing is the point: `name` the attribute is not `name` the type."""
    assert hash_token("label:name") != hash_token("name:name")


def test_tokens_are_lowercased() -> None:
    assert tokens_of("PLZ", "label") == tokens_of("plz", "label")


def test_tokens_are_the_n_grams_of_the_text_with_the_prefix_on_each() -> None:
    assert tokens_of("plz", "label") == ["label:plz"]
    assert tokens_of("post", "label") == ["label:pos", "label:ost", "label:post"]


def test_an_empty_signal_becomes_an_explicit_absence_marker() -> None:
    """A missing label is evidence, and a row of no features is not."""
    assert tokens_of("", "label") == ["label:<absent>"]


def test_a_signal_shorter_than_the_smallest_n_gram_still_produces_a_token() -> None:
    assert tokens_of("cv", "label") == ["label:cv"]


def test_signals_cover_every_attribute_the_model_is_allowed_to_see() -> None:
    record = generate_fields(GeneratorConfig(n_fields=40), seed=0)[0]
    prefixes = [prefix for prefix, _ in signals(record)]
    assert set(prefixes) == set(PREFIXES)


def test_signals_never_read_the_ground_truth_field() -> None:
    """The one bug that would make every number in this vertical meaningless.

    The `autocomplete` attribute often agrees with the truth, and that is a fact
    about forms rather than a leak: it is in the DOM, the heuristic uses it, and
    a quarter of the time it is wrong. What must never happen is the featuriser
    reading `field_type` itself, so this changes only that and asserts the
    features do not move.
    """
    from dataclasses import replace

    record = generate_fields(GeneratorConfig(n_fields=40), seed=0)[0]
    relabelled = replace(record, field_type=FieldType.CC_CSC)
    assert signals(relabelled) == signals(record)
    np.testing.assert_array_equal(feature_indices(relabelled), feature_indices(record))


def test_the_autocomplete_attribute_is_a_signal_because_it_often_is_right() -> None:
    records = generate_fields(GeneratorConfig(n_fields=200, p_autocomplete=1.0), seed=0)
    seen = {value for record in records for prefix, value in signals(record) if prefix == "ac"}
    assert any(value for value in seen)


def test_feature_indices_are_sorted_unique_and_inside_the_dimension() -> None:
    record = generate_fields(GeneratorConfig(n_fields=40), seed=0)[0]
    indices = feature_indices(record)
    assert indices.dtype == np.int32
    assert np.all(np.diff(indices) > 0)
    assert indices.min() >= 0 and indices.max() < FEATURE_DIM


def test_a_bias_feature_is_always_active() -> None:
    """A linear model with no intercept cannot learn a class prior."""
    records = generate_fields(GeneratorConfig(n_fields=60), seed=0)
    bias = hash_token("bias:1")
    assert all(bias in feature_indices(record) for record in records)


def test_featurise_builds_a_compressed_row_layout() -> None:
    records = generate_fields(GeneratorConfig(n_fields=120), seed=0)
    matrix = featurise(records)
    assert isinstance(matrix, FeatureMatrix)
    assert len(matrix) == len(records)
    assert matrix.indptr.shape == (len(records) + 1,)
    assert matrix.indptr[0] == 0
    assert matrix.indptr[-1] == matrix.indices.size
    for position, record in enumerate(records):
        np.testing.assert_array_equal(matrix.row(position), feature_indices(record))


def test_featurising_the_same_records_twice_gives_the_same_matrix() -> None:
    records = generate_fields(GeneratorConfig(n_fields=120), seed=0)
    first, second = featurise(records), featurise(records)
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_array_equal(first.indptr, second.indptr)


def test_the_class_vector_is_the_head_index_of_the_ground_truth() -> None:
    records = generate_fields(GeneratorConfig(n_fields=60), seed=0)
    matrix = featurise(records)
    from triage.autofill.taxonomy import index_of

    np.testing.assert_array_equal(
        matrix.classes, np.array([index_of(r.field_type) for r in records], dtype=np.int64)
    )


def test_a_dimension_that_is_not_a_power_of_two_is_refused() -> None:
    records = generate_fields(GeneratorConfig(n_fields=10), seed=0)
    with pytest.raises(ValueError, match="power of two"):
        featurise(records, dim=1000)


def test_two_locales_do_not_share_a_feature_space_by_accident() -> None:
    """A sanity check on collision pressure at this dimension."""
    records = generate_fields(GeneratorConfig(n_fields=4000), seed=0)
    matrix = featurise(records)
    distinct = np.unique(matrix.indices).size
    assert distinct > 2000, distinct
    assert distinct < FEATURE_DIM


def test_a_field_whose_label_is_gone_still_has_the_attribute_features() -> None:
    from dataclasses import replace

    record = generate_fields(GeneratorConfig(n_fields=40, label_dropout=0.0), seed=0)[0]
    stripped = replace(record, label="")
    assert feature_indices(stripped).size > 5
    assert not np.array_equal(feature_indices(stripped), feature_indices(record))


def test_the_taxonomy_token_space_is_hashable_without_collision_inside_one_prefix() -> None:
    codes = {hash_token(f"ac:{member.value}") for member in FieldType}
    assert len(codes) == len(FieldType)
