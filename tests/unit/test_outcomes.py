"""Step free evaluation rows: the shape the consumer's file actually has (E5).

`Autofill_audit` writes one row per classified field, not a time series. Their
bridge handed that file to `JsonlParser` as a probe and recorded the refusal in
`analysis.json` as a checkable finding, which was the right thing to do with a
tool that had no shape for it. These tests hold the new shape to three claims:

* their file, in the schema they filed, verbatim, ingests;
* `paired_permutation` over `correct` clustered by `template_id` produces a p
  value from it, which is the comparison they asked for;
* the windowed modes still refuse an `Outcomes`, by construction and by name,
  because a final window of an arbitrary row order is a number with no referent.

The second producer's shape (TPT's `sweep_results.jsonl`, one row per config) is
covered here too: it is outcomes shaped, joined on `config_key`, and it needs
`--outcomes` because it declares no record schema on its rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triage.analysis.comparison import (
    MODE_PAIRED_CLUSTER,
    ComparisonConfig,
    ComparisonError,
    compare,
    compare_seed_replicated,
    compare_window_block,
    paired_permutation,
    window_statistic,
)
from triage.core import Store
from triage.core.experiment import SeriesError
from triage.core.outcomes import Outcomes
from triage.ingest import ingest
from triage.parsers import DEFAULT_PARSERS, JsonlParser, OutcomesParser, ParseError, parsers_for

#: The consumer's schema, key for key, as filed in their issue. Written out here
#: rather than read off the fixture: if the fixture drifts, this list is what
#: notices, and drift is precisely the failure the contract exists to catch.
CONSUMER_KEYS = (
    "schema_version",
    "run_id",
    "engine",
    "split",
    "form_id",
    "template_id",
    "true_label",
    "pred_label",
    "correct",
    "confidence",
    "latency_us",
)


@pytest.fixture
def consumer_run(fixture_root: Path) -> Path:
    return fixture_root / "outcomes" / "autofill_run"


@pytest.fixture
def outcomes(consumer_run: Path) -> Outcomes:
    return OutcomesParser().parse(consumer_run)


# ------------------------------------------------------- the fixture is theirs


def test_the_fixture_is_the_consumers_schema_key_for_key(consumer_run: Path) -> None:
    lines = (consumer_run / "run.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines, "the fixture must not be empty"
    for line in lines:
        assert tuple(json.loads(line)) == CONSUMER_KEYS


# --------------------------------------------------------------- the parser


def test_the_consumer_schema_is_recognised_without_being_asked(consumer_run: Path) -> None:
    """Their `probe_native_ingestion` flips from refused to supported."""
    assert OutcomesParser().can_parse(consumer_run)
    # And the JSONL parser still claims it first by suffix and still cannot read
    # it, which is why the outcomes parser sits ahead of it in DEFAULT_PARSERS.
    assert JsonlParser().can_parse(consumer_run)
    with pytest.raises(ParseError, match="no record carries a step field"):
        JsonlParser().parse(consumer_run)
    assert [type(parser).__name__ for parser in DEFAULT_PARSERS].index("OutcomesParser") < [
        type(parser).__name__ for parser in DEFAULT_PARSERS
    ].index("JsonlParser")


def test_the_rows_split_into_measurements_and_group_keys(outcomes: Outcomes) -> None:
    assert outcomes.n_rows == 48
    assert outcomes.field_names == ["confidence", "correct", "latency_us"]
    assert outcomes.group_names == [
        "engine",
        "form_id",
        "pred_label",
        "run_id",
        "schema_version",
        "split",
        "template_id",
        "true_label",
    ]
    # `correct` is a boolean in the file and a zero or one here, because its mean
    # is an accuracy and a statistic cannot run over `True`.
    correct = outcomes.field("correct")
    assert correct.dtype == np.float64
    assert set(np.unique(correct)) <= {0.0, 1.0}


def test_a_column_that_is_numeric_in_some_rows_only_is_a_group_key(tmp_path: Path) -> None:
    """Schema inference reads every row, not `records[0]` (the E6 defect)."""
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"schema_version": "1", "unit": "a", "score": 1.0, "note": 3},
                {"schema_version": "1", "unit": "b", "score": 2.0, "note": "retry"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = OutcomesParser().parse(path)
    assert parsed.field_names == ["score"]
    assert "note" in parsed.group_names


def test_a_missing_measurement_is_a_hole_rather_than_a_zero(tmp_path: Path) -> None:
    path = tmp_path / "holes.jsonl"
    path.write_text(
        '{"schema_version": "1", "unit": "a", "score": 1.0}\n'
        '{"schema_version": "1", "unit": "b", "score": null}\n',
        encoding="utf-8",
    )
    scores = OutcomesParser().parse(path).field("score")
    assert scores[0] == 1.0
    assert np.isnan(scores[1]), "a null is not measured, and it is certainly not zero"


def test_a_file_with_a_step_is_not_outcomes(fixture_root: Path) -> None:
    """A training log must never be read as a set of rows to pair."""
    training_run = fixture_root / "jsonl" / "jsonl_run"
    assert not OutcomesParser().can_parse(training_run)
    assert not OutcomesParser(strict=False).can_parse(training_run)
    with pytest.raises(ParseError, match="time series rather than a set of outcomes"):
        OutcomesParser(strict=False).parse(training_run)


def test_an_undeclared_step_free_file_needs_the_flag(fixture_root: Path) -> None:
    """Strict recognition refuses to guess; `--outcomes` is the instruction."""
    sweep = fixture_root / "tpt_sweep" / "sweep_results"
    assert not OutcomesParser().can_parse(sweep), "no schema_version on the rows"
    assert OutcomesParser(strict=False).can_parse(sweep)
    assert [type(p).__name__ for p in parsers_for(outcomes=True)] == [
        type(p).__name__ for p in parsers_for(outcomes=False)
    ]
    assert parsers_for(outcomes=True)[2].strict is False
    assert parsers_for(outcomes=False)[2].strict is True


# ------------------------------------------------------------------ pairing


def test_pair_on_matches_the_two_engines_field_by_field(outcomes: Outcomes) -> None:
    rules, ngram = outcomes.pair_on("form_id", "engine", "rules", "ngram")
    assert rules.n_rows == ngram.n_rows == 24
    np.testing.assert_array_equal(rules.group("form_id"), ngram.group("form_id"))
    np.testing.assert_array_equal(rules.group("template_id"), ngram.group("template_id"))
    assert set(rules.group("engine")) == {"rules"}
    assert set(ngram.group("engine")) == {"ngram"}


def test_pair_on_refuses_a_unit_that_is_not_on_both_sides(outcomes: Outcomes) -> None:
    """Dropping the unmatched rows in silence would be losing evidence."""
    keep = np.flatnonzero(
        ~((outcomes.group("engine") == "rules") & (outcomes.group("form_id") == "form-00-00"))
    )
    lopsided = outcomes.take(keep)
    with pytest.raises(SeriesError, match="have no pair"):
        lopsided.pair_on("form_id", "engine", "rules", "ngram")


def test_pair_on_refuses_a_unit_key_that_is_not_unique(outcomes: Outcomes) -> None:
    with pytest.raises(SeriesError, match="not unique"):
        outcomes.pair_on("template_id", "engine", "rules", "ngram")


def test_pair_on_names_the_values_it_can_see(outcomes: Outcomes) -> None:
    with pytest.raises(SeriesError, match="values present: ngram, rules"):
        outcomes.pair_on("form_id", "engine", "rules", "transformer")


# ------------------------------------------- the comparison they asked for


def test_paired_permutation_over_correct_clustered_by_template(outcomes: Outcomes) -> None:
    """E5's acceptance criterion, end to end from their file.

    Clusters are form templates, so whole templates swap together and the p
    value rests on six independent units rather than twenty four correlated
    ones. Six clusters enumerate exhaustively, so the p value is exact and its
    floor is 2 / 2**6 = 0.031.
    """
    rules, ngram = outcomes.pair_on("form_id", "engine", "rules", "ngram")
    result = paired_permutation(
        rules.field("correct"),
        ngram.field("correct"),
        statistic=np.mean,
        clusters=ngram.group("template_id"),
        higher_is_better=True,
    )
    assert result.mode == MODE_PAIRED_CLUSTER
    assert 0.0 < result.p_value <= 1.0
    assert result.exact, "six clusters enumerate exhaustively"
    assert result.min_attainable_p == pytest.approx(2.0 / 2**6)
    assert result.n_baseline == result.n_candidate == 24
    assert result.effect > 0, "the model is built to beat the rules engine here"
    assert any("6 clusters" in warning or "clusters" in warning for warning in result.warnings)
    # The p value must not depend on the order the rows arrived in.
    order = np.array([23 - index for index in range(24)])
    reversed_result = paired_permutation(
        rules.take(order).field("correct"),
        ngram.take(order).field("correct"),
        statistic=np.mean,
        clusters=ngram.take(order).group("template_id"),
    )
    assert reversed_result.p_value == pytest.approx(result.p_value)


def test_the_tpt_sweep_shape_pairs_on_config_key(fixture_root: Path) -> None:
    """E6's tail: one row per configuration, joined on `config_key`."""
    sweep = OutcomesParser(strict=False).parse(fixture_root / "tpt_sweep" / "sweep_results")
    assert sweep.n_rows == 16
    assert "throughput_samples_per_s" in sweep.field_names
    assert "config_key" in sweep.group_names

    eager, compiled = sweep.pair_on("config_key", "runner", "eager", "compiled")
    assert eager.n_rows == compiled.n_rows == 8
    result = paired_permutation(
        eager.field("throughput_samples_per_s"),
        compiled.field("throughput_samples_per_s"),
        statistic=np.mean,
        clusters=compiled.group("config_key"),
        higher_is_better=True,
    )
    assert 0.0 < result.p_value <= 1.0
    assert result.effect > 0, "the compiled runner is generated faster"


# --------------------------------------------------- the refusal by construction


def a_windowed_call(name: str, outcomes: Outcomes):
    calls = {
        "window_statistic": lambda: window_statistic(outcomes, ComparisonConfig()),
        "compare_window_block": lambda: compare_window_block(outcomes, outcomes, "correct"),
        "compare_seed_replicated": lambda: compare_seed_replicated(
            [outcomes, outcomes], [outcomes, outcomes], "correct"
        ),
        "compare": lambda: compare([outcomes], [outcomes], "correct"),
    }
    return calls[name]


@pytest.mark.parametrize(
    "entry_point",
    ["window_statistic", "compare_window_block", "compare_seed_replicated", "compare"],
)
def test_a_windowed_mode_refuses_outcomes_and_names_the_right_tool(
    outcomes: Outcomes, entry_point: str
) -> None:
    """E5's refusal half, which matters as much as the acceptance half.

    A window statistic over rows with no step axis is a mean of an arbitrary
    ordering: it would compute, it would look like a number, and it would mean
    nothing. Refusing is the only honest answer, and the message has to name the
    test that IS right for this data or the refusal is a dead end.
    """
    with pytest.raises(ComparisonError, match="paired_permutation") as error:
        a_windowed_call(entry_point, outcomes)()
    assert "no step axis" in str(error.value)


# --------------------------------------------------------------- through ingest


def test_the_consumer_file_ingests_through_the_cli_path(
    fixture_root: Path, temp_database: Path
) -> None:
    """ "Ingests" means what it says: into the database, and back out again."""
    with Store(temp_database) as store:
        result = ingest(fixture_root / "outcomes", store, show_progress=False)
        assert result.added == ["autofill_run"]
        assert result.outcome_runs == ["autofill_run"]
        assert store.run_ids() == [], "an outcomes file is not a training run"
        assert store.outcome_run_ids() == ["autofill_run"]
        loaded = store.load_outcomes("autofill_run")

    original = OutcomesParser().parse(fixture_root / "outcomes" / "autofill_run")
    assert loaded.n_rows == original.n_rows
    assert loaded.field_names == original.field_names
    assert loaded.group_names == original.group_names
    for name in original.field_names:
        np.testing.assert_array_equal(loaded.field(name), original.field(name))
    for name in original.group_names:
        np.testing.assert_array_equal(loaded.group(name), original.group(name))


def test_reingesting_outcomes_skips_the_unchanged_file(
    fixture_root: Path, temp_database: Path
) -> None:
    with Store(temp_database) as store:
        ingest(fixture_root / "outcomes", store, show_progress=False)
        second = ingest(fixture_root / "outcomes", store, show_progress=False)
    assert second.skipped == ["autofill_run"]
    assert second.added == []


def test_the_undeclared_sweep_ingests_only_with_the_outcomes_parsers(
    fixture_root: Path, temp_database: Path
) -> None:
    root = fixture_root / "tpt_sweep"
    with Store(temp_database) as store:
        strict = ingest(root, store, show_progress=False)
        assert strict.added == []
        assert [name for name, _ in strict.failed] == ["sweep_results"]

    with Store(temp_database) as store:
        relaxed = ingest(root, store, parsers=parsers_for(outcomes=True), show_progress=False)
        assert relaxed.added == ["sweep_results"]
        assert store.load_outcomes("sweep_results").n_rows == 16


def test_the_outcomes_flag_is_what_the_command_line_calls_it(
    fixture_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`triage ingest --outcomes` (Section 3.4), end to end and both ways round."""
    from triage.cli import EXIT_OK, EXIT_RUNS_FAILED, main

    database = tmp_path / "triage.db"
    argv = ["ingest", str(fixture_root / "tpt_sweep"), "--database", str(database), "--quiet"]

    assert main(argv) == EXIT_RUNS_FAILED, "undeclared and unasked for: refused, and said so"
    capsys.readouterr()

    assert main([*argv, "--outcomes"]) == EXIT_OK
    printed = capsys.readouterr()
    assert "outcome files" in printed.out
    assert "paired_permutation" in printed.err, "the summary must name the test that fits"
    with Store(database) as store:
        assert store.outcome_run_ids() == ["sweep_results"]
        assert store.run_ids() == []
