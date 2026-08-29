"""Parser robustness: the inputs that real logging code actually produces.

Every test here pins a defect that was verified against v1.0.0. The theme is
that a parser must never invent a number, never lose one silently, and never
leak an exception type that is not part of its documented contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from triage.core.experiment import MetricSeries, SeriesError
from triage.parsers import CsvParser, JsonlParser, ParseError, TensorBoardParser

# --------------------------------------------------------------------- D1


def test_metric_series_drops_non_finite_points_and_counts_them() -> None:
    """NaN and inf poison every downstream statistic, so they never enter."""
    series = MetricSeries(
        tag="loss",
        steps=np.arange(6, dtype=np.int64),
        values=np.array([1.0, np.nan, 0.5, np.inf, 0.25, -np.inf], dtype=np.float32),
    )
    np.testing.assert_array_equal(series.steps, [0, 2, 4])
    np.testing.assert_allclose(series.values, [1.0, 0.5, 0.25], rtol=1e-6)
    assert series.dropped_non_finite == 3
    assert bool(np.isfinite(series.values).all())


def test_metric_series_drops_the_wall_time_of_a_dropped_point() -> None:
    series = MetricSeries(
        tag="loss",
        steps=np.arange(3, dtype=np.int64),
        values=np.array([1.0, np.nan, 0.5], dtype=np.float32),
        wall_times=np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    assert series.wall_times is not None
    np.testing.assert_array_equal(series.wall_times, [10.0, 12.0])


def test_metric_series_with_a_non_finite_step_is_rejected() -> None:
    with pytest.raises(SeriesError, match="non finite"):
        MetricSeries(
            tag="loss",
            steps=np.array([0.0, math.nan], dtype=np.float64),
            values=np.array([1.0, 2.0], dtype=np.float32),
        )


def _poisoned_wide_csv(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("metrics.csv").write_text(
        "step,loss\n0,1.0\n1,NaN\n2,inf\n3,-inf\n4,0.25\n", encoding="utf-8"
    )


def _poisoned_long_csv(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("metrics.csv").write_text(
        "step,tag,value\n0,loss,1.0\n1,loss,NaN\n2,loss,inf\n3,loss,-inf\n4,loss,0.25\n",
        encoding="utf-8",
    )


def _poisoned_jsonl(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0}\n'
        '{"step": 1, "loss": NaN}\n'
        '{"step": 2, "loss": Infinity}\n'
        '{"step": 3, "loss": -Infinity}\n'
        '{"step": 4, "loss": 0.25}\n',
        encoding="utf-8",
    )


def test_every_format_agrees_on_a_poisoned_run(tmp_path: Path) -> None:
    """The CHANGELOG claims formats are numerically identical; prove it under NaN."""
    _poisoned_wide_csv(tmp_path / "wide")
    _poisoned_long_csv(tmp_path / "long")
    _poisoned_jsonl(tmp_path / "jsonl")

    parsed = [
        CsvParser().parse(tmp_path / "wide").series("loss"),
        CsvParser().parse(tmp_path / "long").series("loss"),
        JsonlParser().parse(tmp_path / "jsonl").series("loss"),
    ]
    for series in parsed:
        np.testing.assert_array_equal(series.steps, [0, 4])
        np.testing.assert_allclose(series.values, [1.0, 0.25], rtol=1e-6)
        assert series.dropped_non_finite == 3


def test_dropped_counts_reach_the_experiment_metadata(tmp_path: Path) -> None:
    _poisoned_jsonl(tmp_path / "run")
    experiment = JsonlParser().parse(tmp_path / "run")
    assert experiment.metadata["dropped_non_finite"] == {"loss": 3}
    assert experiment.metadata["n_dropped_non_finite"] == 3


# --------------------------------------------------------------------- D6


def test_fractional_epochs_raise_rather_than_collapse(tmp_path: Path) -> None:
    """12 fractional epochs used to truncate to 3 integer steps, silently."""
    run = tmp_path / "run"
    run.mkdir()
    rows = "\n".join(f"{index * 0.25},{1.0 - index * 0.01}" for index in range(12))
    run.joinpath("metrics.csv").write_text(f"epoch,loss\n{rows}\n", encoding="utf-8")
    with pytest.raises(ParseError) as caught:
        CsvParser().parse(run)
    message = str(caught.value)
    assert "metrics.csv" in message
    assert "epoch" in message
    assert "integer" in message


def test_non_finite_json_step_is_a_parse_error_not_an_overflow(tmp_path: Path) -> None:
    """`int(Infinity)` raised OverflowError, which aborted the whole sweep."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0}\n{"step": Infinity, "loss": 0.5}\n', encoding="utf-8"
    )
    with pytest.raises(ParseError) as caught:
        JsonlParser().parse(run)
    assert "step" in str(caught.value)


def test_out_of_range_step_is_a_parse_error(tmp_path: Path) -> None:
    """1e30 cast to INT64_MIN and sorted to the front of the series."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0}\n{"step": 1e30, "loss": 0.5}\n', encoding="utf-8"
    )
    with pytest.raises(ParseError, match="range"):
        JsonlParser().parse(run)


def test_integral_float_steps_are_accepted(tmp_path: Path) -> None:
    """`2.0` is a step; `2.5` is not. Only the second is an error."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.csv").write_text(
        "epoch,loss\n0.0,1.0\n1.0,0.5\n2.0,0.25\n", encoding="utf-8"
    )
    series = CsvParser().parse(run).series("loss")
    np.testing.assert_array_equal(series.steps, [0, 1, 2])


# --------------------------------------------------------------------- D7


def test_multiple_csv_files_concatenate_instead_of_overwriting(tmp_path: Path) -> None:
    """part1 plus part2 used to yield part2 alone: 60% of the data gone."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("part1.csv").write_text("step,loss\n0,1.0\n1,0.9\n2,0.8\n", encoding="utf-8")
    run.joinpath("part2.csv").write_text("step,loss\n3,0.7\n4,0.6\n", encoding="utf-8")
    series = CsvParser().parse(run).series("loss")
    np.testing.assert_array_equal(series.steps, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(series.values, [1.0, 0.9, 0.8, 0.7, 0.6], rtol=1e-6)


def test_csv_accumulation_sorts_by_step_across_files(tmp_path: Path) -> None:
    """Alphabetical file order is not step order; the series must still sort."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("a_later.csv").write_text("step,loss\n3,0.7\n4,0.6\n", encoding="utf-8")
    run.joinpath("b_earlier.csv").write_text("step,loss\n0,1.0\n1,0.9\n", encoding="utf-8")
    series = CsvParser().parse(run).series("loss")
    np.testing.assert_array_equal(series.steps, [0, 1, 3, 4])
    np.testing.assert_allclose(series.values, [1.0, 0.9, 0.7, 0.6], rtol=1e-6)


# -------------------------------------------------------------------- D21a


def test_tensorboard_parser_ignores_a_file_merely_named_tfevents(tmp_path: Path) -> None:
    """`notes_tfevents.md` used to claim the run and hide the real metrics.csv."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("notes_tfevents.md").write_text("about the tfevents format\n", encoding="utf-8")
    run.joinpath("metrics.csv").write_text("step,loss\n0,1.0\n1,0.5\n", encoding="utf-8")
    assert not TensorBoardParser().can_parse(run)
    assert CsvParser().can_parse(run)


def test_tensorboard_parser_rejects_a_prefixed_file_with_a_bad_header(tmp_path: Path) -> None:
    """The name is necessary but not sufficient; the record header must check out."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("events.out.tfevents.1700000000.host").write_bytes(b"not a record stream")
    assert not TensorBoardParser().can_parse(run)


def test_tensorboard_parser_still_claims_the_real_fixture(fixture_root: Path) -> None:
    assert TensorBoardParser().can_parse(fixture_root / "tensorboard" / "tb_run")


# -------------------------------------------------------------- D21b and E6


def test_tpt_environment_header_line_does_not_break_schema_inference(
    fixture_root: Path,
) -> None:
    """TPT writes an environment header first; records[0] inference choked on it."""
    experiment = JsonlParser().parse(fixture_root / "tpt_jsonl" / "healthy_steps")
    series = experiment.series("loss")
    assert len(series) == 8
    assert experiment.metadata["records_without_step"] == 2


def test_multiple_headers_in_one_file_are_all_skipped(tmp_path: Path) -> None:
    """A resumed sweep writes a second environment header mid file."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"type": "environment", "run_id": "a", "row_schema_version": 2}\n'
        '{"step": 0, "loss": 1.0}\n'
        '{"type": "environment", "run_id": "a", "row_schema_version": 2}\n'
        '{"step": 1, "loss": 0.5}\n',
        encoding="utf-8",
    )
    experiment = JsonlParser().parse(run)
    np.testing.assert_array_equal(experiment.series("loss").steps, [0, 1])
    assert experiment.metadata["records_without_step"] == 2


def test_string_encoded_non_finite_literals_are_parsed_then_dropped(tmp_path: Path) -> None:
    """TPT writes "NaN"/"Infinity" as strings; they are values, then D1 drops them."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0}\n'
        '{"step": 1, "loss": "NaN"}\n'
        '{"step": 2, "loss": "Infinity"}\n'
        '{"step": 3, "loss": "-Infinity"}\n'
        '{"step": 4, "loss": 0.25}\n',
        encoding="utf-8",
    )
    series = JsonlParser().parse(run).series("loss")
    np.testing.assert_array_equal(series.steps, [0, 4])
    assert series.dropped_non_finite == 3


def test_arbitrary_strings_are_still_not_values(tmp_path: Path) -> None:
    """Only the three documented literals are numbers; a checkpoint path is not."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0, "ckpt": "step-0.pt"}\n'
        '{"step": 1, "loss": 0.5, "ckpt": "step-1.pt"}\n',
        encoding="utf-8",
    )
    assert JsonlParser().parse(run).tags == ["loss"]


def test_json_null_means_the_point_is_absent(tmp_path: Path) -> None:
    """`null` is "not measured", never zero."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"step": 0, "loss": 1.0, "val/acc": null}\n'
        '{"step": 1, "loss": null, "val/acc": 0.5}\n'
        '{"step": 2, "loss": 0.25, "val/acc": 0.6}\n',
        encoding="utf-8",
    )
    experiment = JsonlParser().parse(run)
    np.testing.assert_array_equal(experiment.series("loss").steps, [0, 2])
    np.testing.assert_array_equal(experiment.series("val/acc").steps, [1, 2])


def test_a_file_of_headers_alone_is_a_clear_parse_error(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"type": "environment", "run_id": "a"}\n', encoding="utf-8"
    )
    with pytest.raises(ParseError, match="step"):
        JsonlParser().parse(run)


# -------------------------------------------------------------------- D21c


@pytest.mark.parametrize(
    ("filename", "parser_class"),
    [("metrics.csv", CsvParser), ("metrics.jsonl", JsonlParser)],
)
def test_non_utf8_input_raises_parse_error_naming_the_offset(
    tmp_path: Path, filename: str, parser_class: type
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath(filename).write_bytes(b"step,loss\n0,1.0\n1,\xff\xfe bad\n")
    with pytest.raises(ParseError) as caught:
        parser_class().parse(run)
    message = str(caught.value)
    assert "byte" in message
    assert filename in message


def test_non_utf8_tensorboard_config_raises_parse_error(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("config.json").write_bytes(b'{"lr": 0.1, "note": "\xff\xfe"}')
    with pytest.raises(ParseError) as caught:
        TensorBoardParser().config_for(run)
    assert "byte" in str(caught.value)


# -------------------------------------------------------------------- D21e


def test_uppercase_extensions_are_matched_case_insensitively(tmp_path: Path) -> None:
    """`METRICS.CSV` parsed on Windows and was invisible on Linux."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("METRICS.CSV").write_text("step,loss\n0,1.0\n1,0.5\n", encoding="utf-8")
    assert CsvParser().can_parse(run)
    assert len(CsvParser().parse(run).series("loss")) == 2


def test_uppercase_jsonl_extension_is_matched(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("METRICS.JSONL").write_text(
        '{"step": 0, "loss": 1.0}\n{"step": 1, "loss": 0.5}\n', encoding="utf-8"
    )
    assert JsonlParser().can_parse(run)
    assert len(JsonlParser().parse(run).series("loss")) == 2


def test_uppercase_config_json_is_not_mistaken_for_a_metrics_file(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("CONFIG.JSON").write_text('{"learning_rate": 0.1}', encoding="utf-8")
    assert not JsonlParser().can_parse(run)


# -------------------------------------------------------------------- D21f


def test_long_form_record_without_a_tag_is_skipped_and_counted(tmp_path: Path) -> None:
    """A missing tag used to create a metric literally named `None`."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.jsonl").write_text(
        '{"step": 0, "tag": "loss", "value": 1.0}\n'
        '{"step": 1, "value": 0.5}\n'
        '{"step": 2, "tag": "loss", "value": 0.25}\n',
        encoding="utf-8",
    )
    experiment = JsonlParser().parse(run)
    assert experiment.tags == ["loss"]
    assert "None" not in experiment.metrics
    assert experiment.metadata["records_without_tag"] == 1


# -------------------------------------------------------------------- D21g


def test_extra_numeric_columns_in_a_long_csv_are_counted_and_warned(tmp_path: Path) -> None:
    """The docstring promised loudness; the code dropped them in silence."""
    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("metrics.csv").write_text(
        "step,tag,value,grad_norm,lr\n0,loss,1.0,3.5,0.1\n1,loss,0.5,3.4,0.1\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="grad_norm"):
        experiment = CsvParser().parse(run)
    assert experiment.tags == ["loss"]
    assert experiment.metadata["ignored_long_form_columns"] == {
        "metrics.csv": ["grad_norm", "lr"]
    }


# -------------------------------------------------------------------- D21i


def test_store_writes_standards_compliant_config_json(tmp_path: Path) -> None:
    """`NaN` is not JSON; a non finite config value must not become one."""
    from triage.core.experiment import Experiment
    from triage.core.store import Store

    experiment = Experiment(
        run_id="run",
        source_path="run",
        source_format="csv",
        config={"learning_rate": float("nan"), "warmup": float("inf"), "batch_size": 32},
    )
    database = tmp_path / "triage.db"
    with Store(database) as store:
        store.upsert(experiment, "hash")
        raw = store.connection.execute(
            "SELECT config_json FROM experiments WHERE run_id = 'run'"
        ).fetchone()["config_json"]

    assert "NaN" not in raw
    assert "Infinity" not in raw
    decoded = json.loads(raw)
    assert decoded["learning_rate"] is None
    assert decoded["warmup"] is None
    assert decoded["batch_size"] == 32


def test_store_round_trips_a_sanitized_config(tmp_path: Path) -> None:
    from triage.core.experiment import Experiment
    from triage.core.store import Store

    experiment = Experiment(
        run_id="run",
        source_path="run",
        source_format="csv",
        config={"learning_rate": float("nan")},
        metadata={"note": float("inf")},
    )
    database = tmp_path / "triage.db"
    with Store(database) as store:
        store.upsert(experiment, "hash")
        loaded = store.load("run")
    assert loaded.config["learning_rate"] is None
    assert loaded.metadata["note"] is None
