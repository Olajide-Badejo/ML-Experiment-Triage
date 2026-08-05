"""Every parser on its committed fixture, and all of them against each other.

The fixtures under `tests/fixtures` all encode one identical float32 series in
four different containers. That makes the strongest available assertion cheap:
not merely that each parser runs, but that a TensorBoard event file, a wide
CSV, a long CSV and a JSONL log produce numerically identical experiments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from triage.parsers import (
    DEFAULT_PARSERS,
    CsvParser,
    JsonlParser,
    ParseError,
    TensorBoardParser,
    discover_runs,
)

TAGS = ("train/loss", "val/accuracy")

CASES = [
    ("tensorboard/tb_run", TensorBoardParser, "tensorboard"),
    ("csv_wide/csv_wide_run", CsvParser, "csv"),
    ("csv_long/csv_long_run", CsvParser, "csv"),
    ("jsonl/jsonl_run", JsonlParser, "jsonl"),
]


@pytest.mark.parametrize(("relative", "parser_class", "format_name"), CASES)
def test_parser_reads_its_fixture(
    fixture_root: Path,
    reference_series: dict[str, np.ndarray],
    relative: str,
    parser_class: type,
    format_name: str,
) -> None:
    parser = parser_class()
    path = fixture_root / relative
    assert parser.can_parse(path)

    experiment = parser.parse(path)
    assert experiment.run_id == path.name
    assert experiment.source_format == format_name
    assert experiment.tags == sorted(TAGS)

    for tag in TAGS:
        series = experiment.series(tag)
        assert len(series) == reference_series[tag].size
        np.testing.assert_array_equal(series.steps, np.arange(len(series), dtype=np.int64))
        np.testing.assert_array_equal(series.values, reference_series[tag])


@pytest.mark.parametrize(("relative", "parser_class", "format_name"), CASES)
def test_config_is_read_from_the_adjacent_file(
    fixture_root: Path, relative: str, parser_class: type, format_name: str
) -> None:
    experiment = parser_class().parse(fixture_root / relative)
    assert experiment.config["learning_rate"] == 0.001
    assert experiment.config["batch_size"] == 32
    assert experiment.config["optimizer"] == "adamw"
    assert experiment.seed == 0


def test_all_formats_agree_exactly(fixture_root: Path) -> None:
    """The point of one model: four containers, one set of numbers."""
    experiments = [
        parser_class().parse(fixture_root / relative) for relative, parser_class, _ in CASES
    ]
    first = experiments[0]
    for other in experiments[1:]:
        assert other.tags == first.tags
        for tag in first.tags:
            np.testing.assert_array_equal(other.series(tag).steps, first.series(tag).steps)
            np.testing.assert_array_equal(other.series(tag).values, first.series(tag).values)


def test_truncated_jsonl_keeps_every_complete_record(fixture_root: Path) -> None:
    """A killed job leaves half a line; that must not cost the other 30 records."""
    experiment = JsonlParser().parse(fixture_root / "jsonl_truncated" / "killed_run")
    assert len(experiment.series("train/loss")) == 30
    assert experiment.metadata["malformed_lines"] == 1


def test_discovery_finds_every_run_once(fixture_root: Path) -> None:
    found = discover_runs(fixture_root, DEFAULT_PARSERS)
    names = sorted(parser.run_id(path) for parser, path in found)
    assert names == [
        "csv_long_run",
        "csv_wide_run",
        "jsonl_run",
        "killed_run",
        "tb_run",
    ]
    assert len(names) == len(set(names))


def test_discovery_claims_the_right_parser(fixture_root: Path) -> None:
    by_run = {
        parser.run_id(path): parser.format_name
        for parser, path in discover_runs(fixture_root, DEFAULT_PARSERS)
    }
    assert by_run["tb_run"] == "tensorboard"
    assert by_run["csv_wide_run"] == "csv"
    assert by_run["csv_long_run"] == "csv"
    assert by_run["jsonl_run"] == "jsonl"


def test_fingerprint_changes_only_when_the_source_does(tmp_path: Path) -> None:
    log = tmp_path / "run" / "metrics.csv"
    log.parent.mkdir()
    log.write_text("step,loss\n0,1.0\n1,0.5\n", encoding="utf-8")
    parser = CsvParser()

    before = parser.fingerprint(log.parent)
    assert parser.fingerprint(log.parent) == before, "fingerprint must be stable"

    log.write_text("step,loss\n0,1.0\n1,0.5\n2,0.25\n", encoding="utf-8")
    assert parser.fingerprint(log.parent) != before


def test_missing_step_column_is_a_clear_error(tmp_path: Path) -> None:
    log = tmp_path / "run" / "metrics.csv"
    log.parent.mkdir()
    log.write_text("epoch_number,loss\n0,1.0\n", encoding="utf-8")
    with pytest.raises(ParseError, match="no step column"):
        CsvParser().parse(log.parent)


def test_non_numeric_columns_are_dropped_not_coerced(tmp_path: Path) -> None:
    log = tmp_path / "run" / "metrics.csv"
    log.parent.mkdir()
    log.write_text("step,loss,checkpoint\n0,1.0,ckpt-0.pt\n1,0.5,ckpt-1.pt\n", encoding="utf-8")
    experiment = CsvParser().parse(log.parent)
    assert experiment.tags == ["loss"]


def test_repeated_steps_keep_the_later_write(tmp_path: Path) -> None:
    """A resumed job rewrites steps; the surviving value is the later one."""
    log = tmp_path / "run" / "metrics.csv"
    log.parent.mkdir()
    log.write_text("step,loss\n0,1.0\n1,0.5\n1,0.4\n2,0.3\n", encoding="utf-8")
    series = CsvParser().parse(log.parent).series("loss")
    np.testing.assert_array_equal(series.steps, [0, 1, 2])
    np.testing.assert_allclose(series.values, [1.0, 0.4, 0.3], rtol=1e-6)
