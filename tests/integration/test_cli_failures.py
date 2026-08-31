"""Every way the command line was verified to fail badly, and how it fails now.

A command line tool is judged on its bad paths as much as its good one, and this
one was failing all of them the same way: a raw traceback and exit 1. That is
the worst possible answer, because exit 1 already meant "some runs failed", so a
CI job could not tell a crash from a result, and a traceback tells a user
nothing they can act on.

The rules these tests hold the CLI to:

* every numeric flag is validated at parse time, against the bounds its meaning
  actually has, so a nonsense value is rejected by name rather than surfacing as
  a `ZeroDivisionError` eight frames down or, worse, not surfacing at all;
* every failure this tool can foresee prints `error: <message>` on stderr and
  exits 2, and the message names the file, the flag or the run it is about;
* the exit codes mean one thing each, and the table is in `--help`;
* a `compare` that performed zero comparisons does not exit 0, because a CI gate
  that passes vacuously is worse than one that fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import triage.cli
from triage.cli import EXIT_CODES, EXIT_NO_COMPARISONS, EXIT_RUNS_FAILED, main
from triage.core.store import Store
from triage.synthetic import CurveSpec, generate_condition

BASELINE = "base"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    """Two conditions with seed replicates: a database compare can work on."""
    rng = np.random.default_rng(4242)
    runs = generate_condition(BASELINE, CurveSpec(n_steps=600), 3, rng) + generate_condition(
        "candidate", CurveSpec(n_steps=600, effect=0.05), 3, rng
    )
    path = tmp_path / "triage.db"
    with Store(path) as store:
        for run in runs:
            store.upsert(run, source_hash=run.run_id)
    return path


def compare_argv(database: Path, *extra: str) -> list[str]:
    return ["compare", "--database", str(database), "--baseline", BASELINE, "--quiet", *extra]


# ----------------------------------------------------------- the exit contract


def test_the_exit_code_table_is_in_the_help(capsys: pytest.CaptureFixture[str]) -> None:
    """A caller writing a CI gate has to be able to read the codes off `--help`."""
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    for code, meaning in EXIT_CODES.items():
        assert f"{code}  {meaning}" in printed


def test_every_exit_code_means_exactly_one_thing() -> None:
    """Exit 1 used to mean both 'some runs failed' and 'the tool crashed'."""
    assert set(EXIT_CODES) == {0, 1, 2, 3, 4}
    assert len(set(EXIT_CODES.values())) == len(EXIT_CODES)
    assert EXIT_RUNS_FAILED == 3
    assert EXIT_NO_COMPARISONS == 4


def test_version_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from triage import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


# ------------------------------------------------------ the numeric validators


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--alpha", "2.0", "at or below 1"),
        ("--alpha", "0", "above 0"),
        ("--fdr", "-1", "above 0"),
        ("--fdr", "1.5", "at or below 1"),
        ("--permutations", "-5", "at or above 1"),
        ("--permutations", "0", "at or above 1"),
        ("--window-fraction", "5", "at or below 1"),
        ("--window-fraction", "0", "above 0"),
        ("--window-minimum", "1", "at or above 2"),
        ("--seed", "-1", "at or above 0"),
        ("--practical-threshold", "-3", "at or above 0"),
        ("--practical-threshold-absolute", "-3", "at or above 0"),
    ],
)
def test_an_out_of_range_number_is_rejected_at_parse_time(
    database: Path, flag: str, value: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--alpha 2.0 --fdr -1 --permutations -5` were all accepted in silence."""
    with pytest.raises(SystemExit) as exit_info:
        main(compare_argv(database, flag, value))
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert flag in message
    assert expected in message


def test_a_number_that_is_not_a_number_is_rejected_by_name(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(compare_argv(database, "--alpha", "loose"))
    assert exit_info.value.code == 2
    assert "loose" in capsys.readouterr().err


def test_a_valid_number_at_the_boundary_is_accepted(database: Path) -> None:
    """The bounds are the domain of the parameter, not a guess at good taste."""
    assert main(compare_argv(database, "--window-fraction", "1.0", "--fdr", "1.0")) == 0


# --------------------------------------------------------- the handled failures


def test_a_file_that_is_not_a_database_is_a_handled_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verified: a raw sqlite3.DatabaseError traceback at exit 1."""
    path = tmp_path / "notes.txt"
    path.write_text("plain text, not a database\n", encoding="utf-8")
    assert main(["compare", "--database", str(path), "--baseline", BASELINE, "--quiet"]) == 2
    assert "not a triage database" in capsys.readouterr().err


def test_a_directory_as_the_database_is_a_handled_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "a_directory"
    directory.mkdir()
    assert main(["compare", "--database", str(directory), "--baseline", BASELINE, "--quiet"]) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_a_directory_as_the_report_output_is_a_handled_error(
    database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "output_directory"
    directory.mkdir()
    code = main(
        [
            "report",
            "--database",
            str(database),
            "--baseline",
            BASELINE,
            "--output",
            str(directory),
            "--quiet",
        ]
    )
    assert code == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_ingesting_a_path_that_does_not_exist_is_a_handled_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["ingest", str(tmp_path / "nowhere"), "--database", str(tmp_path / "t.db"), "--quiet"]
    )
    assert code == 2
    assert "no such path" in capsys.readouterr().err


def test_a_key_error_message_is_not_printed_with_its_repr_quotes(
    monkeypatch: pytest.MonkeyPatch, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`print(f"{error}")` on a KeyError yields the message wrapped in quotes.

    KeyError is the one exception whose `str` is `repr(args[0])`, so a carefully
    written message came out as `error: "no run 'x' in triage.db"`, quotes and
    all, which reads like the tool quoting somebody else rather than speaking.
    """

    def explode(_args: object) -> int:
        raise KeyError("no run 'ghost' in the database")

    monkeypatch.setitem(triage.cli.HANDLERS, "compare", explode)
    assert main(compare_argv(database)) == 2
    assert capsys.readouterr().err.strip() == "error: no run 'ghost' in the database"


# ------------------------------------------------- the codes that are not two


def test_a_sweep_with_a_failing_run_exits_three(tmp_path: Path) -> None:
    """Exit 1 meant a crash as well, so a CI job could not tell them apart."""
    root = tmp_path / "sweep"
    good = root / "good_run"
    good.mkdir(parents=True)
    good.joinpath("metrics.csv").write_text(
        "step,loss\n" + "".join(f"{i},{1.0 / (i + 1):.5f}\n" for i in range(30)), encoding="utf-8"
    )
    bad = root / "bad_run"
    bad.mkdir()
    bad.joinpath("metrics.csv").write_text("epoch_number,loss\n0,1.0\n", encoding="utf-8")

    code = main(["ingest", str(root), "--database", str(tmp_path / "t.db"), "--quiet"])
    assert code == EXIT_RUNS_FAILED


def test_a_compare_that_made_no_comparisons_does_not_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A vacuous gate that passes is worse than a gate that fails."""
    rng = np.random.default_rng(11)
    path = tmp_path / "one.db"
    with Store(path) as store:
        for run in generate_condition(BASELINE, CurveSpec(n_steps=400), 3, rng):
            store.upsert(run, source_hash=run.run_id)

    code = main(["compare", "--database", str(path), "--baseline", BASELINE, "--quiet"])
    assert code == EXIT_NO_COMPARISONS
    assert "no comparisons" in capsys.readouterr().out.lower()


def test_a_report_that_made_no_comparisons_does_not_exit_zero(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    path = tmp_path / "one.db"
    with Store(path) as store:
        for run in generate_condition(BASELINE, CurveSpec(n_steps=400), 3, rng):
            store.upsert(run, source_hash=run.run_id)

    code = main(
        [
            "report",
            "--database",
            str(path),
            "--baseline",
            BASELINE,
            "--output",
            str(tmp_path / "r.html"),
            "--quiet",
        ]
    )
    assert code == EXIT_NO_COMPARISONS
    assert (tmp_path / "r.html").exists(), "the report is still written; only the code differs"


# ------------------------------------------------------------------ the flags


def test_quiet_drops_the_count_summary_and_nothing_else(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--quiet` was a documented no-op on compare and report.

    Honoured, it removes the one line that states nothing the table does not:
    the count summary. It cannot remove more than that without printing numbers
    stripped of their provenance, which is ground rule 3 and not negotiable for
    the sake of a shorter CI log.
    """
    assert main(compare_argv(database)) == 0
    quiet = capsys.readouterr().out

    assert main(["compare", "--database", str(database), "--baseline", BASELINE]) == 0
    loud = capsys.readouterr().out

    assert "comparisons against" in loud
    assert "comparisons against" not in quiet
    for provenance in ("gates:", "families:", "statistic:", "permutation seed"):
        assert provenance in quiet, f"{provenance} qualifies the table and must survive --quiet"
    assert "verdict" in quiet
    assert "candidate" in quiet


def test_quiet_is_honoured_by_report(
    database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "report",
        "--database",
        str(database),
        "--baseline",
        BASELINE,
        "--output",
        str(tmp_path / "r.html"),
    ]
    assert main([*argv, "--quiet"]) == 0
    quiet = capsys.readouterr().out
    assert main(argv) == 0
    loud = capsys.readouterr().out

    assert "comparisons against" in loud
    assert "comparisons against" not in quiet
    assert str(tmp_path / "r.html") in quiet, "the path is the result and always prints"


def test_compare_does_not_run_the_sensitivity_pass(
    database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_compare` computed the whole sensitivity analysis and discarded it."""
    calls: list[int] = []

    def counted(*args: object, **kwargs: object) -> list[object]:
        calls.append(1)
        return []

    monkeypatch.setattr("triage.cli.analyse", counted)
    assert main(compare_argv(database)) == 0
    assert calls == [], "compare must not pay for a table it never prints"


def test_report_still_runs_the_sensitivity_pass(
    database: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def counted(*args: object, **kwargs: object) -> list[object]:
        calls.append(1)
        return []

    monkeypatch.setattr("triage.cli.analyse", counted)
    code = main(
        [
            "report",
            "--database",
            str(database),
            "--baseline",
            BASELINE,
            "--output",
            str(tmp_path / "r.html"),
            "--quiet",
        ]
    )
    assert code == 0
    assert calls == [1]


# ----------------------------------------------------- the database is intact


def test_a_handled_failure_leaves_no_database_behind(tmp_path: Path) -> None:
    """The read verbs must not create the file they were mistakenly pointed at."""
    missing = tmp_path / "typo.db"
    assert main(["compare", "--database", str(missing), "--baseline", BASELINE, "--quiet"]) == 2
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []
