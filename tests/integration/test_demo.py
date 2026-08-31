"""`triage demo`: the tool demonstrating itself from a pip install (D32).

Before this verb the demo was a `make` target over a script in `examples/`,
which is to say it was available only to somebody who had already cloned the
repository, which is to say only to somebody who had already decided to trust
it. The sweep therefore moved into the package, and this is the test that the
one command does the whole thing: synthesise, ingest, compare, report.

The sweep is shortened here by pointing `triage.demo.N_STEPS` at a smaller
number. That is not a stub: every line of the real path runs, over the real
generator, the real parsers, the real store and the real renderer. Only the
curves are shorter, and the full length sweep is covered by the session fixture
in `test_cli_end_to_end.py` besides.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import triage.demo
from triage.cli import EXIT_OK, main

SHORT_STEPS = 400


@pytest.fixture
def short_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same sweep, with shorter curves. `loss_spec` reads this at call time."""
    monkeypatch.setattr(triage.demo, "N_STEPS", SHORT_STEPS)


def test_demo_writes_a_report_and_leaves_nothing_else_behind(
    short_sweep: None, tmp_path: Path
) -> None:
    output = tmp_path / "demo.html"
    assert main(["demo", "--output", str(output), "--quiet"]) == EXIT_OK
    assert output.exists()
    assert [path.name for path in tmp_path.iterdir()] == ["demo.html"], (
        "the sweep and its database are scaffolding and belong in the temporary directory"
    )


def test_the_demo_report_names_the_condition_that_really_is_best(
    short_sweep: None, tmp_path: Path
) -> None:
    """The whole argument of the demo: it finds the winner it was not told about."""
    output = tmp_path / "demo.html"
    assert main(["demo", "--output", str(output), "--quiet"]) == EXIT_OK
    html = output.read_text(encoding="utf-8")
    assert triage.demo.KNOWN_BEST in html
    assert triage.demo.BASELINE in html
    # At this shortened length the single seed condition's final window cannot
    # hold the eight independent blocks the weak mode requires, so the tool
    # refuses that one comparison and the report says so. That is the refusal
    # behaving correctly on a smaller sweep, and it is worth asserting here:
    # the full length run, where the weak mode does apply and is labelled
    # WEAKER CLAIM, is covered in test_cli_end_to_end.py.
    assert "Comparisons not made" in html
    assert "lr0.0030_bs128" in html


def test_keep_leaves_the_sweep_and_the_database_where_it_was_told(
    short_sweep: None, tmp_path: Path
) -> None:
    workspace = tmp_path / "kept"
    output = tmp_path / "demo.html"
    assert main(["demo", "--output", str(output), "--keep", str(workspace), "--quiet"]) == EXIT_OK

    from triage.core import Store

    assert (workspace / "demo_sweep").is_dir()
    with Store(workspace / "triage.db") as store:
        expected = sum(condition.n_seeds for condition in triage.demo.CONDITIONS)
        assert len(store) == expected
        assert {run.source_format for run in store.load_all()} == {"tensorboard", "csv", "jsonl"}


def test_the_demo_names_the_ground_truth_after_the_verdicts(
    short_sweep: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Loud, the demo says what was really true, and says it last."""
    assert main(["demo", "--output", str(tmp_path / "demo.html")]) == EXIT_OK
    printed = capsys.readouterr().out
    assert "synthesising 31 runs" in printed
    assert printed.index("report:") < printed.index(triage.demo.KNOWN_BEST)
    assert "produced without either of those facts" in printed


def test_the_packaged_sweep_and_the_example_script_are_one_sweep() -> None:
    """`examples/make_synthetic_runs.py` is a CLI over the package, not a copy.

    Two copies of a generator seeded the same way would agree until the day one
    of them was edited, and the committed reports would then be reproducible
    from a script that no longer described them.
    """
    from examples import make_synthetic_runs

    assert make_synthetic_runs.generate is triage.demo.generate
    assert make_synthetic_runs.CONDITIONS is triage.demo.CONDITIONS
    assert make_synthetic_runs.BASELINE == triage.demo.BASELINE
    assert make_synthetic_runs.KNOWN_BEST == triage.demo.KNOWN_BEST
    assert make_synthetic_runs.SWEEP_SEED == triage.demo.SWEEP_SEED
