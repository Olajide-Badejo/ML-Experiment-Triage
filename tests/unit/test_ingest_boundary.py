"""The ingest boundary: run identity, discovery, and the error boundary.

Everything here pins a defect verified against v1.0.0, and every one of them
was a silent one. A sweep that ingested "1 added, 1 updated" and stored a
single row; a sweep whose two real runs were skipped in favour of a stray file
at the root; a poisoned line that cost forty runs rather than one. The theme is
that the boundary must lose a run loudly or not at all.
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from triage.cli import main
from triage.core.experiment import Experiment
from triage.core.store import Store
from triage.ingest import ingest
from triage.parsers import DEFAULT_PARSERS, Parser, discover_runs
from triage.progress import track


def write_run(directory: Path, rows: int = 30, value: float = 1.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["step,loss"] + [f"{i},{value / (i + 1):.6f}" for i in range(rows)]
    directory.joinpath("metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def write_poisoned_run(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath("metrics.csv").write_text(
        "step,loss\n0,1.0\n1,NaN\n2,inf\n3,0.25\n", encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------------- D1


def test_the_ingest_summary_reports_dropped_non_finite_points(
    tmp_path: Path, temp_database: Path
) -> None:
    """Dropping a point silently is the defect; the count belongs in the summary."""
    root = tmp_path / "sweep"
    write_poisoned_run(root / "poisoned")
    write_run(root / "clean")

    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)

    assert result.dropped_non_finite == 2
    assert "2 non finite points dropped" in result.summary()


def test_a_clean_sweep_says_nothing_about_dropped_points(
    tmp_path: Path, temp_database: Path
) -> None:
    root = tmp_path / "sweep"
    write_run(root / "clean")
    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)
    assert result.dropped_non_finite == 0
    assert "non finite" not in result.summary()


# --------------------------------------------------------------------- D5


def discovered(root: Path) -> list[Path]:
    """The run directories `discover_runs` claims under `root`, sorted."""
    return sorted(path for _parser, path in discover_runs(root, DEFAULT_PARSERS))


def test_a_stray_file_at_the_sweep_root_does_not_swallow_the_sweep(tmp_path: Path) -> None:
    """The verified defect: a sweep of two runs discovered as ONE run, exit 0.

    `walk()` tested the parsers against a directory before descending, so an
    `index.csv` written at the sweep root claimed the root itself and both real
    runs were never looked at.
    """
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    write_run(root / "run_b")
    root.joinpath("index.csv").write_text("step,note\n0,1.0\n", encoding="utf-8")

    assert discovered(root) == [root / "run_a", root / "run_b"]


def test_the_stray_file_case_ingests_both_runs(tmp_path: Path, temp_database: Path) -> None:
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    write_run(root / "run_b")
    root.joinpath("index.csv").write_text("step,note\n0,1.0\n", encoding="utf-8")

    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)
        assert len(store) == 2
    assert len(result.added) == 2


def test_a_run_directory_with_unparseable_subdirectories_is_still_one_run(
    tmp_path: Path,
) -> None:
    """Leaf claiming must not fragment a run that holds checkpoints or plots."""
    run = write_run(tmp_path / "sweep" / "run_a")
    run.joinpath("checkpoints").mkdir()
    run.joinpath("checkpoints", "epoch0.pt").write_bytes(b"not a log")

    assert discovered(tmp_path / "sweep") == [run]


def test_a_nested_tensorboard_layout_yields_the_leaves_not_the_parent(
    tmp_path: Path,
) -> None:
    """`runX/train` and `runX/val` are two runs; `runX` itself is neither."""
    root = tmp_path / "sweep"
    write_run(root / "runX" / "train")
    write_run(root / "runX" / "val")
    root.joinpath("runX", "metrics.csv").write_text("step,loss\n0,1.0\n", encoding="utf-8")

    assert discovered(root) == [root / "runX" / "train", root / "runX" / "val"]


def test_claim_and_descend_decisions_are_logged_at_debug_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    root.joinpath("index.csv").write_text("step,note\n0,1.0\n", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger="triage.parsers"):
        discover_runs(root, DEFAULT_PARSERS)

    messages = [record.getMessage() for record in caplog.records]
    assert any("claim" in message and "run_a" in message for message in messages)
    assert any("descend" in message for message in messages)


# --------------------------------------------------------------------- D4


def test_two_sweeps_with_the_same_seed_name_are_two_rows(
    tmp_path: Path, temp_database: Path
) -> None:
    """The verified defect: ONE row holding the second run, "1 added, 1 updated"."""
    root = tmp_path / "sweep"
    write_run(root / "sweep_a" / "seed0", value=1.0)
    write_run(root / "sweep_b" / "seed0", value=2.0)

    with Store(temp_database) as store:
        result = ingest(root, store, show_progress=False)
        assert store.run_ids() == ["sweep_a/seed0", "sweep_b/seed0"]
        assert len(store) == 2
        # And each row holds its own numbers, not the last one parsed twice.
        first = store.load("sweep_a/seed0").series("loss").values[0]
        second = store.load("sweep_b/seed0").series("loss").values[0]
        assert first != second
    assert len(result.added) == 2
    assert result.updated == []


def test_a_run_directly_under_the_root_keeps_its_plain_name(
    tmp_path: Path, temp_database: Path
) -> None:
    """The common flat layout must not gain a path prefix it does not need."""
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    with Store(temp_database) as store:
        ingest(root, store, show_progress=False)
        assert store.run_ids() == ["run_a"]


def test_run_ids_use_forward_slashes_on_every_platform(tmp_path: Path, temp_database: Path) -> None:
    root = tmp_path / "sweep"
    write_run(root / "group" / "run_a")
    with Store(temp_database) as store:
        ingest(root, store, show_progress=False)
        assert store.run_ids() == ["group/run_a"]
        assert "\\" not in store.run_ids()[0]


def test_a_run_id_arriving_from_a_different_source_path_fails_that_run(
    tmp_path: Path, temp_database: Path
) -> None:
    """Identity collision must cost one run loudly, never overwrite in silence."""
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    with Store(temp_database) as store:
        ingest(root, store, show_progress=False)

        # The same run id now arrives from somewhere else entirely, which is
        # what a rename or two roots ingested into one database produces.
        moved = write_run(tmp_path / "other" / "run_a", value=5.0)
        result = ingest(moved.parent, store, show_progress=False)

        assert result.added == []
        assert [name for name, _ in result.failed] == ["run_a"]
        assert len(store) == 1
        # The stored row is untouched: still the original path and numbers.
        assert store.load("run_a").source_path.replace("\\", "/").endswith("sweep/run_a")
    message = result.failed[0][1]
    assert "already stored" in message or "different source" in message


def test_reingesting_the_same_sweep_from_the_same_root_is_not_a_collision(
    tmp_path: Path, temp_database: Path
) -> None:
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    with Store(temp_database) as store:
        ingest(root, store, show_progress=False)
        second = ingest(root, store, show_progress=False)
    assert second.failed == []
    assert second.skipped == ["run_a"]


# -------------------------------------------------------------------- D20


class ExplodingParser(Parser):
    """A parser whose `parse` raises whatever it was constructed with.

    The point is to raise something OUTSIDE the old catch tuple of
    (ParseError, OSError, ValueError). OverflowError, RecursionError,
    sqlite3.Error and zlib.error were all reachable in practice and all fatal.
    """

    format_name = "exploding"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def can_parse(self, path: Path) -> bool:
        return path.is_dir() and path.name.startswith("bad")

    def parse(self, path: Path, root: Path | None = None) -> Experiment:
        raise self.error

    def fingerprint(self, path: Path, root: Path | None = None) -> str:
        return "exploding"


@pytest.mark.parametrize(
    "error",
    [
        OverflowError("step out of range"),
        RecursionError("too deep"),
        sqlite3.Error("database is locked"),
        zlib.error("invalid block"),
        RuntimeError("something nobody predicted"),
    ],
)
def test_one_unexpected_error_costs_one_run_not_the_sweep(
    tmp_path: Path, temp_database: Path, error: Exception
) -> None:
    """Verified: an OverflowError from one poisoned line aborted the whole walk."""
    root = tmp_path / "sweep"
    write_run(root / "good_a")
    (root / "bad_run").mkdir(parents=True)
    write_run(root / "good_b")
    parsers = [ExplodingParser(error), *DEFAULT_PARSERS]

    with Store(temp_database) as store:
        result = ingest(root, store, parsers=parsers, show_progress=False)
        assert sorted(store.run_ids()) == ["good_a", "good_b"]

    assert sorted(result.added) == ["good_a", "good_b"]
    assert [name for name, _ in result.failed] == ["bad_run"]


def test_a_failure_records_its_type_and_traceback(tmp_path: Path, temp_database: Path) -> None:
    """str(error) alone loses which exception it was and where it came from."""
    root = tmp_path / "sweep"
    (root / "bad_run").mkdir(parents=True)
    parsers = [ExplodingParser(OverflowError("step out of range")), *DEFAULT_PARSERS]

    with Store(temp_database) as store:
        result = ingest(root, store, parsers=parsers, show_progress=False)

    message = result.failed[0][1]
    assert "OverflowError" in message
    assert "step out of range" in message
    assert "bad_run" in result.tracebacks
    assert "Traceback" in result.tracebacks["bad_run"]


def test_keyboard_interrupt_is_not_swallowed(tmp_path: Path, temp_database: Path) -> None:
    """A per run catch of Exception must still let the user stop the ingest."""
    root = tmp_path / "sweep"
    (root / "bad_run").mkdir(parents=True)
    parsers = [ExplodingParser(KeyboardInterrupt()), *DEFAULT_PARSERS]

    with Store(temp_database) as store, pytest.raises(KeyboardInterrupt):
        ingest(root, store, parsers=parsers, show_progress=False)


def test_system_exit_is_not_swallowed(tmp_path: Path, temp_database: Path) -> None:
    root = tmp_path / "sweep"
    (root / "bad_run").mkdir(parents=True)
    parsers = [ExplodingParser(SystemExit(3)), *DEFAULT_PARSERS]

    with Store(temp_database) as store, pytest.raises(SystemExit):
        ingest(root, store, parsers=parsers, show_progress=False)


def test_an_empty_run_warns_again_on_reingest(tmp_path: Path, temp_database: Path) -> None:
    """Verified: an empty run stored a fingerprint, so its warning never returned.

    The second pass skipped it as unchanged and said nothing, so a run that
    produced no scalars looked identical to a healthy one from the second
    ingest onwards.
    """
    root = tmp_path / "sweep"
    empty = root / "empty_run"
    empty.mkdir(parents=True)
    # Parses cleanly, carries a step column, and yields no scalar series.
    empty.joinpath("metrics.csv").write_text("step,note\n0,hello\n1,world\n", encoding="utf-8")

    with Store(temp_database) as store:
        first = ingest(root, store, show_progress=False)
        assert first.empty == ["empty_run"]

        second = ingest(root, store, show_progress=False)
        assert second.empty == ["empty_run"], "the warning must survive the skip"


def _link_to_self(link: Path, target: Path) -> None:
    """Make `link` a directory link pointing at `target`, or skip the test.

    A plain symlink needs Developer Mode or an elevated shell on Windows, so a
    directory junction is used there instead: it creates the same cycle for a
    walk that resolves paths, which is the thing under test, and it needs no
    privilege. If neither is available the test is skipped rather than passing
    on a filesystem where the defect cannot exist.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform != "win32":
        pytest.skip("this platform does not allow creating symlinks unprivileged")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not link.exists():
        pytest.skip(f"could not create a directory junction: {result.stderr.strip()}")


def test_a_symlink_cycle_does_not_end_the_walk_in_a_recursion_error(
    tmp_path: Path,
) -> None:
    """Verified fatal: walk() recursed forever through a directory loop."""
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    _link_to_self(root / "loop", root)

    found = sorted(path for _parser, path in discover_runs(root, DEFAULT_PARSERS))
    assert root / "run_a" in found


def test_an_unreadable_directory_does_not_stop_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PermissionError on one subdirectory used to abort before the loop began."""
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    forbidden = root / "forbidden"
    forbidden.mkdir()

    real_iterdir = Path.iterdir

    def guarded(self: Path):
        if self.name == "forbidden":
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", guarded)
    found = sorted(path for _parser, path in discover_runs(root, DEFAULT_PARSERS))
    assert found == [root / "run_a"]


# ------------------------------------------------- D20: progress and logging


def test_progress_lines_go_to_stderr_never_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Verified: progress on stdout interleaved with piped report output."""
    consumed = list(track(range(3), "ingest", enabled=True))
    captured = capsys.readouterr()
    assert consumed == [0, 1, 2]
    assert captured.out == ""
    assert "ingest" in captured.err


def test_the_progress_bar_is_closed_even_when_the_consumer_raises() -> None:
    """A generator abandoned mid iteration must still tear its bar down."""
    closed: list[bool] = []

    class FakeBar:
        def __init__(self, items, **_kwargs) -> None:
            self.items = items

        def __iter__(self):
            return iter(self.items)

        def set_postfix_str(self, *_args, **_kwargs) -> None:
            return None

        def close(self) -> None:
            closed.append(True)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("triage.progress.tqdm_class", lambda: FakeBar)
        patch.setattr("triage.progress.is_terminal", lambda: True)
        stream = track(range(10), "ingest", enabled=True)
        next(stream)
        stream.close()

    assert closed == [True], "the bar must be closed by a finally, not by falling off the end"


def test_the_progress_bar_degrades_to_plain_lines_when_tqdm_is_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E1. tqdm ships in the `parsers` extra, and a core install has no bar.

    `triage.progress` sits on the import path of `triage.ingest`, so a hard
    dependency here would mean a core install could not ingest anything. A
    missing progress bar is a missing convenience: it must not stop the work or
    raise, and the plain line branch is already there for CI logs.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("triage.progress.tqdm_class", lambda: None)
        patch.setattr("triage.progress.is_terminal", lambda: True)
        consumed = list(track(range(3), "ingest", enabled=True))

    captured = capsys.readouterr()
    assert consumed == [0, 1, 2]
    assert captured.out == ""
    assert "ingest" in captured.err


def test_the_cli_routes_triage_logs_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    root.joinpath("index.csv").write_text("step,note\n0,1.0\n", encoding="utf-8")
    database = tmp_path / "triage.db"

    exit_code = main(
        ["ingest", str(root), "--database", str(database), "--quiet", "--log-level", "debug"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "descend" in captured.err, "debug discovery decisions belong on stderr"
    assert "descend" not in captured.out


def test_the_default_log_level_is_quiet_about_discovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "sweep"
    write_run(root / "run_a")
    database = tmp_path / "triage.db"

    assert main(["ingest", str(root), "--database", str(database), "--quiet"]) == 0
    assert "descend" not in capsys.readouterr().err
