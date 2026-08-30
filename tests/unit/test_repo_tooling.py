"""The repository's own scripts, tested like the rest of the code.

`scripts/` holds the gates that run in CI and the generators that produce the
committed fixtures. They were the one part of the tree with no tests at all,
which is how `scripts/clean.py --venv` came to print success for a deletion
that had not happened. Each test here pins a promise one of those scripts makes
to the person reading its output.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from scripts import check_no_dashes, clean, make_fixtures

# Written as escapes so this file never contains the characters under test, and
# so the guard passing over this file is not itself the thing being asserted.
EN_DASH = "\u2013"
EM_DASH = "\u2014"
FULLWIDTH_HYPHEN = "\uff0d"


def findings_for(root: Path) -> list[str]:
    findings, _text, _pdf = check_no_dashes.check(root, skip_pdf=True)
    return [finding.detail for finding in findings]


def test_the_dash_guard_reads_prose_and_finds_an_em_dash(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(f"a{EM_DASH}b\n", encoding="utf-8")
    assert len(findings_for(tmp_path)) == 1


@pytest.mark.parametrize("suffix", [".json", ".csv", ".jsonl"])
def test_the_dash_guard_leaves_data_files_alone(tmp_path: Path, suffix: str) -> None:
    """A dash in data is content, not prose.

    The guard is ground rule 1 applied to what this project writes. A fixture
    that encodes a user supplied string, or a CSV cell copied out of somebody
    else's log, is data passing through: rewriting it to satisfy a prose rule
    would corrupt the record the fixture exists to be.
    """
    (tmp_path / f"payload{suffix}").write_text(f'{{"label": "a{EN_DASH}b"}}\n', encoding="utf-8")
    assert findings_for(tmp_path) == []


def test_the_dash_guard_does_not_object_to_a_fullwidth_hyphen(tmp_path: Path) -> None:
    """U+FF0D is the CJK width of an ASCII hyphen, not a dash.

    It was in the forbidden table beside the em dash, so a Japanese or Chinese
    string in any scanned file was a violation of a rule about typographic
    dashes in English prose. Nothing in this repository writes one, which is
    exactly why it could sit there being wrong.
    """
    (tmp_path / "notes.md").write_text(f"a{FULLWIDTH_HYPHEN}b\n", encoding="utf-8")
    assert findings_for(tmp_path) == []


def test_the_dash_guard_still_catches_a_latex_ligature(tmp_path: Path) -> None:
    """`--` in LaTeX source is an en dash in the PDF, so it is still a finding."""
    (tmp_path / "paper.tex").write_text("a range of 2--8 percent\n", encoding="utf-8")
    details = findings_for(tmp_path)
    assert len(details) == 1
    assert "en dash" in details[0]


def test_the_dash_guard_reports_what_it_read(tmp_path: Path) -> None:
    """The count in the OK line is the scope claim, so it has to be true."""
    (tmp_path / "notes.md").write_text("clean prose\n", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}\n", encoding="utf-8")
    findings, text_checked, pdf_checked = check_no_dashes.check(tmp_path, skip_pdf=True)
    assert findings == []
    assert (text_checked, pdf_checked) == (1, 0)


def test_the_scripts_directory_is_importable_as_a_package() -> None:
    """Guards the import above: `scripts` is a namespace package on the path."""
    assert Path(check_no_dashes.__file__).parent.name == "scripts"
    assert "scripts.check_no_dashes" in sys.modules


def test_the_committed_fixtures_match_what_the_generator_produces() -> None:
    """D35f. The fixtures are committed, so nothing proved they were current.

    A parser fixture is a claim about bytes. Committing the bytes and the
    generator separately means the generator can drift from what is in the tree
    and nobody finds out until someone regenerates and gets a diff they cannot
    explain. This is the check that CI runs.
    """
    assert make_fixtures.differences(make_fixtures.FIXTURES) == []


def test_the_fixture_check_notices_a_changed_byte(tmp_path: Path) -> None:
    copied = tmp_path / "fixtures"
    shutil.copytree(make_fixtures.FIXTURES, copied)
    target = copied / "jsonl" / "jsonl_run" / "metrics.jsonl"
    target.write_text(target.read_text(encoding="utf-8") + '{"step": 999}\n', encoding="utf-8")

    differences = make_fixtures.differences(copied)

    assert len(differences) == 1
    assert "jsonl/jsonl_run/metrics.jsonl" in differences[0]


def test_the_fixture_check_notices_a_missing_file(tmp_path: Path) -> None:
    copied = tmp_path / "fixtures"
    shutil.copytree(make_fixtures.FIXTURES, copied)
    (copied / "csv_wide" / "csv_wide_run" / "metrics.csv").unlink()

    differences = make_fixtures.differences(copied)

    assert len(differences) == 1
    assert "missing" in differences[0]


def test_the_fixture_check_reads_the_event_file_rather_than_hashing_it(tmp_path: Path) -> None:
    """The TensorBoard fixture cannot be compared byte for byte, and why.

    `EventFileWriter` opens every file with a `file_version` event stamped with
    the wall clock at the moment of writing, so two runs of the generator
    produce different bytes from identical inputs. Hashing it would make the
    check fail every time it ran. It is compared through this project's own
    parser instead, which is the stronger statement anyway: the committed file
    still decodes to the series the generator describes.
    """
    copied = tmp_path / "fixtures"
    shutil.copytree(make_fixtures.FIXTURES, copied)
    event_file = copied / "tensorboard" / "tb_run" / "events.out.tfevents.1700000000.fixture"
    original = event_file.read_bytes()

    assert make_fixtures.differences(copied) == []

    truncated = copied / "jsonl" / "jsonl_run" / "metrics.jsonl"
    truncated.write_text("", encoding="utf-8")
    assert make_fixtures.differences(copied) != []
    assert event_file.read_bytes() == original


def test_the_cleaner_refuses_to_delete_the_environment_it_is_running_in() -> None:
    """D34. `make distclean` asked the venv python to delete its own venv.

    On Windows the running `python.exe` is locked, so the tree is left standing
    with pieces missing. `ignore_errors=True` swallowed that and the script
    printed the removal as done, which is the worst of the three possible
    outcomes: the next `make env` reuses a half deleted environment.
    """
    refusal = clean.venv_refusal(Path(sys.prefix))
    assert refusal is not None
    assert "running" in refusal


def test_the_cleaner_will_delete_a_virtual_environment_it_is_not_inside(tmp_path: Path) -> None:
    assert clean.venv_refusal(tmp_path / "some-other-venv") is None


def test_the_cleaner_reports_only_what_it_actually_removed(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    removed: list[str] = []
    failed: list[str] = []

    clean.remove(gone, tmp_path, removed, failed)

    assert (removed, failed) == (["gone"], [])
    assert not gone.exists()


def test_the_cleaner_reports_a_removal_that_did_not_happen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked directory is a failure, not a success with a warning suppressed."""
    locked = tmp_path / "locked"
    locked.mkdir()
    monkeypatch.setattr(clean.shutil, "rmtree", lambda *args, **kwargs: None)
    removed: list[str] = []
    failed: list[str] = []

    clean.remove(locked, tmp_path, removed, failed)

    assert removed == []
    assert failed == ["locked"]
    assert locked.exists()


def test_the_cleaner_exits_non_zero_when_something_survived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "figures").mkdir()
    monkeypatch.setattr(clean, "ROOT", tmp_path)
    monkeypatch.setattr(clean.shutil, "rmtree", lambda *args, **kwargs: None)

    assert clean.main([]) == 1
    assert "could not be removed" in capsys.readouterr().out
