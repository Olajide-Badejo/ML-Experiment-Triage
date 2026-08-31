"""The repository's own scripts, tested like the rest of the code.

`scripts/` holds the gates that run in CI and the generators that produce the
committed fixtures. They were the one part of the tree with no tests at all,
which is how `scripts/clean.py --venv` came to print success for a deletion
that had not happened. Each test here pins a promise one of those scripts makes
to the person reading its output.
"""

from __future__ import annotations

import re
import shutil
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from scripts import check_no_dashes, clean, make_fixtures

ROOT = Path(__file__).resolve().parents[2]

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


def test_the_dash_guard_scans_a_tree_whose_absolute_path_holds_an_excluded_name(
    tmp_path: Path,
) -> None:
    """The skip list names directories of this repository, not of the disk.

    `private` is on that list. macOS resolves its temporary directory to
    `/private/var/folders/...`, so matching the list against the absolute path
    skipped every file the guard was pointed at on that platform and reported a
    clean scan of nothing: three tests here passed by scanning zero files, and
    the CI job would have done the same over a checkout under such a path.
    """
    root = tmp_path / "private" / "checkout"
    root.mkdir(parents=True)
    (root / "notes.md").write_text(f"a{EM_DASH}b\n", encoding="utf-8")

    assert len(findings_for(root)) == 1


def test_the_dash_guard_still_skips_an_excluded_directory_inside_the_tree(
    tmp_path: Path,
) -> None:
    """The skip itself still has to work, relative to the root being scanned."""
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "notes.md").write_text(f"a{EM_DASH}b\n", encoding="utf-8")

    assert findings_for(tmp_path) == []


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


# ------------------------------------------------------- the dependency lock


def locked_versions() -> dict[str, str]:
    """`{name: version}` out of the committed PEP 751 lock."""
    with (ROOT / "pylock.toml").open("rb") as handle:
        lock = tomllib.load(handle)
    assert lock["lock-version"] == "1.0", lock["lock-version"]
    return {
        canonicalize_name(package["name"]): str(package["version"]) for package in lock["packages"]
    }


def declared_requirements() -> dict[str, SpecifierSet]:
    """`{name: specifier}` for every package `pyproject.toml` asks for.

    The self referencing extras (`ml-experiment-triage[all]`) are dropped: they
    compose this project's own extras and are not packages a lock could hold.
    """
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    declared = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        declared.extend(extra)
    found: dict[str, SpecifierSet] = {}
    for text in declared:
        requirement = Requirement(text)
        name = canonicalize_name(requirement.name)
        if name == canonicalize_name(project["name"]):
            continue
        found[name] = requirement.specifier
    return found


def test_the_lock_holds_every_dependency_this_project_declares() -> None:
    """The lock is what CI installs, so a gap in it is a gap in what was tested.

    `pylock.toml` replaced `requirements.txt` at 1.1.0 (Section 7 item 4). It is
    compiled from the `dev` extra, which composes every other extra, so anything
    declared anywhere in `pyproject.toml` has to be in it. A dependency added to
    the declarations and not to the lock would install for a user and be absent
    from every job that proves the thing works.
    """
    locked = locked_versions()

    missing = sorted(name for name in declared_requirements() if name not in locked)

    assert missing == [], f"declared but not locked: {missing}. Run `nox -s lock`"
    # The tools the gates are made of, named rather than counted, so that
    # dropping one out of the dev extra fails here and not in CI.
    assert {"hypothesis", "mypy", "nox", "pytest", "ruff", "uv"} <= set(locked)


def test_the_locked_versions_satisfy_the_declared_ranges() -> None:
    """D30. A floor is a claim about what was installed, so it has to hold.

    The declared floors are the versions the calibration was measured against,
    and the lock is what the measurement actually ran under. If those two ever
    disagree, one of them is a fiction, and this says which.
    """
    locked = locked_versions()

    for name, specifier in declared_requirements().items():
        if not str(specifier) or name not in locked:
            continue
        assert Version(locked[name]) in specifier, (
            f"{name} is locked at {locked[name]}, which does not satisfy {specifier}"
        )


def test_the_commit_hook_runs_the_same_ruff_the_lock_installs() -> None:
    """A formatter is only a gate if there is one of it.

    `ruff format` output is version dependent, so a commit hook running a
    different ruff from `nox -s lint` would rewrite files that CI then reports
    as unformatted: a loop the contributor cannot get out of by doing what
    either tool told them. pre-commit pins its hook repository by revision and
    the lock pins the package, and nothing but this holds the two together.
    """
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(r"astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v?([0-9][^\s]*)", config)

    assert match is not None, "the ruff hook repository is not pinned to a revision"
    assert match.group(1) == locked_versions()["ruff"]


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
