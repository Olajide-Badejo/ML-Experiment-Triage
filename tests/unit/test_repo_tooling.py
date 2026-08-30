"""The repository's own scripts, tested like the rest of the code.

`scripts/` holds the gates that run in CI and the generators that produce the
committed fixtures. They were the one part of the tree with no tests at all,
which is how `scripts/clean.py --venv` came to print success for a deletion
that had not happened. Each test here pins a promise one of those scripts makes
to the person reading its output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import check_no_dashes

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
