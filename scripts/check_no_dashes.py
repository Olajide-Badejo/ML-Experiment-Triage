"""Repo wide guard against em dashes, en dashes, and LaTeX dash ligatures.

Ground rule 1 of this project forbids U+2014 and U+2013 anywhere in the
repository, including inside the compiled PDFs. LaTeX turns ``--`` into an en
dash and ``---`` into an em dash, so plain ASCII source can still smuggle a
dash into a PDF. This script therefore runs three checks:

1. every text file is scanned for the forbidden code points;
2. every ``.tex`` and ``.bib`` file is scanned for runs of two or more hyphens
   outside verbatim style environments, which is how the ligature sneaks in;
3. every ``.pdf`` is converted with ``pdftotext`` and the extracted text is
   scanned for the forbidden code points.

Exit status is 0 when the repository is clean and 1 when anything is found.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Written as escapes so this file never contains the characters it bans.
# U+2010 and U+2011 are deliberately absent: they are hyphens rather than
# dashes, and pdftotext legitimately emits U+2010 for a typeset hyphen.
FORBIDDEN: dict[str, str] = {
    "\u2012": "FIGURE DASH",
    "\u2013": "EN DASH",
    "\u2014": "EM DASH",
    "\u2015": "HORIZONTAL BAR",
    "\u2e3a": "TWO EM DASH",
    "\u2e3b": "THREE EM DASH",
    "\ufe58": "SMALL EM DASH",
    "\ufe63": "SMALL HYPHEN MINUS",
    "\uff0d": "FULLWIDTH HYPHEN MINUS",
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "build",
    "_minted",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".tex",
    ".bib",
    ".cls",
    ".sty",
    ".txt",
    ".toml",
    ".cfg",
    ".ini",
    ".yml",
    ".yaml",
    ".json",
    ".html",
    ".css",
    ".js",
    ".jsonl",
    ".csv",
    ".sh",
    ".gitignore",
}

TEXT_NAMES = {"Makefile", "LICENSE", ".gitignore", "makefile"}

LATEX_SUFFIXES = {".tex", ".bib", ".cls", ".sty"}

# Environments whose bodies render hyphens literally, so ``--`` is safe there.
VERBATIM_BEGIN = re.compile(r"\\begin\{(verbatim|lstlisting|Verbatim|minted|alltt)\}")
VERBATIM_END = re.compile(r"\\end\{(verbatim|lstlisting|Verbatim|minted|alltt)\}")
INLINE_VERB = re.compile(r"\\verb\*?(.)(?:(?!\1).)*\1|\\url\{[^}]*\}|\\href\{[^}]*\}")
HYPHEN_RUN = re.compile(r"-{2,}")


class Finding:
    """One forbidden occurrence, ready to print as a compiler style line."""

    def __init__(self, path: Path, line: int, column: int, detail: str) -> None:
        self.path = path
        self.line = line
        self.column = column
        self.detail = detail

    def render(self, root: Path) -> str:
        try:
            shown = self.path.relative_to(root)
        except ValueError:
            shown = self.path
        return f"{shown}:{self.line}:{self.column}: {self.detail}"


def is_text_file(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES


def scan_code_points(path: Path, text: str) -> list[Finding]:
    found: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for column, character in enumerate(line, start=1):
            name = FORBIDDEN.get(character)
            if name is not None:
                code = f"U+{ord(character):04X}"
                found.append(Finding(path, line_number, column, f"{name} ({code})"))
    return found


def strip_inline_verbatim(line: str) -> str:
    return INLINE_VERB.sub("", line)


def scan_latex_hyphens(path: Path, text: str) -> list[Finding]:
    found: list[Finding] = []
    in_verbatim = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if VERBATIM_BEGIN.search(line):
            in_verbatim = True
            continue
        if VERBATIM_END.search(line):
            in_verbatim = False
            continue
        if in_verbatim:
            continue
        comment = line.find("%")
        if comment != -1 and (comment == 0 or line[comment - 1] != "\\"):
            line = line[:comment]
        for match in HYPHEN_RUN.finditer(strip_inline_verbatim(line)):
            run = match.group(0)
            rendered = "em dash" if len(run) >= 3 else "en dash"
            found.append(
                Finding(
                    path,
                    line_number,
                    match.start() + 1,
                    f"{len(run)} hyphens in LaTeX source render as an {rendered}",
                )
            )
    return found


def scan_pdf(path: Path, pdftotext: str) -> list[Finding]:
    with tempfile.TemporaryDirectory() as work:
        out = Path(work) / "extracted.txt"
        result = subprocess.run(
            [pdftotext, "-enc", "UTF-8", "-q", str(path), str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not out.exists():
            message = result.stderr.strip() or f"pdftotext exited {result.returncode}"
            return [Finding(path, 0, 0, f"could not extract PDF text: {message}")]
        text = out.read_text(encoding="utf-8", errors="replace")
    return scan_code_points(path, text)


def git_tracked_files(root: Path) -> list[Path] | None:
    """List the files git would ship, or None when this is not a git checkout.

    Scoping the guard to git's view is the point: ignored files such as the
    virtual environment and build scratch are not part of the repository, and
    files that are merely untracked so far still are.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return [root / name for name in result.stdout.split("\0") if name]


def iter_files(root: Path) -> list[Path]:
    tracked = git_tracked_files(root)
    candidates = tracked if tracked is not None else list(root.rglob("*"))
    files: list[Path] = []
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def check(root: Path, skip_pdf: bool = False) -> tuple[list[Finding], int, int]:
    pdftotext = shutil.which("pdftotext")
    findings: list[Finding] = []
    text_checked = 0
    pdf_checked = 0

    for path in iter_files(root):
        if path.suffix.lower() == ".pdf":
            if skip_pdf or pdftotext is None:
                continue
            findings.extend(scan_pdf(path, pdftotext))
            pdf_checked += 1
            continue
        if not is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        text_checked += 1
        findings.extend(scan_code_points(path, text))
        if path.suffix in LATEX_SUFFIXES:
            findings.extend(scan_latex_hyphens(path, text))

    if pdftotext is None and not skip_pdf:
        print(
            "warning: pdftotext was not found on PATH, so PDFs were not checked",
            file=sys.stderr,
        )
    return findings, text_checked, pdf_checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="directory to scan (defaults to the current directory)",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="skip the compiled PDF check, for use before the reports exist",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings, text_checked, pdf_checked = check(root, skip_pdf=args.skip_pdf)

    for finding in findings:
        print(finding.render(root))

    scope = f"{text_checked} text files and {pdf_checked} PDFs"
    if findings:
        print(f"\nFAIL: {len(findings)} dash violations across {scope}.")
        return 1
    print(f"OK: no em dashes or en dashes in {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
