"""Prose wide guard against em dashes, en dashes, and LaTeX dash ligatures.

Ground rule 1 of this project forbids U+2014 and U+2013 in the text this
project writes, including inside the compiled PDFs. LaTeX turns ``--`` into an
en dash and ``---`` into an em dash, so plain ASCII source can still smuggle a
dash into a PDF. This script therefore runs three checks:

1. every prose or source file is scanned for the forbidden code points;
2. every ``.tex`` and ``.bib`` file is scanned for runs of two or more hyphens
   outside verbatim style environments, which is how the ligature sneaks in;
3. every ``.pdf`` is converted with ``pdftotext`` and the extracted text is
   scanned for the forbidden code points.

**The scope is prose, not data.** ``.json``, ``.csv`` and ``.jsonl`` are
deliberately not scanned. A rule about typography in English writing has no
business rewriting a fixture, a config, or a log row: those files record what
some other system produced, and a dash inside one is content that this project
is carrying rather than text that this project wrote. Scanning them meant the
only way to keep a fixture containing a legitimate dash was to corrupt the
record it existed to be.

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
#
# The three CJK compatibility forms that used to sit here (U+FE58, U+FE63,
# U+FF0D) are gone for the same reason. U+FF0D is the fullwidth spelling of an
# ASCII hyphen and U+FE63 its small form: banning them made a Japanese or
# Chinese string a violation of a rule about typography in English prose, which
# is not a rule this project has. Nothing here has ever written one, so the
# entries were unreachable as well as wrong.
FORBIDDEN: dict[str, str] = {
    "\u2012": "FIGURE DASH",
    "\u2013": "EN DASH",
    "\u2014": "EM DASH",
    "\u2015": "HORIZONTAL BAR",
    "\u2e3a": "TWO EM DASH",
    "\u2e3b": "THREE EM DASH",
}

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    "_minted",
    # Local working material. git already hides it, so this only matters on the
    # fallback walk used when the guard is run outside a checkout.
    "private",
}

# Prose and the source that carries prose: docstrings, comments, templates,
# help text and configuration a person reads. `.json`, `.jsonl` and `.csv` are
# absent on purpose (see the module docstring): they hold data rather than
# writing, and this guard has no standing over what another system logged.
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
    ".html",
    ".css",
    ".js",
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


def is_excluded(path: Path, root: Path) -> bool:
    """True when `path` lies under one of `EXCLUDED_DIRS` inside `root`.

    The names in `EXCLUDED_DIRS` are directories of this repository, so they
    only mean anything relative to the tree being scanned. Matching them
    against the absolute path instead made the guard's answer depend on where
    the tree happened to sit: on macOS a temporary directory is
    `/private/var/folders/...`, whose first component is `private`, so every
    file under one was skipped and the guard reported a clean scan of nothing.
    """
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in EXCLUDED_DIRS for part in parts)


def iter_files(root: Path) -> list[Path]:
    tracked = git_tracked_files(root)
    candidates = tracked if tracked is not None else list(root.rglob("*"))
    files: list[Path] = []
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if is_excluded(path, root):
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
