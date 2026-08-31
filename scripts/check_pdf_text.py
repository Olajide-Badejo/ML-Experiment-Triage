"""The committed PDFs say what a fresh build of this tree says.

The two reports are compiled artifacts that live in the repository, and the CI
job that rebuilds them only proved that they compile. Whether the file somebody
downloads from the repository is the document this source produces was nobody's
job, so a figure regenerated from a changed database, or a paragraph edited in
the PDF viewer's source and never recompiled, would have gone unnoticed.

Comparison is on extracted text rather than on bytes. A PDF carries a creation
timestamp, a producer string and a document id, so two compilations of one
unchanged source are never byte identical; the words on the pages are.

    python scripts/check_pdf_text.py --snapshot text-before   # before rebuilding
    python scripts/check_pdf_text.py --compare  text-before   # after rebuilding

Exit status is 0 when every page of every report reads the same and 1 when
anything differs, with a unified diff naming the file and the line.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The compiled documents, relative to the repository root.
REPORTS = (
    Path("report") / "main.pdf",
    Path("report_debug") / "debug_report.pdf",
)


def snapshot_name(pdf: Path) -> str:
    """`report/main.pdf` becomes `report__main.txt`, which is a legal filename."""
    return "__".join(pdf.with_suffix("").parts) + ".txt"


def extract(pdf: Path) -> list[str]:
    """The text of one PDF as lines, normalised for comparison.

    Trailing whitespace is stripped because the column geometry decides it and
    it is not content. The page separator that pdftotext writes is kept, so a
    document that gains or loses a page shows up as a difference rather than as
    a long unexplained shift.
    """
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise SystemExit("pdftotext was not found on PATH; install poppler utils")
    with tempfile.TemporaryDirectory() as work:
        out = Path(work) / "extracted.txt"
        result = subprocess.run(
            [pdftotext, "-enc", "UTF-8", "-q", str(pdf), str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not out.exists():
            message = result.stderr.strip() or f"pdftotext exited {result.returncode}"
            raise SystemExit(f"{pdf}: could not extract text: {message}")
        text = out.read_text(encoding="utf-8", errors="replace")
    return [line.rstrip() for line in text.splitlines()]


def snapshot(destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for pdf in REPORTS:
        source = ROOT / pdf
        if not source.exists():
            raise SystemExit(f"{pdf} is not in the tree, so there is nothing to snapshot")
        (destination / snapshot_name(pdf)).write_text(
            "\n".join(extract(source)) + "\n", encoding="utf-8"
        )
        print(f"snapshotted {pdf.as_posix()}")
    return 0


def compare(saved: Path) -> int:
    differences = 0
    for pdf in REPORTS:
        expected_file = saved / snapshot_name(pdf)
        if not expected_file.exists():
            raise SystemExit(f"no snapshot for {pdf.as_posix()} under {saved}")
        expected = expected_file.read_text(encoding="utf-8").splitlines()
        actual = extract(ROOT / pdf)
        diff = list(
            difflib.unified_diff(
                expected,
                actual,
                fromfile=f"{pdf.as_posix()} (committed)",
                tofile=f"{pdf.as_posix()} (rebuilt)",
                lineterm="",
                n=2,
            )
        )
        if diff:
            differences += 1
            print("\n".join(diff))
        else:
            print(f"OK: {pdf.as_posix()} reads the same as the committed copy")

    if differences:
        print(
            f"\nFAIL: {differences} report(s) differ from what this tree builds. "
            f"Recompile them with `make pdfs` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", metavar="DIR", help="write the current PDFs' text into DIR")
    group.add_argument("--compare", metavar="DIR", help="diff the current PDFs against DIR")
    args = parser.parse_args(argv)

    if args.snapshot:
        return snapshot(Path(args.snapshot))
    return compare(Path(args.compare))


if __name__ == "__main__":
    raise SystemExit(main())
