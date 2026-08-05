"""Compile a LaTeX document to PDF on Windows (MiKTeX) or Linux (TeX Live).

On Windows the preferred driver is MiKTeX `texify`, which decides for itself how
many pdflatex passes and whether bibtex is needed. Where `texify` is absent, for
example on the Ubuntu runner used by CI, the script falls back to an explicit
pdflatex, bibtex, pdflatex, pdflatex sequence, which produces the same document.

The finished PDF is left beside its source (`report/main.pdf`) because that is
where the dash guard and the release assets expect it. Every intermediate file
is swept into a `build/` directory next to the source so a failed run leaves a
readable log behind.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

INTERMEDIATE_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".idx",
    ".ilg",
    ".ind",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".synctex.gz",
    ".toc",
}


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print(f"  $ {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def show_log_tail(log: Path, lines: int = 40) -> None:
    if not log.exists():
        return
    text = log.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"\n--- tail of {log.name} ---")
    for line in text[-lines:]:
        print(line)
    print("--- end of log ---\n")


def build_with_texify(texify: str, source: Path) -> bool:
    # texify decides the pass count and runs bibtex itself when it sees
    # citations, so the bibtex flag is not passed through to it.
    # The flag is `--batch`, not `--batch-mode`: the latter is pdflatex's
    # spelling and texify rejects it, which sent every build down the fallback
    # path until it was noticed.
    command = [texify, "--pdf", "--batch", "--quiet", source.name]
    result = run(command, source.parent)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:], file=sys.stderr)
    return result.returncode == 0


def build_with_pdflatex(source: Path, use_bibtex: bool) -> bool:
    workdir = source.parent
    stem = source.stem
    latex = ["pdflatex", "-interaction=batchmode", "-halt-on-error", source.name]

    if run(latex, workdir).returncode != 0:
        return False
    if use_bibtex:
        bibtex = run(["bibtex", stem], workdir)
        if bibtex.returncode != 0:
            print(bibtex.stdout[-2000:])
    # Two further passes settle the table of contents and the citation labels.
    return all(run(latex, workdir).returncode == 0 for _ in range(2))


def sweep_intermediates(source: Path) -> Path:
    workdir = source.parent
    build = workdir / "build"
    build.mkdir(exist_ok=True)
    for path in sorted(workdir.iterdir()):
        if not path.is_file() or path.stem != source.stem:
            continue
        if path.suffix in INTERMEDIATE_SUFFIXES:
            shutil.move(str(path), str(build / path.name))
    return build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="path to the .tex file to compile")
    parser.add_argument(
        "--bibtex",
        action="store_true",
        help="the document cites a bibliography and needs a bibtex pass",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "texify", "pdflatex"],
        default="auto",
        help="force a driver instead of picking one from what is installed",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"error: {source} does not exist", file=sys.stderr)
        return 1

    texify = shutil.which("texify")
    engine = args.engine
    if engine == "auto":
        engine = "texify" if texify else "pdflatex"
    if engine == "texify" and texify is None:
        print("error: texify was requested but is not on PATH", file=sys.stderr)
        return 1

    print(f"building {source.name} with {engine}")
    if engine == "texify":
        ok = build_with_texify(texify, source)
        if not ok and shutil.which("pdflatex"):
            print("texify failed, retrying with the explicit pdflatex sequence")
            ok = build_with_pdflatex(source, args.bibtex)
    else:
        ok = build_with_pdflatex(source, args.bibtex)

    pdf = source.with_suffix(".pdf")
    build = sweep_intermediates(source)

    if not ok or not pdf.exists():
        show_log_tail(build / f"{source.stem}.log")
        print(f"FAIL: {source.name} did not produce a PDF", file=sys.stderr)
        return 1

    size_kb = pdf.stat().st_size / 1024
    print(f"OK: {pdf.name} written ({size_kb:.0f} KB), intermediates in {build.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
