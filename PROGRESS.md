# Build progress

Running record of the build. One section per phase, each closed only after its
checks were run and the output pasted in.

## Toolchain, verified at Phase 0

Target machine: Intel Core i7-14700K, 32 GB DDR5, Windows 11 Pro 10.0.26200,
run natively. The RTX 5070 in this machine is not used by this project at any
point; every computation here is CPU only.

| Component | Version | Note |
|---|---|---|
| Python | 3.13.14 | installed through the `py` launcher at Phase 0 |
| numpy | 2.5.1 | |
| pandas | 3.0.5 | |
| scipy | 1.18.0 | |
| plotly | 6.9.0 | |
| jinja2 | 3.1.6 | |
| tensorboard | 2.21.0 | event file reader only |
| tqdm | 4.70.0 | |
| ruff | 0.16.1 | formatter and linter |
| pytest | 9.1.1 | |
| git | 2.53.0.windows.2 | |
| MiKTeX texify | on PATH | primary PDF driver |
| pdftotext | on PATH | used by the dash guard to check compiled PDFs |
| GNU make | ezwinports build | resolves `SHELL` to Git's `sh.exe`, so one Makefile serves Windows and CI |

### Substitutions and deviations from the specification

1. **Python version.** The machine had only 3.14 registered. The specification
   asks for 3.13, so 3.13.14 was installed with `py install 3.13` rather than
   building against 3.14, where the tensorboard and protobuf wheel situation is
   still moving. Recorded here because it is a deliberate choice, not a default.
2. **Package versions.** Every package named in the specification is still
   current and was installed at its latest release. Nothing was substituted.
   The exact versions are pinned in `requirements.txt`.
3. **PDF driver.** `texify` is the driver on Windows as specified.
   `scripts/build_pdf.py` falls back to an explicit pdflatex and bibtex
   sequence where `texify` does not exist, which is what CI uses on Ubuntu.
4. **Build specification file.** The specification this repository was built
   from is kept out of git history. It is an input to the work rather than part
   of the shipped project.

## Phase 0: environment and the dash guard

Status: complete, 2026-08-05.

Built: `.venv` on Python 3.13.14, `pyproject.toml`, pinned `requirements.txt`,
MIT `LICENSE`, `.gitignore`, the portable `Makefile`, and three scripts,
`check_no_dashes.py`, `build_pdf.py` and `clean.py`.

The dash guard scopes itself to the files git would ship
(`git ls-files --cached --others --exclude-standard`) rather than walking the
filesystem. That decision was forced by its first run, which correctly flagged
an em dash inside the build specification file sitting in the working
directory. That file is ignored and is not part of the repository, so a
filesystem walk was answering the wrong question. The guard checks three
things: forbidden code points in text files, runs of two or more hyphens in
LaTeX sources outside verbatim environments (which is how `--` becomes a
typeset en dash), and the text extracted from every compiled PDF by
`pdftotext`.

Checks run at the close of this phase:

```text
$ ruff format --check . && ruff check .
4 files left unchanged
All checks passed!

$ python scripts/check_no_dashes.py .
OK: no em dashes or en dashes in 9 text files and 0 PDFs.
```

A TensorBoard write and read round trip was also confirmed on this toolchain
before any parser code was written, so that a later parser failure could not be
confused with a broken dependency.

## Phase 1: model, store and parsers

Status: pending.

## Phase 2: statistics and the calibration suite

Status: pending.

## Phase 3: regression, sensitivity, HTML report

Status: pending.

## Phase 4: documentation from measured numbers

Status: pending.

## Phase 5: main report and debug report

Status: pending.

## Phase 6: final QA

Status: pending.
