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

Status: complete, 2026-08-05.

Built: `Experiment` and `MetricSeries`, the SQLite `Store`, the three parsers
behind a common `Parser` contract, run discovery, and the resumable `ingest`
pass. Fixtures under `tests/fixtures` are committed and regenerated only on
purpose, by `scripts/make_fixtures.py`.

Two decisions worth recording.

**Series are float32 in memory, not only on disk.** The store writes float32
blobs, so holding float64 in memory would have made a round trip lossy in a way
that no test could have called exact. Fixing the model at the storage precision
turns "round trip preserves the series" into a bit for bit assertion, which is
what `test_round_trip_is_bit_for_bit` now checks.

**All four fixtures encode one identical series.** A TensorBoard event file, a
wide CSV, a long CSV and a JSONL log all carry the same 60 point float32 series.
That makes `test_all_formats_agree_exactly` possible, which is a far stronger
statement than four separate smoke tests: it proves the parsers are
interchangeable, which is the premise the whole comparison layer rests on.

Checks run at the close of this phase:

```text
$ ruff format --check . && ruff check .
19 files already formatted
All checks passed!

$ pytest tests/unit -q
40 passed in 0.67s

$ python scripts/check_no_dashes.py .
OK: no em dashes or en dashes in 33 text files and 0 PDFs.
```

## Phase 2: statistics and the calibration suite

Status: complete, 2026-08-05.

Built: `triage/analysis/comparison.py` with both permutation modes, effect
sizes and intervals; `triage/synthetic.py` with the controlled curve generator;
the calibration suite in `tests/statistics`; and 29 fast mechanics tests.

### The calibration suite found a real defect, as expected

The first full run of the calibration suite measured the window block mode at a
**type I error of 12 percent against a nominal 5**, well outside the [2, 8]
gate. The seed replicated mode passed on its first run at 4.6 percent. Two root
causes, both found by measurement rather than reading:

1. **The autocorrelation time was estimated on the two windows concatenated.**
   Joining two runs end to end puts a step change at the join, which the
   estimator reads as long range dependence. It inflated tau by roughly a factor
   of two, and inflating tau is not even the safe direction here, because the
   block length was then clipped by a target block count that quietly undid it.
   Fixed by estimating tau on each window separately and taking the larger.

2. **The block length was one tau, and one tau is not enough.** At one
   autocorrelation time the block means are still visibly correlated with their
   neighbours. A permutation that scatters neighbouring blocks across both
   groups then produces a null distribution narrower than the truth, which is
   exactly how a test becomes anticonservative. Measured across window lengths
   and autocorrelation levels, three tau puts the error back on nominal.

There was a third decision hiding behind the second. Once the block is three
tau, a short run may not contain enough blocks, and the obvious fix, shortening
the block to fit, is precisely the failure above. So the mode now **refuses**:
below eight blocks per run it raises `ComparisonError` naming the remedy rather
than returning a p value it cannot stand behind.

### Measured calibration, all gates met

| Measurement | Result | Gate |
|---|---|---|
| Seed replicated, type I error | 4.53 percent (+/- 0.74), 3000 null cases | [2, 8] percent |
| Seed replicated, type I across seed sigma 0.005 to 0.05 | 4.40, 4.70, 5.00 percent | [2, 8] percent |
| Seed replicated, power at a large effect | 96.25 percent, 800 cases | above 90 percent |
| Window block, type I error | 5.81 percent (+/- 1.04), 1928 cases, 72 refused | [2, 8] percent |
| Window block, power at a large effect | 100 percent, 291 cases | above 90 percent |
| Null p values uniform at 0.05, 0.10, 0.25, 0.50 | 0.058, 0.099, 0.241, 0.485 | within 3 standard errors |

Checking one alpha can be passed by a test that is wrong in two compensating
directions, so the last row checks the whole null distribution rather than one
threshold.

### The number that justifies the labelling

The same window block test, on runs that carry realistic seed variance and a
true effect of exactly **zero**:

| Seed standard deviation | False positive rate |
|---|---|
| 0.01 | 53.2 percent |
| 0.02 | 77.0 percent |
| 0.04 | 87.4 percent |

The seed replicated mode holds at 4.4 to 5.0 percent on the same data. This is
measured, not argued, and it is why the single run mode is marked as the weaker
claim in every output this tool produces.

Checks run at the close of this phase:

```text
$ pytest tests/statistics -q
9 passed in 58.83s

$ pytest tests/unit -q
69 passed in 1.19s

$ ruff format --check . && ruff check .
All checks passed!
```

## Phase 3: regression, sensitivity, HTML report

Status: pending.

## Phase 4: documentation from measured numbers

Status: pending.

## Phase 5: main report and debug report

Status: pending.

## Phase 6: final QA

Status: pending.
