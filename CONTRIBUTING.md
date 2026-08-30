# Contributing

## Getting set up

```bash
make env       # Python 3.13 venv with the pinned requirements
make test-unit # the inner loop: everything but the calibration gates, about 50 s
make test      # the full suite, calibration included, about 3 minutes
```

`noxfile.py` holds the actual recipes and the Makefile forwards to it, so
`nox -s test -- -m "not slow"` and `make test-unit` are the same command and
either works on any platform. `nox -l` lists every session.

## The inner loop

**`pytest -m "not slow"` is what to run while working.** The `slow` marker is
on the calibration suite: it runs about 24,000 permutation comparisons and takes
around two minutes, which is the right price before a pull request and the wrong
one after every edit. `make test-unit`, `nox -s test -- -m "not slow"` and
`pytest -m "not slow"` are three spellings of it.

Run the gates themselves with `make test-stats` before pushing anything that
touches `triage/analysis/`.

## What the build will refuse

`make all` runs these in order and stops at the first failure.

1. **`ruff format --check` and `ruff check`.** Line length 100. Run `make fmt` to
   fix what is fixable.
2. **The dash guard,** over prose and source. No em dashes or en dashes, including
   inside compiled PDFs. In LaTeX prose this also means never typing `--` or
   `---`, because both become dashes when typeset. Page ranges in `refs.bib` use
   a single hyphen for the same reason. Verbatim blocks are exempt, since hyphens
   there are printed literally. Data files (`.json`, `.jsonl`, `.csv`) are not
   scanned: a dash in a fixture is content this project carries rather than text
   it wrote.
3. **`mypy --strict triage`,** with no per module exceptions for this project's
   own code.
4. **The full test suite,** including the statistical calibration gates.
5. **`make verify-demo`,** which rebuilds the synthetic sweep from fixed seeds
   and checks every verdict against the committed record to four decimal places.
6. **`make package`,** which builds the wheel, runs `twine check --strict`, and
   installs it into two throwaway environments: one with `[cli]`, which has to
   ingest a log and write a report, and one bare, which has to import and do
   statistics with numpy and scipy alone.

## The rule that matters most

**A calibration miss is a stop the line defect.** If a change moves the measured
type I error outside [2, 8] percent, or power below 90, the change is wrong until
proven otherwise. Do not widen the gates to make a change pass. The gates are the
product.

If you change anything in `triage/analysis/comparison.py`, run
`make test-stats` and put the measured numbers in the pull request.

## If a demo verdict changes

`make verify-demo` failing means a number moved. That is either a bug or an
intended improvement, and the difference matters:

* **A bug.** Fix it. The record is the reference.
* **Intended.** Run `python -m examples.demo_workflow --update-record`, commit
  the new record in the same change, and say in the commit message which numbers
  moved and why.

Never update the record to silence a failure you have not explained.

## Writing statistical code

Every statistical function's docstring states its null hypothesis and its
assumptions. If you cannot write down the null hypothesis, the function is not
ready.

Where a method has a limitation, document the limitation and add a test that
pins it, rather than choosing data that hides it. There are two of these already
(rank correlation blind to a U shape, and the single run mode blind to seed
variance) and both are more useful to a reader than a clean result would have
been.

**If the claim is true of every input, write it as a property.** `tests/property/`
holds the ones that are: permutation p values invariant to the order the runs
arrived in, Benjamini Hochberg monotone in its input vector, a store round trip
exact for any finite float32 series, a parser handing back the tag it was given
whatever text that tag is made of. They run inside the ordinary suite at the
`fast` Hypothesis profile; `nox -s test_property` reruns them at `thorough`,
which is what to do after touching `triage/analysis/` or the store. Hypothesis
records a failing input under `.hypothesis/` and replays it first, so a
counterexample stays reproducible locally, but the fix belongs in the tests as
an example rather than in that cache.

## Style

* `snake_case`, `PascalCase`, `UPPER_SNAKE`. Parser classes end in `Parser`.
* Full type hints on anything public.
* CLI verbs are `ingest`, `compare`, `report`.
* Comments explain why, not what. If a constant was chosen by measurement, say
  what was measured.
* No TODO stubs on `main`.

## Commits

One commit per meaningful unit, present tense, plain prose. If a change was
forced by a measurement, put the measurement in the message.
