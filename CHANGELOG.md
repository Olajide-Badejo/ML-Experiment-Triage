# Changelog

All notable changes to this project are recorded here. Format follows Keep a
Changelog; versions follow semantic versioning.

## [1.1.0] 2026-08-30

The release that closes the consumer contract. Read
[For ml-experiment-triage consumers](#for-ml-experiment-triage-consumers) first
if you import this package: it names the five filed issues by number and lists
every behaviour change that can reach code you have already written.

Eleven P0 defects are fixed, the false discovery rate the reports always claimed
to be applying is now actually applied, the calibration suite grew from one
design to seven and publishes the arms that do not flatter, and the package
gained a reference workload, a local model layer, a public API and a
documentation site.

### Added

* **A public API.** `dir(triage)` used to be empty: every name was reachable
  only through a deep module path, which meant this package had an API in
  practice and none on paper. `triage.__all__` is now the list the consumer
  filed (E1 part 3): `Experiment`, `MetricSeries`, `Store`, `StoreError`,
  `ingest`, `IngestResult`, `discover_runs`, `ParseError`, `ComparisonConfig`,
  `ComparisonResult`, `compare_all`, `paired_permutation`, `RegressionConfig`,
  `classify`, `rank`, `TriageReport`, `analyse`, `build_context`, `render`. Most
  resolve lazily on first access, so importing `triage` does not import plotly
  and jinja2; `ingest` is bound eagerly because it is also the name of a
  submodule. The deep paths keep working unchanged and hand back the same
  objects, so an `isinstance` check cannot depend on which import line you
  wrote. A 23 test module pins the contract.
* **`paired_permutation`, a paired test on any statistic, with clustered
  resampling** (E3). Label swap permutation within a unit, with whole clusters
  swapping together when `clusters` is given, exhaustive enumeration while
  `2**n_clusters` is affordable and `min_attainable_p = 2 / 2**n_clusters`. The
  statistic is a callable because the quantity compared is often not a mean of
  anything: macro F1 over a split cannot be written as an average of per row
  numbers. Documented constraint: the callable must be a pure function of the
  vector it is handed, because the cluster bootstrap that produces the interval
  passes a vector of a different length in a different order.
* **`Outcomes` and `OutcomesParser`, for evaluations that are not time series**
  (E5). One row per classified item, named numeric or boolean fields plus
  categorical group keys, no step axis. `--outcomes` reads generic step free
  JSONL as this shape, `pair_on()` is the row aligned join a paired test needs,
  and the refusal half is enforced: `compare_window_block` and anything windowed
  refuse an `Outcomes` input by construction, with a message naming
  `paired_permutation` as the right tool.
* **An MLflow parser, for both of MLflow's backends, with no new dependency.**
  `triage ingest` reads a local `mlflow.db` through the standard library's
  `sqlite3` and an `mlruns/` file store as the plain text it is, so an MLflow
  sweep compares like any other and a core install (numpy and scipy) can do it.
  Params become the run config with numbers read as numbers, metric keys keep
  their slashes, MLflow's `is_nan` flag is turned back into the NaN it stands
  for and dropped where every other format's is, and soft deleted runs are not
  resurrected. One tracking database holds many runs, so `Parser.runs_in`
  expands a claimed source into one path per run and each is named
  `<database>/<run_uuid>`, which keeps run identity unique per D4. Fixtures for
  both layouts are committed and encode the same reference series as every
  other parser fixture.
* **Hypothesis property tests over the statistical core** (`tests/property/`).
  The gates this project publishes are already phrased as properties, so they
  are encoded as ones: permutation p values invariant to the order the runs
  arrived in, `paired_permutation` invariant to the order of the pairs,
  Benjamini Hochberg order preserving and monotone in its input vector and
  matching the published 1995 example, a store round trip exact for any finite
  float32 array, and parser fuzzing with hostile tag text (D3, D28). Two
  profiles: `fast` runs inside the ordinary suite, and `nox -s test_property`
  reruns them at `thorough`. Hypothesis is a development dependency only.
* **A `.pre-commit-config.yaml`,** so that what CI refuses is refused before the
  commit exists: ruff format, ruff check, the dash guard scoped as it always was
  (D35g), `check-yaml` and `end-of-file-fixer`. The slow gates stay in CI, and
  `end-of-file-fixer` is kept away from `tests/fixtures/` for the same reason
  the dash guard is: a fixture records the bytes another system wrote, and
  MLflow's file store writes a param value with no trailing newline. The ruff
  revision is pinned to the locked version and a test holds them in step.
  `pre-commit install` is documented in CONTRIBUTING.md and is optional:
  a contributor without it gets the same answer from CI a few minutes later.
* **Six new calibration arms, covering the designs a permutation test is not
  exact in.** Unequal spreads at 5v5, 3v7 and 7v3 (7.83, 2.08 and 12.92 percent
  type I at a nominal 5, against 8.25, 1.08 and 17.92 for the raw mean
  difference the studentized statistic replaced); unequal counts at one spread
  (4.25); heavy tailed Student t noise at three degrees of freedom (5.00); the
  paired clustered mode on a clustered null (4.90) and on macro F1 (2.00), with
  the same clustered data scored ignoring the clustering (12.10) as the
  measurement that justifies the `clusters` argument. Three gates rather than
  one, each set to what its regime supports and each stated with its reason in
  `docs/ENGINEERING_LOG.md`. Every rate is published in `triage/calibration.py`
  whether or not it flatters.
* `triage.synthetic.null_pair_designed` for a null whose two sides differ in
  spread or in count, `CurveSpec.tail_df` for Student t noise rescaled to the
  spread it replaces, and `clustered_null_pair` for paired scores whose
  differences are correlated inside a cluster. The Gaussian path draws in
  exactly the order it did before, asserted by a test.
* `scripts/render_calibration_docs.py`, which renders the measured numbers into
  every document that quotes one (the README, the documentation site's landing
  page, the methodology and the design decisions) from `triage/calibration.py`,
  the same single source the LaTeX tables already came from. `--check` runs in
  CI, so no published number can be edited in one place and left in another.
* **`RegressionConfig.practical_threshold_absolute`, a practical gate in the
  metric's own units** (`--practical-threshold-absolute` on the command line).
  Exactly one of `practical_threshold_pct` and `practical_threshold_absolute`
  may be set, `ValueError` when both or neither, and `describe()` prints the one
  in force. A relative gate divides by the baseline and a baseline can be zero,
  in which case the relative effect is reported as 0.0 rather than as an
  infinity and a genuine regression on a zero crossing metric was silently
  downgraded to "significant but below the practical threshold". Both fields are
  now `float | None`. The `practical_threshold` constructor keyword is accepted
  as a deprecated alias of `practical_threshold_pct` with a `DeprecationWarning`
  and is removed in 1.2.0.
* **An extras split, so a core install is numpy and scipy** (E1). `pip install
  ml-experiment-triage` now gets the statistics API, the JSONL parser and the
  MLflow parser and nothing else; `parsers`, `report`, `cli`, `agentic` and
  `all` are opt in. A module that needs an extra imports it at the point of use
  and raises a message naming the extra, rather than failing at import with a
  traceback about a package nobody asked for. `tqdm` is not an error at all: its
  absence degrades the progress bar to plain lines. A CI job installs the bare
  core and does statistics with it, and another installs from the declared
  version ranges alone. `triage/py.typed` ships, so the consumer's
  `ignore_missing_imports` override for `triage.*` can be deleted.
* **An exit code table, so a CI gate can tell the cases apart.** 0 success, 1 a
  bug in triage with its traceback, 2 a usage error or a foreseen failure, 3 the
  work was done but at least one run failed to parse, 4 the run performed zero
  comparisons. `triage --help` prints it and `EXIT_CODES` holds it.
* **`triage demo`,** which synthesises the sweep into a temporary directory and
  produces the report, so a pip installed tool can demonstrate itself with no
  clone and no data.
* **`triage --version`,** and `--quiet`, which is now honoured (the count
  summary only; provenance lines stay).
* **The autofill vertical (`triage/autofill/`), this repository's reference
  workload.** A form field classification problem carried end to end: a WHATWG
  token taxonomy shared with the sibling repositories, a byte deterministic
  corpus generator whose noise knobs are recorded in `meta.json`, hashed
  character n gram features, a numpy softmax head, temperature scaling, an
  evaluation that emits both artifact shapes, and a `triage autofill` verb with
  `generate`, `sweep`, `evaluate` and `agentic` subcommands. There is **zero
  autofill specific code in the ingest, parser, comparison or reporting
  layers**: the sweep drops into `triage ingest` and `triage compare` as any
  other sweep would. On de_DE validation the model reaches 0.9643 macro F1
  against the heuristic's 0.7235 with a seed replicated verdict at p 0.0079, and
  temperature scaling takes expected calibration error from 0.0106 to 0.0035.
* **A fill or skip decision layer (`triage/autofill/policy.py`).** Four policies
  over one threshold class, evaluated exactly offline rather than by importance
  sampling, plus an online Thompson sampling bandit over 40 contextual arms that
  updates only on the fills it actually made. On de_DE validation the per type
  threshold policy is chosen at 0.9233 expected reward against always filling at
  0.8815. The contrary out of sample result is published beside it rather than
  omitted: on the test split, where the head is 98.5 percent accurate, always
  filling wins.
* **An agentic demo (`triage autofill agentic`, extra `[agentic]`).** The
  trained classifier drives headless Chrome over generated pages, fills them
  under the exported policy, and scores every value by reading it back out of
  the DOM. Six pages, two locales, 71 fields: the model fills 64 at a fill
  accuracy of 1.0000 where the keyword baseline fills 30 at 0.8667. The task
  metrics are then compared by this tool through the ordinary verbs, at +151.40
  percent reward against the baseline, adjusted p 0.0002. `--no-browser` parses
  the same page with the standard library and every number agrees, which is what
  makes the CI run a test of the demo rather than of a second implementation.
* **A local LLM and retrieval layer (`triage/llm/`), Ollama on localhost, stdlib
  only.** A minimal client, an embedding store, a kNN annotator, a grounded
  summarizer, and `triage llm ask` (aliased as `triage ask`) over the prose
  corpus. Measured rather than asserted: retrieval beats zero shot at 0.8879
  against 0.8007 macro F1 (p 0.0079, adjusted 0.0212), and the numpy head at
  0.9381 beats the 12B model while being about a million times faster, which is
  documented rather than hidden. The grounding pass deletes any generated
  sentence containing a number the report's own table does not support. Exact
  search over the corpus takes 15.5 microseconds against 2233 milliseconds of
  HTTP, which is why there is no vector database. `triage report --llm-summary`
  is off by default, because a report without it is a byte identical function of
  the database.
* **A documentation site** (Section 7 item 1): mkdocs-material on GitHub Pages
  with an API reference rendered by mkdocstrings from the docstrings themselves,
  built with `--strict` in CI so a dead cross reference is a failing build.
  `nox -s docs-site` builds it locally.
* **`CITATION.cff` and `CODE_OF_CONDUCT.md`.** GitHub renders the citation
  widget from the CFF and the Zenodo integration mints a DOI per release from
  the same metadata. There is deliberately no `.zenodo.json`, which would
  silently override it.
* **A release workflow using PyPI Trusted Publishing** (Section 7 item 3).
  OIDC through `pypa/gh-action-pypi-publish`, PEP 740 attestations on by
  default, no API token in the repository or its secrets, and a TestPyPI path
  for rehearsing the publish without spending a version number.

### Changed

* **The development toolchain is a committed PEP 751 `pylock.toml`, resolved and
  installed with uv, and `requirements.txt` is gone.** *Addressed to
  contributors only: nothing about installing this package changes.* `pip
  install ml-experiment-triage` remains the user path and resolves the ranges in
  `pyproject.toml`, which the `ranges` CI job still exercises with pip. Every
  other CI job installs the lock, which carries exactly the resolution
  `requirements.txt` held, in a standard format with hashes and markers for
  every platform in the matrix. `nox -s lock` recompiles it and is idempotent,
  because uv reads the existing lock as a preference source; `nox -s lock --
  --upgrade` is how versions move, and moving numpy or scipy owes the
  calibration suite a rerun. Dependabot covers `pyproject.toml` and the
  workflows and deliberately does not cover the lock, which it cannot read.
* **The window block mode's block length is four autocorrelation times, not
  three.** *Addressed to anyone comparing single runs:* p values from the weak
  mode move slightly and short runs are refused more often. The multiplier had
  been fixed by measurement against a tau estimated on the smoothed window,
  which this release corrected; re-swept against the corrected estimate over
  2000 null cases a point, type I error runs 8.85, 7.02, 6.28, 6.09 and 6.03
  percent at two through six tau, and the refusal rate on 4000 step runs runs 0,
  6, 20, 40 and 78 percent. Four is the knee of both curves. Measured type I for
  the mode moves from 5.81 percent to **6.28**, and the demo's two weak mode
  rows move by about a hundredth of a percentage point.
* **`RegressionConfig.false_discovery_rate` went from inert to operative, with
  no change in default behaviour.** The statistical gate now reads
  `adjusted_p <= config.false_discovery_rate`, which is the Benjamini Hochberg
  decision the reports have always claimed to be making; through 1.0.0 it read
  `adjusted_p < config.alpha` and the rate was passed to a procedure whose output
  does not depend on it, so `--fdr 0.001`, `--fdr 0.10` and `--fdr 0.90` produced
  byte identical verdicts. The default rate moves from 0.10 to **0.05** so that
  the decision made at defaults is exactly the decision 1.0.0 made: no verdict
  shifts under anyone who does not set the flag. `--fdr` now does what it says.
* **`RegressionConfig.alpha` is retained with a narrowed meaning.** It bounds
  admissibility only: a design whose smallest attainable p value exceeds alpha
  is reported as `inconclusive: the design cannot reach alpha` rather than as no
  change. It no longer gates significance. The field is kept because consumers
  construct `RegressionConfig`, and removing a field is a break; its docstring
  and `describe()` both now say what it does.
* `benjamini_hochberg`'s inert `false_discovery_rate` argument defaults to 0.05
  to match. It never changed the returned adjusted values and still does not.
* **The sensitivity table's inference is corrected end to end.** Seed replicates
  entered the rank correlation as independent points: five learning rate levels
  with eight seeds each reported `rho = -0.594, p = 5.3e-05, n = 40` where the
  correct unit is the condition (n = 5, exact two sided floor 2/120 = 0.0167), a
  p value about four orders of magnitude too small. Runs are now averaged onto
  their variant first. The p value comes from `scipy.stats.permutation_test` over
  the pairings, exact where enumeration is affordable and seeded above it, in
  place of `scipy.stats.spearmanr`'s t approximation, which returned exactly 0.0
  at n = 4 and is invalid under ties. The parameter by metric grid is Benjamini
  Hochberg corrected, results are ranked by what survives that correction before
  by the size of rho, `n_variants` is reported beside `n_runs`, and below five
  conditions no p value is reported at all. `SensitivityResult.p_value` and the
  new `adjusted_p` are therefore `float | None`, and `significant` and
  `n_variants` are new fields.
* **The Benjamini Hochberg family is one per comparison mode, and comparisons
  that cannot reach alpha are excluded from the denominator.** Weak mode (single
  run window block) p values were pooled with seed replicated ones, which
  destroys the false discovery control of the whole family given the weak mode's
  measured 54 to 88 percent false positive rate under seed variance; and
  comparisons whose `min_attainable_p` already exceeds alpha inflated the
  denominator while never being able to be discoveries. Each `Finding` now
  carries the `family` it was corrected in, `TriageReport.inadmissible` and
  `.family_note()` report the held out group, and `triage compare` prints the
  family sizes. Adjusted p values on mixed mode sweeps move as a result; verdicts
  on the demo sweep did not.
* **The severity confidence factor is capped at 2.0, so harm leads the ranking.**
  Its docstring called it a tie breaker while it spanned 1 to 13, which let a 3
  percent regression at p = 1e-12 (39.00) outrank a 10 percent regression at
  p = 0.04 (23.98) and put the smaller problem at the top of the table.
  `TriageReport.best_candidate` now prefers a seed replicated improvement over a
  single run one of any size, and the new `best_finding` returns the finding so
  callers can label the mode when only the weaker one exists.
* **`classify()` now returns findings in input order, not severity order.**
  *Addressed to consumers rejoining findings to results:* one `Finding` comes
  back per `ComparisonResult`, positionally, so `zip(results, classify(results))`
  is a valid join and the `id(finding.result)` workaround can be retired.
  `ComparisonResult.key` (`"variant|tag"`, set by `compare_all`) is the other
  half of that: `Finding.tag` was never unique once a metric was compared across
  several conditions. Callers wanting the old order call the already public
  `rank(findings)`, which is what `triage compare` and the HTML report now do,
  so no output of this tool changed.
* **`compare_all` returns a `ComparisonResults`, not a bare list.** It is a list
  subclass, so anything that iterated or indexed it is unaffected, and it
  carries `refusals`: a comparison that could not be made is now a
  `ComparisonRefusal` record naming the candidate, the tag and the reason,
  rather than a silent absence. Refusals are printed in the CLI caveats and
  rendered in a "Comparisons not made" table in the HTML report (D8).
* **The permuted statistic is the Welch t, not the difference of means.**
  *Addressed to anyone comparing unbalanced designs:* p values move. Balanced
  designs are bit identical, because the studentization is a monotone transform
  of the difference when the two groups have the same size and spread. Unbalanced
  and heteroscedastic ones move a long way, and in the right direction: on a
  7 against 3 design with unequal spreads the type I error at a nominal 5 goes
  from 17.92 percent to 12.92 (Janssen 1997). The same statistic is used in the
  window block mode.
* **`analyse_database` returns an `Analysis` dataclass.** It was a 3 tuple
  through 1.0.0, briefly a 4 tuple when refusals were added, and is now a
  dataclass with named fields, so the next thing that has to be reported does
  not change the arity again. `build_context` gained `refusals=()`.
* **A `compare` that performed zero comparisons exits 4, not 0.** *Addressed to
  anyone who wired `triage compare` into a build:* a run in which every
  comparison was refused used to look exactly like a run in which everything
  passed. This is the change most likely to turn a green build red, and that is
  the point of it.
* **The mode vocabulary is open.** `ComparisonResult.mode_label` and `to_dict`
  did a private table lookup and raised `KeyError` on any mode string they did
  not know, so a caller writing a truthful custom mode could then call neither
  member. An unknown mode now falls back to the mode string itself with a
  generic claim label.
* **Run identity is the path relative to the ingest root, and the store schema
  is v2.** `sweep_a/seed0` and `sweep_b/seed0` were one row holding the second
  run's values (D4). Upsert now refuses a `run_id` arriving from a different
  `source_path`. The v1 to v2 migration is additive and runs automatically;
  legacy identities are reported through `legacy_identity_runs()` rather than
  silently recomputed, and a database written by a newer version is refused by
  name instead of being read as though the difference did not matter.
* **The store is opened read only by everything that only reads.**
  `Store.open_read_only()` uses a `mode=ro` URI, so `compare` and `report`
  cannot modify the bytes they are reporting on. WAL, a 30 second busy timeout,
  chunked id lookups and `StoreError` everywhere else.
* **The report determinism contract is stated and tested.** A report is a byte
  identical function of the database, the permutation seed and
  `SOURCE_DATE_EPOCH`. The Plotly runtime is inlined only when there are figures
  to draw, so an empty report went from 4.86 MB to under 100 KB; ragged seed
  lengths are NaN padded and aggregated with NaN aware functions rather than
  fabricating a band (D9); at most 12 conditions are plotted, chosen by
  severity then effect size then name with the baseline always kept, and the
  rest are folded into one envelope trace.
* **`triage compare` and the report call `rank()` themselves,** which is how the
  `classify()` order change left every output of this tool unchanged.
* **Parsers, at the boundary.** A directory is claimed as a run only when none
  of its subdirectories are parseable, so a stray `index.csv` at a sweep root no
  longer swallows the sweep (D5); several CSVs in one run directory concatenate
  instead of overwriting (D7); step columns keep their fractional values (D6);
  a TensorBoard run is claimed only on the `events.out.tfevents.` prefix with a
  structural header check; JSONL skips and counts records with no step key,
  which is what makes another project's environment header lines readable, and
  accepts the strings `"NaN"`, `"Infinity"` and `"-Infinity"` as their float
  values at parse so that the non finite filter can count and drop them. `null`
  means the point is absent, never zero.
* **Ingest catches per run failures with their type and traceback,** logs
  through the `triage.*` logger to stderr, and gained `--log-level`. One
  poisoned line costs one run, never the sweep.
* **Fingerprints are relative path plus content hash** for files up to 8 MiB and
  stat only above that, rather than path, size and modification time. A
  consequence worth stating because it is intended rather than accidental: a
  sweep that was MOVED skips as unchanged rather than tripping the identity
  check, since the fingerprint no longer depends on where the tree sits. There
  is a test for it.
* **`variant_key` canonicalises integral numerics,** so `batch_size: 32` and
  `batch_size: 32.0` are one condition. Booleans are deliberately not folded
  into 0 and 1, because a flag and a count are not the same axis.
* **Metric direction is inferred on word boundaries, and can be declared.**
  `val/loss_scale` is no longer read as a loss because it contains the letters;
  `mape` and `smape` are known to be lower is better; and
  `ComparisonConfig.directions`, `--higher-is-better` and `--lower-is-better`
  state a direction the name cannot imply.
* **The window block mode's internals were corrected together.** The
  autocorrelation time is estimated on the raw window rather than the smoothed
  one, a window that does not divide evenly keeps the LAST points rather than
  the first, and every block statistic comes from the same block means, so the
  interval and the p value are computed from one partition rather than two.
  Windowed statistics refuse a non finite window outright.
* **`triage compare` no longer computes the sensitivity table,** which belongs to
  `report` and was doing a sweep wide correlation nobody had asked the compare
  verb for.
* **The report is more legible and more accessible.** Muted ink at a 5.03:1
  contrast ratio, captions and a scope line on every table, `main` and `aria`
  landmarks, and dash cycling once the categorical palette runs past its eighth
  slot, so two conditions never rest on colour alone.
* **The store gained additive tables for embeddings and cached LLM responses**
  with no schema version bump, since nothing existing changed shape and an older
  reader is unaffected by tables it does not know about.
* **The version is written in exactly one place,** `triage/_version.py`, which
  `pyproject.toml` reads with setuptools' `attr:` and `triage.__version__` reads
  back out of the installed metadata. Releasing is one edit.
* **`mypy --strict` runs over the whole package with no per module exceptions
  for this project's own code,** and introducing it found four real defects
  rather than only annotations: a `None` comparison in the practical gate, the
  long form JSONL types, the `Outcomes` column handling, and an `Any` escaping
  at a boundary.
* **The development toolchain moved to nox.** `noxfile.py` is the canonical
  cross platform runner and the Makefile is a thin wrapper that forwards to it,
  because the Makefile only ever worked on Windows through Make's direct exec
  fast path and broke the moment a recipe gained a pipe. `make distclean` no
  longer asks the venv's own interpreter to delete itself and then report
  success either way.
* **CI is a matrix.** Three operating systems by two Python versions for lint,
  unit and integration, with the calibration gates on one runner because they
  are a wall clock measurement. New jobs: `package` (build the wheel, twine
  check, install it into two throwaway environments outside the tree and make
  each do its job), `core-only`, `ranges`, `reports` (the committed PDFs are the
  ones this tree builds, checked by text diff), and `docs`. `mypy --strict` over
  the package, coverage with a floor of 90 against a measured 94, and a
  `dependabot.yml`.
* **A deprecation is an error in the test suite,** with two named third party
  ignores rather than the blanket `ignore::DeprecationWarning` that used to
  silence this package's own deprecations too.
* The four large curve PNGs were re exported at scale 1 (3.2 MB to 1.2 MB), the
  dash guard was scoped to prose files, `.gitignore` gained the build and cache
  paths, and the private file list was replaced with one neutral `/private/`
  pattern.

### Fixed

Every P0 from the defect register, by name.

* **D1. Non finite values poisoned the statistics and fabricated top ranked
  findings.** No code path filtered NaN or Inf. One NaN in a candidate's final
  window made the window statistic NaN, made every `>=` comparison against the
  null False, and produced an exact p value of `0/total = 0.0` that passed the
  gate; one `+inf` made the effect, the relative effect and the severity all
  infinite, so a **diverged run was ranked first in the verdict table**.
  `MetricSeries` now filters non finite points at construction and records
  `dropped_non_finite` in metadata, JSON `NaN` and `Infinity` literals are
  parsed and then dropped like any other non finite value, and CSV reading no
  longer keeps `inf` while dropping NaN, so the three formats agree on a
  poisoned file. Config JSON is written with `allow_nan=False`.
* **D2. The false discovery rate control was not the one advertised and `--fdr`
  was inert.** See the Changed entry above: the gate read `adjusted_p < alpha`
  while the rate was passed to a procedure whose output does not depend on it,
  so `--fdr 0.001`, `--fdr 0.10` and `--fdr 0.90` produced byte identical
  verdicts.
* **D3. Stored XSS in the HTML report through metric tag names.** A tag name is
  attacker influenced text in some pipelines and reached the template through
  unescaped interpolation and generated element ids. Jinja autoescaping is now
  unconditional, div ids are slugs rather than raw names, Plotly text is
  escaped, and a renderer suite renders a hostile fixture and asserts it is
  inert.
* **D4. Run identity was the directory basename, so distinct runs silently
  overwrote each other.** Covered in Changed above: identity is now the relative
  POSIX path, and an upsert from a different source path fails that run loudly.
* **D5. A stray parseable file at the sweep root swallowed the entire sweep.**
  Discovery tested parsers against a directory before descending and stopped at
  the first match, so a top level `index.csv` made the sweep root itself the one
  discovered run and both real runs were skipped, at exit 0. Leaf claiming fixes
  it, with claim and descend decisions logged at debug level.
* **D6. Fractional epochs truncated to duplicate integers and destroyed the
  series.** Step columns included `epoch` and were hard cast to int64, so twelve
  rows of epoch 0.00 to 1.10 collapsed onto two steps.
* **D7. Multiple CSVs in one run directory overwrote instead of
  concatenating.** A run logging `train.csv` and `val.csv` kept whichever was
  read last.
* **D8. `compare_all` silently discarded refused comparisons.** They are
  `ComparisonRefusal` records now, named in every output.
* **D9. Ragged run lengths fabricated the seed band in the report.** Series of
  different lengths were truncated onto the shortest, which drew a band that
  was not the data's. They are NaN padded and aggregated NaN aware, with a note
  attached when the length spread exceeds 5 percent.
* **D10. Exact p values of 0.0 from catastrophic cancellation.** The null
  reconstructed group sums by subtraction while the observed statistic was
  computed directly, so with an absolute tie tolerance the observed arrangement
  could fail its own `>=` test: at 3 against 5 with values around 50 this
  produced an impossible `p == 0.0` in 37 of 200 trials. Both group sums are
  now computed directly, the tie tolerance is relative, and in exact mode the
  observed value **is** row zero of the null distribution, so it is inside its
  own null by construction.
* **D11. `min_attainable_p` was wrong for unequal group sizes.** `2 / C` assumes
  the sign flipped arrangement is enumerated, which is true only when the two
  groups are the same size: at 2 against 5 it reported 0.0952 where 0.0476 was
  attainable, so designs that could reach p below 0.05 were declared
  underpowered. It is `2 / C` when the counts are equal and `1 / C` otherwise.

Also fixed:

* **The reproducibility gate could pass by accident.** The test comparing two
  HTML reports byte for byte dropped every line containing "Generated" before
  comparing, but the template writes "generated" in lower case, so the
  timestamps were compared too and the test passed only while both reports
  landed in the same minute. It now masks the timestamp wherever it appears.
* **`--fdr` reached the compiled report as an en dash.** TeX sets two hyphens as
  a ligature, so the flag printed by `RegressionConfig.describe()` was typeset
  as a dash followed by `fdr` in the abstract of the PDF, which is both a banned
  character and the wrong flag. The asset generator's `escape()` now breaks the
  ligature.
* The weak mode cost table compared its two modes at different seed deviations:
  the seed replicated column was borrowed from another arm, at 0.005, 0.02 and
  0.05, under rows labelled 0.01, 0.02 and 0.04. Both columns now come from one
  arm at one set of deviations.

### For ml-experiment-triage consumers

Everything in this section can reach code that already imports this package.
The five numbered items are the issues filed against this repository; the list
after them is every other behaviour change with a blast radius outside this
tree.

#### Issue #1: cross sectional outcomes ingestion

**Supported.** A `run.jsonl` that is one row per classified field is no longer
something to hand to `JsonlParser` as a probe and record the refusal of. There
is a real model for it, `triage.core.outcomes.Outcomes`, and a real parser,
`triage.parsers.outcomes_parser.OutcomesParser`, which recognises that schema
and reads generic step free JSONL when asked explicitly with `--outcomes`.

The fixture in this repository is not a copy of your output, and an earlier
draft of this section said it was. It is generated, in your writer's real row
shape: all 25 keys in your order, `engine_describe` nested, `signals` and
`finding_codes` list valued, `prompt_version` null on every row, one engine per
file, and several selectors under one `form_id`. The version that claimed to be
verbatim held the 11 key illustration from the issue and one row per form, which
is what let the join below be documented wrongly and pass its own tests.

**Your unit of analysis is the `(form_id, selector)` pair, not `form_id`.** A
form holds between 3 and 23 classified fields, so `form_id` repeats inside one
engine's rows and `pair_on("form_id", ...)` raises `SeriesError: 'form_id' is
not unique within 'rules'`, correctly. `pair_on` takes one key, so the composite
is built as a group key first:

```python
from dataclasses import replace

from triage.parsers.outcomes_parser import OutcomesParser

# One directory holding both engines' run.jsonl files: your writer puts one
# engine in each run directory, and a paired test needs both in one Outcomes.
outcomes = OutcomesParser().parse(path)

field_id = [
    f"{form_id}|{selector}"
    for form_id, selector in zip(
        outcomes.group("form_id"), outcomes.group("selector"), strict=True
    )
]
paired = replace(outcomes, groups={**outcomes.groups, "field_id": field_id})
rules, ngram = paired.pair_on("field_id", "engine", "rules", "ngram")
```

Run against your `p6-rules` and `p6-ngram` logs that joins 2,317 rows to 2,317
and reads the pair whole: nested `engine_describe`, list valued `signals` and
`finding_codes` and the nulls all land as group keys, four measured fields.
`paired_permutation` over `correct` clustered by `template_id` then returns
`mode="paired_cluster"`, rules 0.7678 against ngram 0.8226, and p 0.00390625
exact over the 1,024 arrangements of your 10 templates, with `min_attainable_p`
0.001953125.

The refusal half is as deliberate as the acceptance: `compare_window_block` and
anything windowed refuse an `Outcomes` input by construction, with a message
naming `paired_permutation` as the right tool.

**The `probe_native_ingestion` SCENARIO is now supported. Your function is not
going to notice on its own.** It names `JsonlParser`, and `JsonlParser` still
refuses a step free file deliberately: the new path is `OutcomesParser`, a
different class, and nothing about this release changes what the one you call
does. So `test_the_harness_claims_the_run_log_and_cannot_read_it` still passes
against the installed 1.1.0 wheel, exactly as you wrote it to. Pointing the
probe at `OutcomesParser` is one import line, it is yours to make, and until it
is made your `analysis.json` will keep recording the refusal.

#### Issue #2: paired permutation with clustered resampling

**Added**, as `triage.paired_permutation`:

```python
result = paired_permutation(baseline, candidate, statistic=macro_f1, clusters=template_ids)
```

Whole clusters swap together, enumeration is exhaustive while `2**n_clusters`
fits the limit, and `min_attainable_p` is `2 / 2**n_clusters` there. Calibrated
in the published suite: 4.90 percent type I on clustered null data with the
clustering declared, against 12.10 percent with it ignored, which is the
measurement that justifies the argument existing.

Two things to know before you write the callable:

* **The mode vocabulary is open now.** `ComparisonResult.mode_label` and
  `to_dict` used to raise `KeyError` on any mode string not in a private table,
  so writing the truthful `"template_clustered_paired"` left you unable to call
  either member. An unknown mode falls back to the mode string with a generic
  claim label. You can keep your own mode string.
* **`statistic` must be a pure function of the vector it is handed.** The
  permutation loop passes label swapped vectors of your length and order; the
  cluster bootstrap behind the confidence interval passes a vector of a
  different length in a different order. A closure over a fixed truth array
  indexed positionally works for the first and fails on the second, with an
  error when the lengths differ and a silently wrong interval when they match.
  Pack whatever the statistic needs into the row: this repository's macro F1
  statistic uses `truth * n_classes + prediction` and unpacks it inside the
  callable.

#### Issue #3: an absolute practical threshold

**Added** as `RegressionConfig.practical_threshold_absolute`, with
`--practical-threshold-absolute` on the command line. Exactly one of it and
`practical_threshold_pct` may be set, `ValueError` when both or neither, and
`describe()` prints the one in force. The old `practical_threshold` constructor
keyword still works as an alias of `practical_threshold_pct` with a
`DeprecationWarning`, and is removed in 1.2.0, so rename it at your convenience
rather than at ours.

Both fields are `float | None` now, which is a typed break if you were reading
them back.

#### Issue #4: a stable join key and the ranking contract

Two halves, and both are behaviour changes:

* **`classify()` returns findings in input order.** One `Finding` per
  `ComparisonResult`, positionally, so `zip(results, classify(results))` is a
  valid join and the `id(finding.result)` workaround can be retired. If you
  wanted severity order, call the already public `rank(findings)`, which is what
  this tool's own CLI and report now do. Verified against your code before the
  change landed: your sort is unaffected by our ordering.
* **`ComparisonResult.key` exists,** spelled `"variant|tag"` and populated by
  `compare_all`. It is left caller settable for library use. `Finding.tag` was
  never unique once one metric was compared across several slices, which is what
  made the `id()` workaround necessary in the first place.

#### Issue #5: the packaging split, and PyPI

**Done.** Core dependencies are numpy and scipy and nothing else. `parsers`,
`report`, `cli`, `agentic` and `all` are extras, and the modules that need them
import them at the point of use. A CI job installs the bare core into a clean
environment and runs the statistics API in it, so the split is tested rather
than declared.

`triage/py.typed` ships with the wheel and the distribution carries the
`Typing :: Typed` classifier, so the `ignore_missing_imports` override for
`triage.*` in your mypy configuration exists for no reason now and can go.

Publication to PyPI is by Trusted Publishing with PEP 740 attestations, which
means a `git+https://` pin in your dev extra can become an ordinary version
specifier.

#### Everything else that can reach your code

* **The false discovery rate gate is operative, and default behaviour is
  unchanged.** The statistical gate reads `adjusted_p <= false_discovery_rate`.
  Through 1.0.0 it read `adjusted_p < alpha` and the rate was inert. The default
  rate moved from 0.10 to **0.05** precisely so that the decision made at
  defaults is the decision 1.0.0 made: **no verdict shifts for anyone who does
  not set the flag**. If you pass `--fdr` or set `false_discovery_rate`, it now
  does what it says. `alpha` is retained and bounds admissibility only.
* **A `compare` that made zero comparisons exits 4, where it exited 0.** If you
  invoke the CLI from a build, this is the change most likely to turn a green
  build red, and that is what it is for.
* **`analyse_database` returns an `Analysis` dataclass,** not the tuple it
  returned through 1.0.0. Unpacking by position will break; the fields are
  named.
* **`compare_all` returns a `ComparisonResults`,** a list subclass, so iteration
  and indexing are unaffected. What is new is `.refusals`, and reading it is how
  you find out about a comparison that could not be made.
* **p values move on unbalanced designs.** The permuted statistic is the Welch t
  rather than the raw difference of means. Balanced designs with equal spreads
  are bit identical; unbalanced or heteroscedastic ones change, in the direction
  that fixes them (17.92 percent type I to 12.92 on a 7 against 3 design with
  unequal spreads).
* **`SensitivityResult.p_value` is `float | None`,** and `adjusted_p`,
  `significant` and `n_variants` are new. Below five conditions no p value is
  reported at all.
* **Run identity is the path relative to the ingest root and the store schema is
  v2.** The migration is additive and automatic. A database written by a newer
  version is refused by name.
* **A deprecation from this package is now something you will hear about.** The
  test suite treats `DeprecationWarning` as an error rather than ignoring it
  wholesale, and the one deprecation in flight is the `practical_threshold`
  keyword above.

## [1.0.0] 2026-08-05

First release. The repository replaces its previous contents in full.

### Added

* One `Experiment` model behind parsers for TensorBoard event files, CSV in
  either wide or long shape, and JSONL, all producing numerically identical
  results from the same underlying series.
* SQLite storage of series as zlib compressed float32 blobs, with a bit for bit
  exact round trip and source fingerprinting so an unchanged run is never
  reparsed and an interrupted ingest loses only the run in flight.
* Two sided permutation testing on the mean of a smoothed final window, in two
  modes: seed replicated (the strong claim) and single run window block (the
  weaker claim, labelled as such in every output).
* Exhaustive enumeration of the permutation distribution where it is small
  enough, so p values are exact rather than estimated at typical seed counts,
  and reporting of the smallest p value a given design can attain.
* Two gate regression flagging, statistical and practical, with Benjamini
  Hochberg correction and severity ranking that discounts the weaker mode. (This
  release described the correction as running at a 10 percent false discovery
  rate. It did not: the gate compared the adjusted p against alpha 0.05 and the
  rate was inert. Corrected in 1.1.0, which is where the accurate statement of
  the rule now lives.)
* Spearman hyperparameter sensitivity with the run count printed beside every
  correlation.
* A statistical calibration suite as a phase gate: measured type I error inside
  [2, 8] percent at a nominal 5, measured power above 90 percent, and a check
  that the whole null distribution is uniform rather than only correct at 0.05.
* A published measurement of what the weaker mode costs: on data with realistic
  seed variance and a true effect of zero it fires on 53 to 87 percent of
  comparisons, where the strong mode holds at 4.4 to 5.0.
* Self contained HTML reporting with the seed spread drawn as a band, plus the
  main LaTeX report and the debug report.
* `triage ingest`, `triage compare` and `triage report`, with progress bars on a
  terminal and plain lines in a log.
* A deterministic 31 run synthetic demo sweep in three log formats, with a
  committed record of its verdicts that `make verify-demo` checks to four
  decimal places.
* A repository wide guard against em dashes and en dashes, covering LaTeX
  hyphen ligatures and the text extracted from compiled PDFs.

### Fixed during development

Recorded here because the calibration result depends on them; the full accounts
are in `docs/ENGINEERING_LOG.md`.

* Window block mode measured a 12 percent type I error against a nominal 5. The
  autocorrelation time was estimated on the two windows concatenated, which a
  step change at the join inflates, and the block length was one autocorrelation
  time, at which block means are still correlated. Now estimated per window, set
  at three times tau, and refused outright below eight blocks.
* HTML reports were not reproducible because Plotly assigns each figure a random
  div id. Now set explicitly from the metric name.
* The synthetic sweep seeded its generators with `hash()` of a string, which
  Python randomises per process. Now `zlib.crc32`.

### Known limitations

* Rank correlation is blind to non monotone hyperparameter effects, which is the
  normal shape for a learning rate. Documented wherever the sensitivity table
  appears and pinned by a test.
* Sensitivity reports association across a sweep, not controlled effects.
* TensorBoard support covers scalars only.
