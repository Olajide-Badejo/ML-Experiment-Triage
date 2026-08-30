# Changelog

All notable changes to this project are recorded here. Format follows Keep a
Changelog; versions follow semantic versioning.

## [Unreleased]

### Added

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
  the README, the methodology and the design decisions from
  `triage/calibration.py`, the same single source the LaTeX tables already came
  from. `--check` runs in CI, so no published number can be edited in one place
  and left in another.

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

### Changed

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

### Fixed

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
