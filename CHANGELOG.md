# Changelog

All notable changes to this project are recorded here. Format follows Keep a
Changelog; versions follow semantic versioning.

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
  Hochberg correction at a 10 percent false discovery rate and severity ranking
  that discounts the weaker mode.
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
