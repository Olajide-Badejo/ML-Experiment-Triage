# Design decisions

What was chosen, what was rejected, and what it would take to change my mind.

## Statistics

### Permutation test, not a t test

**Chosen** because it needs only exchangeability under the null, which is what
the null hypothesis already asserts. **Rejected:** a t test on curve points,
which needs independence that autocorrelated curves do not have, and normality
that training metrics rarely have. The t test does not merely lose precision
here; it computes a standard error that is far too small and reports
significance that is not present.

**What would change my mind:** nothing at this scale. The permutation test is
exact for the sample sizes involved and costs milliseconds.

### The run is the unit of analysis, not the curve point

The single most consequential decision in the project. Treating curve points as
observations inflates n by a factor of hundreds and produces confident nonsense.
Treating runs as observations gives an n of five, and a five point sample that is
honest beats a five hundred point sample that is not.

### Final window mean, not the last point or the whole curve

The last point is one draw from a distribution. The whole curve mixes "how fast
did it get going" with "how good did it get" into a number that answers neither.
The last 10 percent, minimum 20 points, smoothed, answers the second question.

**Rejected:** best value ever reached, which is a maximum over a noisy series and
therefore biased upward by an amount that grows with run length, so it silently
favours longer runs.

### Two modes, and the weaker one is always labelled

**Rejected:** offering only the strong mode. Sometimes one run is all there is,
and refusing to answer sends people back to eyeballing curves, which is worse
than a labelled weak answer.

**Rejected:** offering both without distinguishing them, which would be the worst
of all: the weak mode fires on <!-- calibration:weakrange -->54 to 88<!-- /calibration:weakrange --> percent of null comparisons when seed
variance is present, so an unlabelled weak result is close to a random number
generator with a p value attached.

**Rejected:** choosing the mode by which gives a smaller p value. That is p
hacking with extra steps. Mode is decided by what data exists.

### Refusing when the block precondition fails

The alternative was shortening the block to fit, which is measurably the failure
mode that caused a 12 percent type I error. A p value that is quietly
miscalibrated is worse than no p value, because it spends credibility that
nothing later can recover.

### Benjamini Hochberg at a 5 percent false discovery rate, not Bonferroni

Training metrics are strongly correlated. Bonferroni assumes worst case
dependence and buys nothing here: on the fifteen p values in the original paper
it rejects three where the step up procedure rejects four. **What would change my
mind:** a use case where a single false discovery is genuinely catastrophic, at
which point Bonferroni's conservatism is the point.

The rate was described as 10 percent through 1.0.0 and was not the number
deciding anything: the gate compared the adjusted p against alpha, which was
0.05, and the rate was inert. 1.1.0 made the rate operative and set its default
to 0.05, so the decision made at defaults is the decision that was always being
made and the sentence describing it is now true.

### Spearman, not ANOVA or a regression

Sweeps are unbalanced because people add runs where the results looked
interesting, and ANOVA wants a designed grid. Effects are usually monotone but
strongly nonlinear.

**Known cost, accepted and documented:** rank correlation is blind to a
non monotone effect, which is the normal shape for a learning rate. The demo
sweep demonstrates the blindness rather than hiding it.

**Future work:** a nonlinear dependence measure such as distance correlation
alongside rho, which would see the U shape. Not done because one number people
understand beats two they do not.

### Welch interval for the strong mode

Slightly inconsistent with the assumption free p value beside it, and named in
the output for exactly that reason. A percentile bootstrap over five seeds is
known to be poor, and the normal approximation is at its most defensible at the
seed level, where each value is already a mean over hundreds of points. The p
value is what decides anything; the interval is there for scale.

## Engineering

### SQLite, not PostgreSQL and not files

Parse once into compressed float32 blobs; every later comparison is
milliseconds. The demo's 372,000 points compress 2.5x into a 2.0 MB database. A
server adds operational cost and buys nothing at megabyte scale. Plain files were
rejected because the skip logic, the tag index and the atomic replace on
reparse all want a transaction.

### float32 in memory, not just on disk

Holding float64 in memory while writing float32 would make a store round trip
lossy in a way no test could honestly call exact. Fixing the model at the storage
precision turns "the round trip preserves the series" into a bit for bit
assertion.

### Fingerprint on path, size and modification time, not content

Content hashing a directory of event files costs about as much as parsing it,
which would defeat the purpose of the check.

### One directory per run

Run identity comes from the directory name and configuration from an adjacent
`config.json`. A bare file is accepted for one off use, but the directory layout
is what the demo and tests use because it is what every training framework
already produces.

### Replace rather than merge on reparse

A source whose fingerprint changed may have gained tags, lost tags, or been
rewritten entirely. Merging would leave stale series behind with no way to notice.

### MLflow is read from its files, with no MLflow dependency

**Chosen** because both of MLflow's on disk formats are made of things the
standard library already opens, and both have stopped moving: the SQLite
backend is the supported one and its `runs`, `metrics`, `params` and `tags`
tables are unchanged in the fields used here from MLflow 1.x through 3.x, while
the `mlruns/` file store went into maintenance mode when MLflow made SQLite the
default in 3.7 (December 2025). The `meta.yaml` fields that matter are flat
scalars, so they are extracted with a regular expression rather than by taking a
PyYAML dependency for them.

**Rejected:** importing MLflow to read MLflow. It would put a large dependency,
its SQLAlchemy stack and its version policy inside a tool whose core install is
numpy and scipy, in exchange for reading two file formats that are frozen.

**Rejected:** Weights and Biases, whose local format is not a stable target, and
a second TensorBoard reader, since `EventAccumulator` is already the supported
path and `tbparse` merely wraps it.

**What would change my mind:** MLflow changing its tracking schema, which the
fixtures for both layouts would catch as a failing parse rather than as a wrong
number.

### One source can hold more than one run

An MLflow tracking database is a single file holding every run of every
experiment, which is the first source here that is not one directory per run.
Reading it into one `Experiment` would average conditions together, and giving
its runs one path would give them one identity, which is the D4 collision by
another route. So parsers expand a claimed path through `runs_in`, each run gets
a path of its own (`<database>/<run_uuid>`), and identity, fingerprint and parse
all key off that path exactly as they do for a directory.

**Known cost, accepted:** the fingerprint of such a run is the fingerprint of
the whole file, so a database that changed reparses every run in it. The
alternative, a per run summary out of the shared tables, would miss a value
rewritten in place, and being conservative here costs time rather than data.

### Parsers accept both wide and long table shapes

Detected from the header rather than configured. Both shapes occur in the wild,
getting it wrong is loud, and asking the user to declare it is one more thing to
get wrong.

### Malformed records are counted, not fatal and not silent

A truncated final line is what a killed training job leaves behind. Losing the
other ten thousand records over it would be wrong; hiding it would also be wrong.
The count goes into the experiment metadata.

## Reporting

### Self contained HTML, with the Plotly runtime inlined

The demo's file is about 6 MB and survives being emailed, copied to a bucket, or
opened on a laptop with no network. Rejected: a CDN link, which turns a report
into something that expires.

The runtime is inlined only when there are figures to draw. A report with no
plottable metric used to carry the whole 4.86 MB bundle for nothing; it is now
under 100 KB, which matters because that is the shape of a report from a sweep
that went wrong.

### The chart draws the spread across seeds as a band

Plotting one line per condition hides the single thing this tool exists to
account for. With the band drawn, a reader can see at a glance whether two
conditions are separated by more than their own run to run noise, which is the
judgement the p value is formalising.

### The report commits to a single light theme

Rejected: following the viewer's theme. This report is evidence that gets
screenshotted into other documents, and it should look identical everywhere. The
palette is the validated reference categorical set, used unchanged and in its
documented order. Identity never rests on colour alone: every series is in the
legend and every number in the charts is also in the table below them.

### Explicit figure ids

Not cosmetic. Plotly's generated UUIDs made every report differ from the last,
which made the reproducibility claim unverifiable in practice.

### Every p value carries its test and mode

A bare p value is an invitation to misread it. The label costs a line of code and
removes the single most common way this kind of output misleads.

## Process

### The calibration suite is a phase gate, not a nice to have

It found a 12 percent type I error in about a minute of measurement, on code that
looked correct and had passed every mechanics test. Nothing else in the project
could have found that.

### The dash guard scopes itself to what git would ship

Path exclusion lists rot. Asking `git ls-files` does not.

### The demo ships a committed record of its verdicts

`make verify-demo` rebuilds from fixed seeds and checks every number to four
decimal places. It turns "reproducible" from a claim into a failing build.

## Repository practice

Six practices were adopted for 1.1.0 and five were rejected. Both halves are
here, because a list of what a project does is only half an argument: the
omissions are the other half, and an omission that is not written down reads as
an omission that was not noticed.

### Adopted

**A documentation site, built strict.** mkdocs-material on GitHub Pages, with
mkdocstrings rendering the docstrings that already exist into an API reference
rather than a second copy of them that can drift. `mkdocs build --strict` turns
a warning into a failure, which is what makes a dead link a build break rather
than a thing a reader finds. This was the most visible gap in the project: a
statistics tool whose method is its product had its method in a Markdown file
that only a repository visitor would ever open.

**Hypothesis property tests on the statistical core.** The gates here were
already phrased as properties, so they are encoded as ones: a permutation p
value invariant to the order the runs arrived in, `paired_permutation` invariant
to the order of the pairs, Benjamini Hochberg monotone in its input vector and
matching the published 1995 example, a store round trip exact for any finite
float32 array. Property tests on a numerical library signal that the author
knows what could be wrong with it, which no coverage percentage does.

**PyPI Trusted Publishing, with attestations.** The release job authenticates by
OIDC through `pypa/gh-action-pypi-publish` and there is no API token anywhere in
the repository or its secrets. PEP 740 Sigstore attestations are produced by
default with no extra configuration, so the whole supply chain story is a
workflow file with `id-token: write` in it. **Rejected:** hand rolling SLSA
provenance on top, which would be a large amount of YAML restating what the
attestation already says.

**uv and a committed `pylock.toml`.** `pip install ml-experiment-triage` stays
the user path and resolves the published ranges; development and CI install the
PEP 751 lock, which carries the exact resolution every published number was
measured under. A lock in a standard format with hashes beats a pinned
`requirements.txt` that only pip could read, and moving a version becomes an
explicit `nox -s lock -- --upgrade` that owes the calibration suite a rerun.

**pre-commit.** ruff format, ruff check, the dash guard, `check-yaml` and
`end-of-file-fixer`. It makes CI green by default rather than after a round
trip. Optional to install: a contributor without it gets the same answer from CI
a few minutes later, which is the property that keeps it from being a barrier.

**CITATION.cff and a Zenodo DOI.** GitHub renders the citation widget from the
CFF, and the Zenodo GitHub integration mints a DOI per release from that same
metadata. **Rejected:** also adding a `.zenodo.json`, which would silently
override the CFF and leave two files that have to agree.

### Rejected, with reasons

**Docker or a devcontainer.** This is a pure Python package whose install story
is one `pip install` line and whose only native dependencies are numpy and
scipy wheels. An image would add a build to maintain, a base to patch, and a
second set of version numbers, in exchange for solving a problem the package
does not have. **What would change my mind:** the Chrome dependency of the
agentic demo, which is the one part of this repository where "it works on my
machine" is a real risk. If that demo grows past a documented optional extra, it
gets a container and nothing else does.

**OpenSSF Scorecard.** Several of its checks are structurally unachievable for a
single author project: code review by a second person, a documented security
response process with more than one responder, a branch protection policy that
somebody other than the author can enforce. A badge reporting a middling score
would advertise the bus factor rather than the engineering, and gaming the
checks that are achievable would be worse. **What would change my mind:** a
second maintainer, at which point most of the unachievable checks become
achievable and the score starts measuring something real.

**An SBOM, or PEP 770 metadata.** Both exist to describe binary dependencies
that a package vendors or links against, which is the case a dependency list
cannot cover. This wheel is pure Python and vendors nothing: its SBOM would be
its `pyproject.toml` in a longer format. **What would change my mind:** vendoring
any compiled artifact, at which point the wheel starts containing things its
metadata does not describe.

**Mutation testing.** The worst maintenance to signal ratio available on
numerical code. A mutation that changes a constant in a permutation loop
produces a p value that is still a plausible number, so a surviving mutant is
usually a question about tolerance rather than a missing test, and the runs are
long. Hypothesis covers the same intent by stating the property directly.
**What would change my mind:** a body of pure combinatorial code, such as the
Benjamini Hochberg step up procedure taken alone, where mutants are decidable.

**A conda-forge feedstock.** A second release channel, with its own review
process and its own update lag, and nobody has asked for it. PyPI is where the
one downstream consumer installs from. **What would change my mind:** a user who
needs it, which is a cheap thing to wait for.
