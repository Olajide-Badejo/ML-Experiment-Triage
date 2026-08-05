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
of all: the weak mode fires on 53 to 87 percent of null comparisons when seed
variance is present, so an unlabelled weak result is close to a random number
generator with a p value attached.

**Rejected:** choosing the mode by which gives a smaller p value. That is p
hacking with extra steps. Mode is decided by what data exists.

### Refusing when the block precondition fails

The alternative was shortening the block to fit, which is measurably the failure
mode that caused a 12 percent type I error. A p value that is quietly
miscalibrated is worse than no p value, because it spends credibility that
nothing later can recover.

### Benjamini Hochberg at 10 percent, not Bonferroni

Training metrics are strongly correlated. Bonferroni assumes worst case
dependence and buys nothing here: on the fifteen p values in the original paper
it rejects three where the step up procedure rejects four. **What would change my
mind:** a use case where a single false discovery is genuinely catastrophic, at
which point Bonferroni's conservatism is the point.

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

The file is about 5.9 MB and survives being emailed, copied to a bucket, or
opened on a laptop with no network. Rejected: a CDN link, which turns a report
into something that expires.

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
