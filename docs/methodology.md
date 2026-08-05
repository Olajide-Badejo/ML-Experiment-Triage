# Methodology

Written for a reader who does not believe me. Every claim below is either
derivable from what is here or measured by a test that ships with the code.

## 1. The problem with the obvious approach

The obvious comparison is: plot two curves, look at the end, pick the lower one.
Four things are wrong with it, and they compound.

**The final point is mostly noise.** A single evaluation of a converged model
varies by roughly the measurement noise of the metric. Deciding on it is
deciding on one draw from a distribution.

**Curve values are autocorrelated.** Consecutive evaluations of the same run are
correlated: the model at step 5000 is nearly the model at step 4999. Any test
that treats a window of curve points as independent samples will compute a
standard error that is far too small, and will report significance that is not
there. This is the failure mode that makes a naive t test on curve points not
merely imprecise but actively misleading.

**Run to run variance is large.** Two identical configurations trained with
different seeds land at different places, and in deep learning that spread is
routinely larger than the effect sizes people publish from single runs
(Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*,
MLSys 2021). A comparison that has only one run per side cannot separate the
effect of the change from the effect of the seed. Not "does so imprecisely":
cannot, in principle, because the data contains no information about it.

**Several metrics are compared at once.** Testing eight metrics at alpha 0.05
gives roughly a one in three chance of at least one false positive when nothing
has changed at all.

## 2. The statistic

For a tagged series of values `v[0..n-1]`:

1. Smooth with a centred moving average of width 9, edge padded so the length is
   preserved. The filter is symmetric, so it lowers the variance of the window
   mean without moving its expectation.
2. Take the final window: the last `k = max(20, ceil(0.1 * n))` points, clipped
   to `n`.
3. The statistic is the mean of that window.

Both the fraction and the minimum are configurable, and the values used are
printed in every report footer.

Why a window rather than the last point: the window mean has roughly
`sqrt(k / tau)` times less noise than a single point, where `tau` is the
autocorrelation time. Why a window rather than the whole curve: the early part
of a run says how fast it got going, not how good it got, and mixing the two
into one number answers neither question.

## 3. Mode one: seed replicated. The strong claim

**Data required.** At least two runs per condition, differing only in seed. Three
or more is recommended and the tool says so when it has fewer.

**Unit of analysis.** The run. Each run contributes exactly one number, its
final window statistic.

**Null hypothesis.** The condition label is exchangeable across runs. That is,
the final window statistics of the two conditions are draws from the same
distribution.

**Test.** Pool the `m + n` run level statistics. For every way of splitting them
into groups of size `m` and `n`, compute the difference of group means. The p
value is the fraction of those differences at least as extreme in absolute value
as the observed one.

**Exact where possible.** With five seeds a side there are C(10,5) = 252
arrangements, which is enumerated exhaustively, so the p value is exact rather
than estimated. Above 50,000 arrangements the tool samples instead, and reports
which it did.

**Why this is the strong claim.** The seed to seed variance is inside the null
distribution rather than assumed away. If two conditions differ by less than the
runs of one condition differ among themselves, the permutation distribution will
be wide and the p value will be large, which is the correct answer.

**What it assumes.** Only that runs within a condition are exchangeable
replicates. No distributional form, no equal variances, no independence between
points within a run, because points within a run are never treated as separate
observations.

**The design limit, reported rather than hidden.** With three seeds a side there
are twenty distinct label assignments, so the smallest attainable two sided p
value is 0.1. No result from that design can ever clear alpha 0.05. The tool
computes this bound, warns when it exceeds alpha, and classifies such findings as
`inconclusive: the design cannot reach alpha` rather than as "no change". A
verdict of "no significant difference" from a design that could not have found
one is not a finding, and calling it one is the most common way this kind of
analysis misleads.

## 4. Mode two: single run window block. The weaker claim

**Data required.** One run per side. This mode exists because sometimes one run
is all there is.

**Null hypothesis.** The two final windows are blockwise exchangeable: both are
segments of the same stationary process.

**Test.** Split each final window into contiguous blocks, take the block means,
and permute the block labels between the two runs.

**Block length.** Three times the estimated integrated autocorrelation time,
estimated on each window separately and the larger taken. Both halves of that
sentence were forced by measurement, and section 7 records what happened when
they were not.

**Precondition, enforced.** The window must hold at least eight such blocks. If
it does not, the tool raises an error naming the remedy rather than shortening
the block to fit. A shortened block leaves the block means correlated, and a
permutation over correlated blocks produces a null distribution narrower than
the truth, which is exactly how a test becomes anticonservative.

**What this mode can and cannot claim.** On its own terms it is calibrated: the
measured type I error is 5.81 percent at a nominal 5, over 1928 null cases. But
its null is "these two windows come from the same process", and that is not the
question anyone is really asking. The question is "would another seed of this
configuration also be better", and one run per side contains no information
about it.

The cost is measured rather than argued. On synthetic runs with realistic seed
variance and a true effect of exactly zero:

| Seed standard deviation | Window block false positive rate |
|---|---|
| 0.01 | 53.2 percent |
| 0.02 | 77.0 percent |
| 0.04 | 87.4 percent |

The seed replicated mode holds at 4.4 to 5.0 percent on the same data. This is
why every output from the weak mode carries a label.

**Mode selection is never a choice.** Two or more runs on both sides selects the
strong mode. Anything less falls back. The mode is decided by what data exists,
never by which produces the smaller p value.

## 5. Effect sizes and intervals

Every result carries the raw difference, the relative difference as a percent, a
standardised effect size, and a 95 percent interval on the absolute difference.
The interval method is named in the output because the two modes use different
ones:

* **Seed replicated:** a Welch interval on the seed level statistics. This does
  lean on a normal approximation, but at the seed level, where each value is
  already a mean over hundreds of points, which is where that approximation is
  at its most defensible. The p value beside it assumes nothing of the kind.
* **Single run:** a percentile bootstrap resampling whole blocks.

The p value decides; the interval gives scale. Neither is reported without the
method that produced it.

## 6. Two gates and the multiplicity correction

A finding is a regression only if **both** hold:

1. **Statistical.** The Benjamini Hochberg adjusted p is below 0.05.
2. **Practical.** The relative effect is at least 2 percent, in the harmful
   direction for that metric.

The direction is inferred from the tag name (a `loss` falls to improve, an
`accuracy` rises) and can be overridden.

The correction controls the false discovery rate at 10 percent across the
metrics compared against one baseline, which is the family a reader looks at
together. Bonferroni was rejected because training metrics are strongly
correlated and Bonferroni assumes the worst case dependence: on the fifteen p
values in the original Benjamini and Hochberg paper it rejects three where the
step up procedure rejects four, and that lost power is real. The implementation
is tested against exactly that published example.

Findings are ranked by severity: the size of the relative harm, scaled by
confidence, and halved for the weaker mode so that a weaker claim never outranks
a stronger claim of the same size.

## 7. What calibration testing changed

The first full run of the calibration suite measured the window block mode at a
**type I error of 12 percent against a nominal 5**. Two root causes:

1. **The autocorrelation time was estimated on the two windows concatenated.**
   Joining two runs end to end puts a step change at the join, which the
   estimator reads as long range dependence, inflating tau by roughly a factor
   of two. Estimating per window and taking the larger fixed it.
2. **The block was one tau, and one tau is not enough.** At one autocorrelation
   time, neighbouring block means are still visibly correlated. A permutation
   that scatters neighbours across both groups then narrows the null. Three tau
   put the measured error back on nominal.

That second fix created the third decision. Once a block is three tau, a short
run may not contain enough of them, and the obvious remedy of shortening the
block is precisely the failure above. So the mode refuses instead.

None of this was visible by reading the code. All of it was visible in about a
minute of measurement, which is the argument for having the calibration suite at
all.

## 8. Sensitivity, and what it cannot see

Spearman rank correlation between each numeric hyperparameter and the final
window statistic, one correlation per parameter per metric, with `n` always
printed beside it.

**Null hypothesis.** The parameter and the metric are independent.

**Rank correlation rather than ANOVA** because sweeps are unbalanced (people add
runs where the results looked interesting) and hyperparameter effects are
usually monotone but strongly nonlinear.

**The limitation, stated plainly.** Rank correlation sees monotone relationships
and nothing else. A hyperparameter with an optimum in the middle of its swept
range, which is the normal shape for a learning rate, gives a U shaped
relationship, and the rank correlation of a U shape is near zero however large
the effect. The demo sweep in this repository has exactly that shape: learning
rate drives the largest effects in the sweep and shows the weaker correlation,
while batch size, swept monotonically, shows the stronger one. A near zero
correlation here means "not monotone", never "no effect".

**Sensitivity is association, not causation.** These are correlations across a
sweep as it was actually run. A parameter only ever varied alongside another
cannot be separated from it, and the report says so under the table.

## 9. Reproducibility

The permutation seed is fixed, printed in every report footer, and settable from
the command line. Reports are pure functions of the database and that seed.

This is tested rather than asserted. `test_report_is_byte_identical_when_rerun`
compares two renderings line by line, and it caught a real defect: Plotly assigns
each figure a random div id, which made every report differ from the last. The
demo also ships a committed record of its verdicts, and
`make verify-demo` rebuilds the sweep from its seeds and checks every number
against it to four decimal places.

## References

* Y. Benjamini and Y. Hochberg. Controlling the false discovery rate: a
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society, Series B*, 57(1):289-300, 1995.
* X. Bouthillier et al. Accounting for variance in machine learning benchmarks.
  *Proceedings of Machine Learning and Systems (MLSys)*, 2021.
* P. Good. *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd
  edition. Springer, 2005.
* B. Phipson and G. K. Smyth. Permutation p values should never be zero.
  *Statistical Applications in Genetics and Molecular Biology*, 9(1), 2010.
