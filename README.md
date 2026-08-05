# ML Experiment Triage

Turns a directory of training runs into a ranked, significance tested comparison:
which runs beat the baseline, by how much, with what confidence, and with the
tool's own error rate measured rather than assumed.

I built this because I kept doing the same thing badly. Two loss curves on a
TensorBoard tab, one of them slightly lower at the end, and a decision made on
it. That reads a difference off a single seed, from a statistic (the last point)
that is mostly noise, on curves that are autocorrelated enough to defeat the
usual tests, across enough metrics that something was always going to look
significant. This replaces that with a stated hypothesis, a test whose
assumptions the data actually satisfies, and a calibration suite that proves the
p values mean what they say.

## The credibility anchor

Everything else here is only worth as much as this table. The calibration suite
generates thousands of synthetic sweeps whose true effect I set myself, runs the
real comparison over them, and counts how often it is wrong.

| Measurement | Result | Gate |
|---|---|---|
| Type I error, seed replicated mode | **4.53 percent** (+/- 0.74), 3000 null cases | inside [2, 8] at a nominal 5 |
| Type I error, holding across seed variance 0.005 to 0.05 | 4.40, 4.70, 5.00 percent | inside [2, 8] |
| Power, seed replicated mode, large effect | **96.25 percent**, 800 cases | above 90 percent |
| Type I error, single run window block mode | **5.81 percent** (+/- 1.04), 1928 cases | inside [2, 8] |
| Power, single run mode, large effect | 100 percent, 291 cases | above 90 percent |
| Null p values uniform at 0.05 / 0.10 / 0.25 / 0.50 | 0.058 / 0.099 / 0.241 / 0.485 | within 3 standard errors |

Reproduce it with `make test-stats`. It takes about a minute and is deterministic.

The last row matters more than it looks. Checking only that the error rate is
right at 0.05 can be passed by a test that is wrong in two compensating
directions, so the suite checks the shape of the whole null distribution.

## The number I would want a sceptic to see

The same tool, run the way people normally compare runs, on synthetic data with
realistic seed to seed variance and a **true effect of exactly zero**:

| Seed standard deviation | Single run mode false positive rate | Seed replicated mode |
|---|---|---|
| 0.01 | 53.2 percent | 4.4 percent |
| 0.02 | 77.0 percent | 4.7 percent |
| 0.04 | 87.4 percent | 5.0 percent |

Comparing one run against one run, on data where nothing is different, calls it
a significant difference most of the time. That is not a flaw in this
implementation, it is what the question "are these two curves different" can
answer when it has no information about how much a curve moves between seeds.
The tool still offers that mode, because sometimes one run is all there is, but
it labels every result from it as a weaker claim in the terminal, the HTML and
both PDFs.

## What it does

1. **Parses** TensorBoard event files, CSV (wide or long) and JSONL into one
   `Experiment` model, stored in SQLite as compressed float32 blobs. Runs are
   parsed once; every later comparison reads the database in milliseconds.
   Re-ingesting an unchanged source does no work.
2. **Compares** any condition against a baseline with a two sided permutation
   test on the mean of a smoothed final window, in two clearly separated modes:
   seed replicated (the strong claim) and single run window block (the weaker
   claim, always labelled).
3. **Flags regressions** on two gates that must both pass: an adjusted p below
   0.05, and a relative effect of at least 2 percent. Both are configurable
   constants, not buried numbers. Multiplicity is handled by Benjamini Hochberg
   at a 10 percent false discovery rate across metrics, and findings are ranked
   by severity.
4. **Ranks hyperparameter sensitivity** by Spearman rank correlation, always
   printing the n behind each correlation.
5. **Reports** into a single self contained HTML file, plus two LaTeX PDFs.

## Quick start

```bash
make env                 # Python 3.13 venv, pinned dependencies
make demo                # synthesise 31 runs, ingest, compare, write the HTML report
make test                # the full suite, including the calibration gates
make all                 # everything, from a clean tree, including both PDFs
```

Against your own runs, one directory per run with an optional `config.json`:

```bash
triage ingest  path/to/runs --database triage.db
triage compare --database triage.db --baseline my_baseline_variant
triage report  --database triage.db --baseline my_baseline_variant --output report.html
```

## What the demo produces

31 runs across 7 conditions, in three log formats at once, generated from fixed
seeds so the verdict table is identical on any machine:

```text
candidate            metric           change     p adj  verdict
---------------------------------------------------------------
lr0.0100_bs32        val/loss        +38.67%    0.0159  regression
lr0.0003_bs32        val/loss        +20.27%    0.0159  regression
lr0.0100_bs32        val/accuracy     -5.03%    0.0159  regression
lr0.0030_bs32        val/loss        -19.52%    0.0159  improvement
lr0.0030_bs128_seed0 val/loss        -16.85%    0.0006  improvement  [weaker mode]
lr0.0030_bs64        val/loss        -12.59%    0.0317  improvement
lr0.0030_bs128_seed0 val/accuracy     +1.10%    0.0006  significant but below the practical threshold
lr0.0003_bs32        val/accuracy     -1.99%    0.0317  significant but below the practical threshold
lr0.0030_bs32        val/accuracy     +0.96%    0.0317  significant but below the practical threshold
lr0.0030_bs64        val/accuracy     +0.91%    0.0857  no significant change
lr0.0010_bs64        val/accuracy     -0.38%    0.5195  no significant change
lr0.0010_bs64        val/loss         +0.08%    0.9762  no significant change

best condition found: lr0.0030_bs32 (ground truth: lr0.0030_bs32)
```

The tool finds the condition that is genuinely best, and does not flag the
condition whose true effect is real but negligible.

Read the fifth row against the fourth. Both are improvements of roughly the same
size, but the single seed condition reports an adjusted p of 0.0006 while the
five seed condition reports 0.0159. That gap is the anticonservatism of the
weaker mode showing up in the demo itself, which is exactly why the row carries
a label.

## Design choices, and why

**A permutation test on a final window mean, not a t test on raw curves.**
Training curve values are autocorrelated and rarely normal, which breaks what a
t test needs. A permutation test needs only exchangeability under the null,
which is what the null hypothesis already asserts (Good, *Permutation,
Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed., Springer 2005). The
statistic is the mean of the last 10 percent of a smoothed curve, minimum 20
points, because the final window is what people mean by "how good did it get"
and one final point is mostly noise.

**Two gates for a regression, not one.** With enough data a meaningless 0.05
percent change becomes statistically significant. Pairing significance with a
visible practical threshold is standard in performance regression work, and it
is the difference between a tool people keep and a tool people mute.

**Benjamini Hochberg, not Bonferroni.** Training metrics are strongly correlated
with one another; Bonferroni assumes the worst case dependence and would cost
real power for no gain here (Benjamini and Hochberg, *JRSS B* 57(1), 1995). The
implementation is checked against the worked example in that paper.

**Spearman, not ANOVA.** Sweeps are unbalanced because people add runs where the
results looked interesting, and hyperparameter effects are usually monotone but
strongly nonlinear. Rank correlation handles both.

**Seed variance taken seriously throughout,** because run to run variance in
deep learning is routinely larger than the effects claimed from single runs
(Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*,
MLSys 2021). It is why the strong mode exists, why the charts draw the spread
across seeds as a band, and why the weak mode is labelled everywhere.

**SQLite, not a server.** Parse once into compressed blobs and every later
comparison is milliseconds. The demo's 372,000 points compress 2.5x and the
whole database is 2.0 MB. A database server would add operational cost and buy
nothing at this scale.

## What it will not tell you

* **A near zero rank correlation means "not monotone", never "no effect".** The
  demo sweep gives learning rate an optimum in the middle of its range, which is
  a U shape, and Spearman is blind to a U shape however strong the effect. Batch
  size, swept monotonically, shows the stronger correlation while driving the
  smaller effect. This is stated wherever the sensitivity table appears.
* **Sensitivity is association across a sweep, not a controlled effect.** A
  parameter only ever changed alongside another cannot be separated from it.
* **The single run mode cannot see seed variance,** with the consequences
  measured in the table above.
* **The single run mode refuses rather than guesses.** If the final window does
  not hold at least eight effectively independent blocks, it raises an error
  naming the remedy instead of returning a p value it cannot stand behind.
* **TensorBoard scalars only.** Histograms and images are out of scope.

## Documentation

* [`docs/methodology.md`](docs/methodology.md), the statistics written out for a
  sceptical reader
* [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md), what was chosen and
  what was rejected
* [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md), dated entries with
  symptom, root cause, fix and verification
* [`PROGRESS.md`](PROGRESS.md), the build record phase by phase
* `report/main.pdf`, the full report; `report_debug/debug_report.pdf`, the
  debugging record

## Measured wall clock

On the target machine (Intel Core i7-14700K, 32 GB, Windows 11 Pro, CPU only;
the GPU in this machine is not used at any point):

| Step | Time |
|---|---|
| `make test-stats`, the calibration suite | 59 s |
| `make test`, the full suite | pending |
| `make demo`, synthesise, ingest, compare, report | 16 s |
| `make all` from a clean tree | pending |

## Requirements

Python 3.13 (3.12 works), the pinned packages in `requirements.txt`, and a TeX
installation only if you want to rebuild the PDFs. Everything runs on CPU.

## Licence

MIT. Sole author, Olajide Badejo.
