# ML Experiment Triage

**Turns a directory of training runs into a ranked, significance tested
comparison.** Which runs beat the baseline, by how much, with what confidence,
and with the tool's own error rate measured rather than assumed.

```bash
pip install ml-experiment-triage
```

**Until the PyPI registration completes that line 404s.** Install from the tag in
the meantime, which resolves the same dependencies:

```bash
pip install "ml-experiment-triage @ git+https://github.com/Olajide-Badejo/ML-Experiment-Triage.git@v1.1.0"
```

This paragraph and its code block go away the day the package is on PyPI. The
[install page](install.md) has the extras and what each one is for.

## The problem

Two loss curves on a TensorBoard tab, one of them slightly lower at the end, and
a decision made on it. That is how most training runs get compared, and it fails
in four ways at once:

- the **final point is mostly noise**, so the comparison rests on a single draw;
- curve values are **autocorrelated**, which breaks the standard significance
  tests;
- **run to run variance** is routinely larger than the effect being claimed;
- several **metrics are compared at once**, so something was always going to
  look significant.

This project replaces that with a stated hypothesis, a test whose assumptions
the data actually satisfies, and a calibration suite that proves the p values
mean what they say.

## The result that matters

Both comparison modes were run thousands of times on synthetic data with **a
true effect of exactly zero**, so every rejection is a false positive. The only
difference between the two is whether the comparison had seed replicates to work
with.

![False positive rate against seed variance. Comparing single runs fires on most
comparisons where the true effect is zero, while the seed replicated comparison
stays on the nominal 5 percent line.](https://raw.githubusercontent.com/Olajide-Badejo/ML-Experiment-Triage/main/assets/images/weak-mode-cost-light.png)

Comparing one run against one run, on data where nothing is different, calls it
a significant difference on <!-- calibration:weakrange -->54 to 88<!-- /calibration:weakrange --> percent of comparisons, where the seed
replicated mode holds at <!-- calibration:strongrange -->3.8 to 4.1<!-- /calibration:strongrange --> percent. That is not a flaw in this
implementation; it is what the question "are these two curves different" can
answer when it has no information about how much a curve moves between seeds.

The tool still offers that mode, because sometimes one run is all there is. It
labels every result from it as a weaker claim, in the terminal, in the HTML
report and in the PDF.

## Calibration

A tool that produces p values is worth nothing unless its p values mean what
they say. The calibration suite generates about <!-- calibration:comparisons -->23,900<!-- /calibration:comparisons --> synthetic
comparisons whose true effect is set in advance, runs the real comparison over
them, and counts how often it is wrong.

<!-- calibration:headline -->
| Measurement | Result | Gate |
|---|---|---|
| Type I error, seed replicated | 4.53 percent (+/- 0.74), 3000 null cases | [2, 8] percent |
| Type I error across seed variance | 4.40, 4.70, 5.00 percent, sigma 0.005 to 0.05 | [2, 8] percent |
| Power, seed replicated | 96.25 percent, 800 cases | above 90 percent |
| Type I error, window block | 6.28 percent (+/- 1.07), 1989 cases, 11 refused | [2, 8] percent |
| Power, window block | 100 percent, 300 cases | above 90 percent |
| Type I error, paired clustered | 4.90 percent (+/- 1.34), 1000 cases, 10 clusters of 4 pairs | [2, 8] percent |
| Null p value uniformity | 0.0580, 0.0993, 0.2407, 0.4847, 1500 cases | within 3 standard errors |
<!-- /calibration:headline -->

Every number on this site is rendered from `triage/calibration.py` by
`scripts/render_calibration_docs.py`, and CI fails if a document and the
measurement disagree. The suite reproduces in <!-- calibration:statsclock -->110 s<!-- /calibration:statsclock -->, deterministically.

The designs a permutation test is *not* exact in are measured too, and published
whether or not they flatter: see [Methodology](methodology.md).

## Where to go next

| If you want to | Read |
| --- | --- |
| Install it, with or without extras | [Install](install.md) |
| Compare a directory of runs from the command line | [Quickstart](quickstart.md) |
| Call the statistics from your own code | [Library tutorial](library.md) |
| Know whether to believe the p values | [Methodology](methodology.md) |
| See the whole tool exercised on a real workload | [The autofill vertical](autofill.md) |
| Understand the local model layer | [The local LLM layer](llm.md) |
| Know how this relates to my other repositories | [Ecosystem](ecosystem.md) |
| Know what was rejected and why | [Design decisions](DESIGN_DECISIONS.md) |
| Read the exact signatures | [API reference](api.md) |
