# Library tutorial

Everything the command line does is a call into `triage`, and the public names
are re exported from the top level package so that a consumer has one list to
pin to rather than a set of deep module paths that happen to work.

```python
from triage import Store, compare_all, classify, rank, RegressionConfig

with Store("triage.db") as store:
    results = compare_all(store.load_all(), baseline="lr0.0010_bs32")

config = RegressionConfig()
for finding in rank(classify(results, config)):
    print(finding.candidate, finding.tag, finding.verdict, finding.adjusted_p)

for refusal in results.refusals:
    print("not compared:", refusal.candidate, refusal.tag, refusal.reason)
```

Fifteen lines, and the last three are the half that most tools leave out: a
comparison that could not be made is a record on `results.refusals`, not a
silent absence.

The deep import paths (`triage.analysis.comparison.compare_all` and the rest)
keep working unchanged and hand back **the same objects**, so an `isinstance`
check cannot depend on which import line you wrote.

## The public surface

`dir(triage)` is the contract:

| Name | What it is |
| --- | --- |
| `Experiment`, `MetricSeries` | the one model every parser produces |
| `Store`, `StoreError` | SQLite storage, and the one exception it raises |
| `ingest`, `IngestResult` | the ingest boundary |
| `discover_runs`, `ParseError` | run discovery and the parse failure |
| `ComparisonConfig`, `ComparisonResult`, `compare_all` | the comparison layer |
| `paired_permutation` | a paired test on any statistic, with clusters |
| `RegressionConfig`, `classify`, `rank`, `TriageReport` | flagging and ranking |
| `analyse` | hyperparameter sensitivity |
| `build_context`, `render` | the HTML report |

Most of these resolve on first access rather than on import, because
`build_context` and `render` live in a module that imports plotly and jinja2 and
a consumer using only the statistics API will not have installed them.

## Comparing runs you already have in memory

`compare_all` groups runs into conditions by their `variant_key` first, so a
sweep with five seeds per setting yields one strong comparison per setting
rather than twenty five noisy pairwise ones. It returns a `ComparisonResults`,
which is a list of `ComparisonResult` with a `refusals` attribute attached, so
existing code that treats it as a list is unaffected.

Each result carries `key`, spelled `"variant|tag"`, which is the stable join key
to rejoin results to anything else you computed. `Finding.tag` is not unique
once one metric is compared across several conditions.

## Flagging: `classify` returns input order

```python
findings = classify(results, RegressionConfig())
assert len(findings) == len(results)

joined = dict(zip(results, findings, strict=True))  # a valid join, positionally
for finding in rank(findings):  # severity order, when you want it
    ...
```

`classify()` returns one `Finding` per `ComparisonResult`, **positionally, in
input order**. Callers wanting severity order call the already public `rank()`,
which is what the CLI and the HTML report do, so no output of this tool changed
when the order did. Before 1.1.0 `classify()` ended `return rank(findings)`,
which left a caller with no way to rejoin the two lists except by object
identity.

## The two gates

A regression needs an adjusted p at or below the false discovery rate **and** a
practical effect. The practical gate is relative by default and absolute on
request, and exactly one of the two may be set:

```python
from triage import RegressionConfig

RegressionConfig(practical_threshold_pct=2.0)  # the default: 2 percent
RegressionConfig(practical_threshold_absolute=0.01)  # in the metric's own units
RegressionConfig(practical_threshold_pct=2.0, practical_threshold_absolute=0.01)  # ValueError
```

Use the absolute gate on a metric whose baseline can be zero. A relative gate
divides by the baseline, and dividing by zero downgrades a real regression to
nothing. `RegressionConfig.describe()` prints whichever one is in force.

`false_discovery_rate` is what decides significance, and `alpha` bounds
admissibility only: a design whose smallest attainable p value already exceeds
alpha is reported as `inconclusive: the design cannot reach alpha` rather than
as no change.

## `paired_permutation`: two conditions on the same units

The seed replicated mode shuffles condition labels across runs. That is the
wrong null when the two conditions were measured on the **same** units, which is
the usual shape of an offline evaluation: the same fields, the same forms, the
same examples, scored twice. `paired_permutation` swaps within a unit instead.

```python
import numpy as np
from triage import paired_permutation

rules = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
ngram = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
template = ["t1"] * 4 + ["t2"] * 4 + ["t3"] * 4

result = paired_permutation(rules, ngram, statistic=np.mean, clusters=template)
print(result.mode, result.p_value, result.min_attainable_p)
```

`clusters` is not optional detail. Fields on one form template share their
markup, their locale and their author, so treating them as independent units
counts evidence that is not there: on this project's own clustered null data the
measured type I error is 4.90 percent with the clustering declared and **12.10
percent** with it ignored. Whole clusters swap together, and the smallest
attainable p value falls out of the number of clusters: `2 / 2**n_clusters`
under exhaustive enumeration. Twelve pairs in three templates can reach 0.25 and
no lower, and the result says so rather than reporting a smaller number that was
never earned.

### The statistic must be a pure function of its argument

The `statistic` argument exists because the quantity being compared is often not
a mean of anything: a macro F1 over a split cannot be written as an average of
per item numbers. It is called as `statistic(vector) -> float`, and it must
depend on **nothing but that vector**.

This is a real constraint and getting it wrong looks like it works.
`paired_permutation` hands the callable two different kinds of vector:

- the **label swapped** vectors, which keep every row in its place, so the
  length and the order are the ones you passed in;
- the **cluster bootstrap** resamples that produce the confidence interval,
  which draw whole clusters with replacement and therefore hand back a vector of
  a **different length in a different order**.

A closure that holds a fixed truth vector beside the predictions satisfies the
first and breaks on the second: loudly if the lengths differ, and silently, with
a wrong interval, if they happen to match.

The fix is to make the row carry everything the statistic needs. This package's
own macro F1 statistic packs the truth into the row as
`truth * n_classes + prediction`, which is the packing a confusion matrix uses
internally, and unpacks it inside the callable:

```python
from triage.autofill.evaluate import encode_outcome, macro_f1_statistic

baseline = encode_outcome(true_labels, rules_predictions)
candidate = encode_outcome(true_labels, ngram_predictions)
result = paired_permutation(
    baseline, candidate, statistic=macro_f1_statistic(), clusters=template_ids
)
```

Anything that travels with the row works: pack it into an integer as above, use
a structured array, or index a lookup table by a row id carried in the vector.
What does not work is a closure over a per row array that the callable indexes
positionally.

## Cross sectional rows, with no step axis

An evaluation that writes one row per classified field is not a time series, and
handing it to a windowed comparison would be a category error. `Outcomes` is the
model for that shape, `OutcomesParser` reads it, and `pair_on` is the join a
paired test needs:

```python
from triage.parsers.outcomes_parser import OutcomesParser

outcomes = OutcomesParser().parse(path)
eager, compiled = outcomes.pair_on("config_key", "runner", "eager", "compiled")
```

That is TPT's throughput sweep, where one row is one configuration and
`config_key` names it, so the unit key is written in the file already.

**Pair on the key that identifies a unit, and check that it does.** `pair_on`
refuses a key that repeats inside one condition rather than guessing which row
goes with which, and the commonest mistake is to reach for the identifier that
reads like the unit rather than the one that is it. `Autofill_audit`'s rows are
one per classified FIELD and a form holds many fields, so their unit is the
`(form_id, selector)` pair and `pair_on("form_id", ...)` raises. `pair_on` takes
one key, so a composite unit is built as a group key first:

```python
from dataclasses import replace

forms = outcomes.group("form_id")
selectors = outcomes.group("selector")
field_id = [f"{form}|{selector}" for form, selector in zip(forms, selectors, strict=True)]
paired = replace(outcomes, groups={**outcomes.groups, "field_id": field_id})
rules, ngram = paired.pair_on("field_id", "engine", "rules", "ngram")
```

Both engines have to be in the one `Outcomes` for this, so both engines' files
have to be in the one directory: a parser reads the files beside each other, and
a writer that puts each engine in its own run directory has written half a pair
in each.

The refusal half matters as much as the acceptance: `compare_window_block` and
anything windowed refuse an `Outcomes` input by construction, with a message
naming `paired_permutation` as the right tool.

## Sensitivity

```python
from triage import analyse

report = analyse(experiments)
for result in report.results:
    print(result.parameter, result.tag, result.rho, result.adjusted_p, result.n_variants)
```

The unit is the condition, not the run: seed replicates are averaged onto their
variant before the rank correlation, and below five conditions no p value is
reported at all. `p_value` and `adjusted_p` are therefore `float | None`.

Rank correlation is blind to a non monotone effect, which is the normal shape
for a learning rate. That is documented wherever the table appears and pinned by
a test, so it does not get quietly "fixed".

## Exceptions

| Raised | When |
| --- | --- |
| `StoreError` | anything the store refuses: a newer schema, a run id that moved source, a missing id |
| `ParseError` | a source that claimed a parser and then could not be read |
| `ComparisonError` | a comparison whose preconditions do not hold, naming the remedy |
| `ValueError` | a configuration that cannot mean anything, such as both practical thresholds |

## Next

- [API reference](api.md), rendered from the docstrings themselves.
- [Methodology](methodology.md), for what is behind these calls.
