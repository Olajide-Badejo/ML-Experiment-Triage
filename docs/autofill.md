# The autofill vertical

> Skeleton. Part 13 owns the finished prose, the taxonomy provenance section in
> full, and `docs/ecosystem.md` beside it. What is here is the method and every
> number this repository has actually measured, so that the prose is written
> against measurements rather than the other way round.

Browser form field classification is this repository's reference workload: the
machine learning problem behind autofill, which is deciding that an input is a
given name, a postal code or a card security code, across locales whose address
formats disagree. It is here because a tool that ranks training runs needs a
training run of its own to rank, and this problem exercises every axis the tool
advertises at once: a dataset with a definition, a sweep with seeds, two
conditions to compare, calibrated probabilities to check, and a cross sectional
evaluation output that is not a time series.

It is **not** a competitor to `Autofill_audit`, which is the production grade
auditor (a 392 rule engine over nine locale vocabularies, a calibrated ONNX
model, a Playwright driver), and it is not a competitor to the PyTorch
Performance Toolkit's transformer field classifier, which is the production
grade trainer. It is pure numpy, it trains in under a second, and its job is to
be a workload this repository can be honest about end to end.

## The taxonomy

`FieldType` is a `StrEnum` whose VALUES are WHATWG autocomplete tokens, taken
verbatim for every token included: eighteen tokens plus `unknown`. The value
space is `autofill_audit.taxonomy.Label`'s, so a label written here means the
same thing there, and `from_tpt_label()` maps the PyTorch Performance Toolkit's
seventeen snake_case labels onto it. Three projects that disagree about how to
spell a postal code cannot exchange a single evaluation file; adopting the
standard's spelling rather than inventing a fourth makes the strings
interoperate by construction (E7b).

`unknown` is a real class rather than a null: a form holds newsletter
checkboxes, comment boxes and honeypots, and a head that cannot say "none of
these" has to spend probability mass on a wrong answer instead.

## The generator, and the knobs the numbers were measured at

Deterministic to the byte. Seeding is `numpy.random.SeedSequence` with the form
index as the `spawn_key`, never `hash()`, so form 40 is the same form in a
corpus of 50 forms and in one of 5,000. The German lexicon is written in the
ASCII transliteration the specification uses (`Strasse`, `Pruefziffer`), which
makes every corpus file pure ASCII and its bytes independent of any filesystem
or console encoding.

Three templates (checkout, registration, address book) crossed with two locales
(en_US, de_DE) and four markup styles (`billing_postal_code`,
`billingPostalCode`, `billing-postal-code`, `postal_code`) give up to 24
template identities, which are the clusters the paired test resamples.

The noise knobs are the difficulty dial, and these are the values every number
below was produced at:

| Knob | Value | What it does |
| --- | ---: | --- |
| `label_dropout` | 0.35 | the field has no visible label at all |
| `abbreviation_prob` | 0.35 | the label is the short form (`PLZ`, `CVC`, `Str.`) |
| `generic_attribute_prob` | 0.45 | `name` and `id` collapse to `input_7` |
| `placeholder_dropout` | 0.50 | no placeholder text |
| `p_autocomplete` | 0.28 | an `autocomplete` attribute is present |
| `p_autocomplete_correct` | 0.72 | and it is right when it is present |
| `distractor_prob` | 0.08 | an adversarial field of no known type follows a real one |
| `split_weights` | 0.70 / 0.15 / 0.15 | train, val, test, drawn per FORM |

**How they were chosen.** They were swept against acceptance criterion 2: a
de_DE margin that is real and not trivial, over a baseline that is still a
serious one. At the first values tried (dropout 0.18, generic 0.22, placeholders
0.35, autocomplete present 0.35 and correct 0.80) the model reached 0.994 macro
F1 on de_DE against the heuristic's 0.754: a real margin, but over a corpus so
clean that the model sat at its ceiling and the sweep had nothing left to rank.
At the values above the model reaches about 0.96 and the heuristic about 0.72,
both visibly imperfect, and the learning rate still moves the result. Turning
them further up (dropout 0.50) takes the heuristic to 0.60 and starts measuring
how much signal was removed rather than what a classifier is worth.

Every corpus writes these values into `meta.json` beside it, because a
difficulty setting nobody wrote down makes every number downstream
unreproducible.

## Features and model

Hashed character n grams, n in 3 to 5, over six namespaced signals (`label:`,
`name:`, `ph:`, `ac:`, `type:`, `ctx:`), folded into 2^16 slots with
`zlib.crc32`. Never `hash()`: Python randomises string hashing per process, so a
model trained in one process would score differently in the next and nothing
would say so. An absent signal emits an explicit `<absent>` marker, because a
missing label is evidence rather than the absence of evidence, and a constant
`bias:1` token gives the model its intercept without a dense column.

The model is multinomial logistic regression in pure numpy: softmax cross
entropy, minibatch SGD with momentum, L2, seeded reshuffling per epoch. L2 and
momentum are applied to the rows a batch TOUCHES rather than to the whole
65,536 by 19 matrix, which is the standard sparse linear update and is what
makes the sweep fit in the time budget; the consequence, that a weight's
momentum decays in its own step count, is written down in the module rather
than left to be found in the numbers.

## The measured results

Corpus: 4,000 fields at seed 0, both locales. Splits: 2,790 train, 598
validation, 612 test. Six HTML pages with `data-truth` on every input.

### The sweep

The grid is `lr {0.03, 0.1, 0.3}` crossed with `l2 {0, 1e-4}` over seeds 0 to 4:
thirty runs.

| Measurement | Value |
| --- | ---: |
| Wall clock, thirty runs, CPU only | **22.6 s** (23.4 s for the whole command) |
| Budget (5.2) | under 120 s |
| Best condition | `lr0.3_l20`, seed 0 |
| Its validation macro F1 | 0.9775 |

The corpus is read and hashed once and the thirty runs share it; the sweep keeps
only the best head (`best.npz`), because thirty weight files are tens of
megabytes of scaffolding the comparison layer never opens.

### Model against heuristic, on the validation split

The heuristic reads the `autocomplete` attribute when there is one and otherwise
matches a keyword table holding every canonical term the specification names, in
both locales, longest phrase first. It loses on the synonym tail: `Rufnummer`,
`Wohnort`, `Sicherheitscode`, `Anschrift`.

| Split | Engine | Accuracy | Macro F1 | ECE |
| --- | --- | ---: | ---: | ---: |
| all | ngram (model) | 0.9699 | 0.9775 | 0.0047 |
| all | rules (heuristic) | 0.7559 | 0.7692 | 0.1622 |
| en_US | ngram | 0.9839 | 0.9877 | 0.0028 |
| en_US | rules | 0.8071 | 0.8085 | 0.1836 |
| **de_DE** | **ngram** | **0.9547** | **0.9643** | 0.0035 |
| **de_DE** | **rules** | **0.7003** | **0.7235** | 0.1634 |

The de_DE macro F1 margin is **0.2408 absolute**, and the heuristic is a
baseline that still gets seven fields in ten right.

Per field latency, measured as the mean over the split: 4.0 us for the model,
3.7 us for the rule engine.

### The verdict, from this tool's own comparison

Five bootstrap replicates per engine, replicate `k` as seed `k`, each written as
an ordinary run directory and read by `triage ingest` with no autofill specific
flag anywhere in the ingest or comparison layers:

```text
triage autofill evaluate --data DATA --weights runs/best.npz --locale de_DE --bootstrap 5 --out eval
triage ingest eval/runs --database eval.db
triage compare --database eval.db --baseline rules_de_DE_val --tags eval/macro_f1
```

| Candidate | Metric | Change | p | p adjusted | Verdict | Mode |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `ngram_de_DE_val` | `eval/macro_f1` | +37.54% | 0.0079 | 0.0079 | improvement | seed replicated |

0.0079 is `2 / 252`, the smallest p value five against five seed replicates can
attain, so the separation is as complete as this design can report.

### The same claim, from the consumer's schema

`evaluate.py` also writes a per field `run.jsonl` in `Autofill_audit`'s E5
schema verbatim, which is this repository's own fixture for `OutcomesParser` and
`paired_permutation`:

| Quantity | Value |
| --- | ---: |
| Rows | 574 (287 fields, two engines) |
| Pairs after `pair_on("form_id", "engine", "rules", "ngram")` | 287 |
| Clusters (`template_id`) | 11 |
| Macro F1, rules | 0.7235 |
| Macro F1, ngram | 0.9643 |
| Mode | `paired_cluster` |
| p | 0.00098 |

The test warns, correctly, that the p value rests on 11 independent units rather
than 287. Macro F1 is not the mean of anything, which is why
`paired_permutation` takes a callable; each row is packed as
`truth * n_classes + prediction` so that the statistic stays a pure function of
its argument under the clustered bootstrap as well as under the label swap.

### Probability calibration

Post hoc temperature scaling, one scalar fitted on the validation split, with
expected and maximum calibration error over 15 equal MASS bins (equal width bins
would put most of a trained classifier's rows in the top bin and estimate the
rest from nothing).

| Split | ECE before | ECE after | MCE before | MCE after | Temperature |
| --- | ---: | ---: | ---: | ---: | ---: |
| de_DE | 0.0106 | **0.0035** | 0.0955 | 0.0222 | 1.171 |
| en_US | 0.0157 | 0.0028 | | | 0.561 |
| all | 0.0050 | 0.0047 | | | 0.962 |

The rule engine's ECE of 0.16 and MCE of 0.44 are the point being made rather
than a defect being reported: its confidences are TIERS, which rule fired rather
than how likely it is to be right. That is why the decision layer in Section 5.3
reads the model's probabilities and not the heuristic's, and it is what "an
uncalibrated 0.9 is not a usable 0.9" means in numbers.

`triage report --autofill EVAL_DIR` renders this table and the reliability
diagram into the report. Without the flag the report is byte identical to what
it was before this section existed.

## Reproducing all of it

```text
triage autofill generate --out data --seed 0
triage autofill sweep    --data data --out runs
triage autofill evaluate --data data --weights runs/best.npz --locale de_DE --bootstrap 5 --out eval
triage ingest  runs      --database sweep.db
triage compare --database sweep.db --baseline lr0.03_l20
triage ingest  eval/runs --database eval.db
triage compare --database eval.db --baseline rules_de_DE_val --tags eval/macro_f1
triage report  --database sweep.db --baseline lr0.03_l20 --autofill eval --output autofill.html
```

Everything above is deterministic given the seeds, with one labelled exception:
`latency_us` in the outcomes file is a measured wall clock mean and moves with
the machine.

## Not here yet

`policy.py` (the fill or skip contextual bandit), the local LLM annotator, and
the agentic browser demo. `triage autofill evaluate --policy llm` is declared and
refuses with "not yet implemented until part 11" rather than falling back to the
model, because a silent fallback would report LLM numbers that no LLM produced.
