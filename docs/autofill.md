# The autofill vertical

Browser form field classification is this repository's reference workload: the
machine learning problem behind autofill, which is deciding that an input is a
given name, a postal code or a card security code, across locales whose address
formats disagree. It is here because a tool that ranks training runs needs a
training run of its own to rank, and this problem exercises every axis the tool
advertises at once: a dataset with a definition, a sweep with seeds, two
conditions to compare, calibrated probabilities to check, and a cross sectional
evaluation output that is not a time series.

!!! note "This is a demo workload, not a product"

    It is **not** a competitor to `Autofill_audit`, which is the production
    grade auditor (a 392 rule engine over nine locale vocabularies, a calibrated
    ONNX model, a Playwright driver), and it is not a competitor to the PyTorch
    Performance Toolkit's transformer field classifier, which is the production
    grade trainer. It is pure numpy, it trains in under a second, and its job is
    to be a workload this repository can be honest about end to end. See
    [Ecosystem](ecosystem.md) for who does what.

## The taxonomy

### Where the spelling comes from

`FieldType` is a `StrEnum` whose VALUES are WHATWG autocomplete tokens, taken
verbatim for every token included: eighteen tokens plus `unknown`. The value
space is `autofill_audit.taxonomy.Label`'s, so a label written here means the
same thing there, and `from_tpt_label()` maps the PyTorch Performance Toolkit's
seventeen snake_case labels onto it.

Three projects that disagree about how to spell a postal code cannot exchange a
single evaluation file. Adopting the standard's spelling rather than inventing a
fourth makes the strings interoperate by construction (E7b), and because the
enum is a `StrEnum` a member *is* its token: it serialises to JSON as
`"postal-code"` and compares equal to the string a sibling repository wrote,
with no `.value` at any boundary.

This taxonomy is a strict SUBSET of `Autofill_audit`'s thirty seven tokens,
chosen to cover the identity, address and payment groups a checkout or
registration form actually contains, which is what the generator below
produces. A label the mapping does not recognise raises rather than becoming
`unknown`, because mapping it would turn a taxonomy mismatch into an accuracy
figure that quietly counts the wrong class as a correct refusal.

### `unknown` is a real class, not a null

A form holds newsletter checkboxes, comment boxes and honeypots, and a head that
cannot say "none of these" has to spend probability mass on a wrong answer
instead. It is the last index by convention, so a caller that only fills the
tokens it recognises can slice the head down to a prefix.

The declaration order is the model head order, and therefore part of the on disk
format of a saved `.npz` of weights: tokens are appended, never reordered.

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
serious one. Three settings were tried, and the middle one is what is published:

* **too easy** (dropout 0.18, generic 0.22, placeholders 0.35, autocomplete
  present 0.35 and correct 0.80): the model reached 0.994 macro F1 on de_DE
  against the heuristic's 0.754. A real margin, but over a corpus so clean that
  the model sat at its ceiling and the sweep had nothing left to rank;
* **the values above**: the model reaches about 0.96 and the heuristic about
  0.72, both visibly imperfect, and the learning rate still moves the result;
* **too hard** (dropout 0.50): the heuristic falls to 0.60, and what is being
  measured is how much signal was removed rather than what a classifier is
  worth.

Every corpus writes these values into `meta.json` beside it, because a
difficulty setting nobody wrote down makes every number downstream
unreproducible.

## Features and model

### The features

Hashed character n grams, n in 3 to 5, over six namespaced signals (`label:`,
`name:`, `ph:`, `ac:`, `type:`, `ctx:`), folded into 2^16 slots with
`zlib.crc32`.

Never `hash()`: Python randomises string hashing per process, so a model trained
in one process would score differently in the next and nothing would say so.

Two details do work that is easy to miss:

* an absent signal emits an explicit `<absent>` marker, because a missing label
  is evidence rather than the absence of evidence;
* a constant `bias:1` token gives the model its intercept without a dense
  column.

### The model

Multinomial logistic regression in pure numpy: softmax cross entropy, minibatch
SGD with momentum, L2, seeded reshuffling per epoch.

L2 and momentum are applied to the rows a batch TOUCHES rather than to the whole
65,536 by 19 matrix, which is the standard sparse linear update and is what
makes the sweep fit in the time budget. The consequence, that a weight's
momentum decays in its own step count, is written down in the module rather than
left to be found in the numbers.

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

`evaluate.py` also writes a per field `run.jsonl` in the abridged 11 key form of
`Autofill_audit`'s E5 schema, which is what this repository's own workload has
rows for. It is not their file: theirs carries 25 keys, and the committed
fixture under `tests/fixtures/outcomes` is the one that models those.

One difference matters to anyone copying the join out of here. This generator
writes a `form_id` per FIELD, so `form_id` is a unit key in this table and
`pair_on("form_id", ...)` is right for it. `Autofill_audit`'s writer emits one
row per field under a `form_id` that names the FORM, with a `selector` naming
the field, so their unit is the `(form_id, selector)` pair and pairing on
`form_id` alone raises `SeriesError`. [The library page](library.md) has the
composite spelling.

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

### The fill or skip decision policy

A classifier reports a distribution; a product has to choose an action, and the
two are not the same problem. Autofill's product truth is that a wrong fill
costs the user more than no fill, so whether to fill a field at all is its own
decision. It is modelled as a **contextual bandit** over the context
`(predicted type, calibrated confidence, locale)` with the two actions `fill`
and `skip`. A bandit and not reinforcement learning, and the distinction is
load bearing rather than pedantic: nothing one field's decision does changes the
next field's context, there is no episode and no discounting, and calling it RL
would claim machinery that is not here.

The reward is `+1` for a correct fill, `0` for a skip, and a type dependent
penalty for a wrong one, in units of one correct fill:

| Cost tier | Tokens | Wrong fill |
| --- | --- | ---: |
| payment | `cc-name`, `cc-number`, `cc-exp-month`, `cc-exp-year`, `cc-csc` | -4 |
| identity | `given-name`, `family-name`, `name`, `email`, `tel`, `username` | -4 |
| address | the four address lines and levels, `postal-code`, `country-name` | -2 |
| other | `organization`, `unknown` | -1 |

Configurable through `--penalties payment=8,address=3`; tiers left out keep the
default. `organization` sits in `other` beside `unknown` rather than in
`address`, because a company name is neither a person's identity nor a locative
component of an address, and getting one wrong is a typo rather than a leak.

**The penalty is charged against the TRUE type of the field, not the predicted
one.** A wrong fill leaves a bad value in a field whose type is what it is: the
user goes and repairs a payment field, an address field or a comment box, and
the work is the work regardless of what the model thought it was doing. That has
a consequence worth stating rather than discovering, and it is exactly what
makes this a bandit problem: **the cost of an action is not observable at
decision time**, because the policy sees only the predicted type. The latent
truth is what the reward depends on and what the policy never gets to look at,
which is why the learned thresholds are indexed by what the policy CAN see.

#### Why there is no importance sampling estimator, and why that is the honest answer

Off policy evaluation normally needs an estimator (IPS, self normalised IPS,
doubly robust) for one specific reason: logged data records the reward of the
action that WAS taken and nothing about the action that was not, so the value of
a new policy has to be reconstructed from a reweighted sample, at a variance
cost that grows with how far the new policy is from the logging one.

That reason does not apply here. The evaluation corpus is fully labelled and the
reward is a known function of `(prediction, truth, action)`, so BOTH arms of
every row are computable: filling earns `+1` or the penalty, skipping earns `0`.
A policy's expected reward is therefore a sum over rows rather than an estimate
with a confidence interval. Fitting an IPS estimator on top of that would add
variance to a quantity that is already exact, and reporting its error bars would
be theatre. What the exactness does NOT cover, and the module says so: it cannot
value a policy that would have shown the user a different set of fields.
Nothing here changes which fields exist.

#### Four policies, one threshold class

All four compared policies are one object: a vector of nineteen thresholds, one
per predicted type, with `fill` when the calibrated confidence reaches the
threshold for the predicted type. Always filling is the vector of zeros, never
filling the vector of twos (no probability exceeds one), a global cut the
constant vector, and the learned policy the interesting one.

That is not a trick to save code. It is the statement that these are the same
policy class evaluated at different points, and it has a caveat that has to be
read out loud: **because always filling is a member of the class, a per type
policy fitted on a split cannot lose to it on that same split, by
construction.** The in sample number below is therefore not evidence on its own,
which is why the out of sample number is measured too and why the report labels
the in sample case rather than quoting an unbeatable figure silently.

Fitting is exact and needs no search: expected reward is a sum over rows and the
predicted type partitions the rows, so maximising each type's own block
maximises the total. Within a block, sorting by confidence turns "every
threshold" into "every prefix" and one cumulative sum finds the best, with
candidate cuts taken only at the ends of runs of EQUAL confidence, since a
threshold cannot split two rows that look identical to it. A type with fewer
than twenty rows in the fitting split inherits the global cut rather than a
threshold decided by three rows.

#### Measured, on de_DE validation

287 rows, the sweep's best head (`lr0.3_l20` seed 0) at temperature 1.171,
thresholds fitted on the same split, which is the default and is labelled
`in_sample` in the artifact and in the report:

| Policy | Fill rate | Expected reward | Correction cost |
| --- | ---: | ---: | ---: |
| never fill | 0.0000 | 0.0000 | 0.0000 |
| always fill | 1.0000 | 0.8815 | 0.0732 |
| global threshold | 0.9512 | 0.9059 | 0.0279 |
| **per type threshold** (chosen) | 0.9373 | **0.9233** | **0.0070** |
| Thompson sampling (online) | 0.8711 | 0.7770 | 0.0592 |

Correction cost is what the user pays to undo the wrong fills, per field of the
form. It is deliberately not the negative of the reward: a skip costs the user
typing, which is what they were doing anyway, and charging for it would make
doing nothing look expensive. Read beside the raw accuracy above, this is the
number that changes: the per type policy gives up 6.3% of the fills and removes
**90% of the correction cost**, 0.0732 down to 0.0070.

Never filling is in the table because it is the bar. A policy that cannot beat
zero is worse than shipping nothing, and a comparison that omits it is grading
on a curve.

#### The online version, and what learning costs

The Thompson sampling row is the same decision learned ONLINE, one field at a
time, over a seeded permutation of the evaluation stream. The posterior is one
Gaussian per context cell over the mean reward of filling there, conjugate under
a Normal prior with known noise. Gaussian rather than the textbook Beta because
the reward is not a coin flip: it lives on `[-4, 1]` and its scale is exactly
what the decision turns on. Each step draws once from the cell's posterior and
fills when the draw beats the reward of skipping, which is Thompson sampling;
the exploration is the width of the posterior and nothing else.

**It updates only on the fills**, because a skip teaches a bandit nothing. That
partial feedback is the whole difference between this row and the ones above it,
and a simulation that updated on skips would be reporting numbers from full
feedback while calling itself a bandit.

The arms are `(locale, cost tier, equal mass confidence bucket)`, which is 2 x 4
x 5 = **40 cells**. Four choices went into that:

* **the nineteen predicted types are pooled down to their four cost tiers.** The
  arms have to be learnable from one pass over a few hundred decisions, and
  `2 x 19 x 5` arms would leave most of them with three observations and turn
  the simulation into a demonstration of the prior;
* **the fitted threshold policy still uses all nineteen types**, because it is
  fitted offline on the whole split at once, where support is not the binding
  constraint;
* **bucket edges are equal MASS**, the same choice the calibration bins make and
  for the same reason;
* **arm indices are computed arithmetically** from integer codes rather than
  from a dictionary keyed by tuples, and every draw comes from one
  `SeedSequence`, so the whole simulation is byte reproducible under its seed.

Over 287 decisions it gave up **42.00 of reward against the best fixed policy**,
which is 0.1463 per decision. That gap is the price of learning the policy
rather than being handed it, and it is measured rather than bounded. Starting
from a flat prior and never seeing the reward of anything it declined to fill,
the bandit still reached 84% of the best fixed policy's reward inside one short
stream. Regret can come out negative, which would not be a bug: the fixed class
is restricted, and a bandit that adapts within a stream can beat every member of
it on that stream.

#### The finding that goes the other way

On the held out test split the same thresholds do NOT beat always filling:

| Policy | Fill rate | Expected reward | Correction cost |
| --- | ---: | ---: | ---: |
| **always fill** | 1.0000 | **0.9619** | 0.0235 |
| global threshold | 0.9736 | 0.9589 | 0.0088 |
| per type threshold | 0.9619 | 0.9472 | 0.0088 |
| Thompson sampling (online) | 0.8768 | 0.8534 | 0.0147 |

341 rows, thresholds fitted on val and scored here, so this is the version of
the claim that costs something. The head is 0.9853 accurate on this split
against 0.9547 on validation, and that is the whole explanation:

!!! warning "The value of a skip policy scales with the error rate"

    When the classifier is wrong three times in two hundred, the fills a
    threshold gives up are worth more than the mistakes it prevents. A near
    perfect classifier does not need a decision layer, and this table is what
    that looks like when it is measured rather than assumed.

There is a second effect underneath it, worth naming because it is the reason
the per type policy also loses to the single global cut here: nineteen
thresholds fitted on 287 rows generalise worse than one, and the support floor
of twenty shrinks most types onto the global cut but not all of them.

The online row moves with it: the best fixed policy here is always filling, and
the Thompson run gave up 37.00 of reward against it over 341 decisions, 0.1085
each. A bandit that has to learn a threshold pays for the lesson whether or not
the threshold turns out to be worth having.

Neither number was tuned to make this table look better. The support floor was
not raised against the test split, and the chosen policy is still the per type
one, because 5.3 names it and because the default evaluation is the validation
split where it wins. The honest summary is that the decision layer earns its
place at the error rates this corpus is generated at, and that a near perfect
classifier does not need one.

#### The artifact

`triage autofill evaluate` writes `policy.json` beside `calibration.json`: the
reward function, every policy's exact numbers, the fitted thresholds, the
Thompson summary, and which policy was chosen. `--decision-policy` names the one
exported, and the Thompson run is deliberately not an export target, because its
decisions are a function of the stream it saw and a demo needs a policy that is
the same on every run. The agentic demo loads that file; the report renders it
with no import of `triage.autofill` in the reporting layer, so the numbers on
the page are the ones that were measured.

## The agentic demo: the same classifier, driving a browser

Everything above is a number about a dataset. `triage autofill agentic` is the
product question the dataset stands in for: given a page, how many of its fields
get the right value typed into them, and what did the mistakes cost.

The loop is one page at a time. Headless Chrome opens the generated page over
`file://`; every input's signals are read out of the LIVE DOM with one
`Runtime.evaluate` (attributes plus the associated label text plus the enclosing
`fieldset`'s legend); the fields are classified by the trained model, the keyword
heuristic or the local annotator; the exported `policy.json` decides fill or
skip per field; the values come from a locale matched synthetic profile and are
set with `input` and `change` dispatched the way a user agent dispatches them;
then the page is read back and scored against the `data-truth` attribute the
generator embedded.

**The read back is a check rather than a formality.** A fill counts as correct
only when the predicted type is the true type AND the value read out of the
element afterwards is the value that was typed. Scoring the classifier's
intentions instead would report a perfect run through a browser that silently
dropped every keystroke, and "did the value land" is exactly the failure a
browser demo is supposed to be able to see.

**Synthetic, local, and nothing on the network.** The profiles are invented
identities using the ranges reserved for the purpose: `example.com` (RFC 2606),
the 555-0100 telephone block, and `4242 4242 4242 4242`, the card number every
processor documents as a test number (Section 10). The pages are `file://` URLs
in a temporary directory. There is no real site anywhere in it.

![The demo's checkout page, filled. Nineteen inputs across an account, a
shipping and a payment section, each holding the synthetic profile value for the
type the model predicted: an example.com address, a US address, and the
documented test card number 4242 4242 4242 4242.](https://raw.githubusercontent.com/Olajide-Badejo/ML-Experiment-Triage/main/assets/images/agentic-demo.png)

### Measured, in a real browser

Six pages, three templates in two locales, 71 fields, filling under the
`per_type_threshold` policy exported by the evaluation above. Chrome headless on
Windows, one browser for all six pages.

| Engine | Fields | Filled | Fill accuracy | Reward per field | Correction cost | Wall clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword rules | 71 | 30 | 0.8667 | 0.2394 | 0.0563 | 0.7 s |
| n gram model | 71 | 64 | **1.0000** | **0.9014** | **0.0000** | 0.8 s |
| Local LLM, kNN k = 8 | 71 | 42 | 0.9286 | 0.4789 | 0.0423 | 434.1 s |

The LLM column is `mistral-nemo:12b-instruct-2407-q4_K_M` on an RTX 5070, 6.1
seconds a field against the numpy model's microseconds, and it comes third on
both task metrics. That is the same ordering `docs/llm.md` measures on the
validation split, and it is worth stating plainly rather than burying: the local
model is a useful third condition on this task, not the best one. A second run
of it answers from the response cache, 71 of 71, no HTTP, in 0.9 s, and writes
byte identical results.

Per page, for the model:

| Page | Locale | Fields | Filled | Fill accuracy | Reward per field |
| --- | --- | ---: | ---: | ---: | ---: |
| `address_book` | de_DE | 9 | 9 | 1.0000 | 1.0000 |
| `address_book` | en_US | 10 | 7 | 1.0000 | 0.7000 |
| `checkout` | de_DE | 18 | 16 | 1.0000 | 0.8889 |
| `checkout` | en_US | 17 | 16 | 1.0000 | 0.9412 |
| `registration` | de_DE | 8 | 8 | 1.0000 | 1.0000 |
| `registration` | en_US | 9 | 8 | 1.0000 | 0.8889 |

Read the two tables together. The model fills more than twice as many fields as
the keyword baseline AND gets all of them right, which is the whole argument for
a calibrated confidence behind a per type threshold: the policy is not trading
coverage against correctness here, it is skipping the seven fields the model is
not sure about and being right about the rest. The rules engine is punished
twice over: it is wrong more often, and its six confidence TIERS cannot meet a
threshold fitted on probabilities, so it also skips fields it would have got
right.

### The task metrics land in the ordinary tool

Each `(engine, page, replicate)` is a run directory in the format `triage
ingest` already reads, logging `task/fill_accuracy` and `task/reward`. The
condition is the engine and the replicate index runs over `(page, resample)`
pairs, so replicate k is the same page and the same bootstrap resample of it for
every engine and the comparison is paired:

```text
candidate   metric               change          p      p adj  verdict
ngram       task/reward        +151.40%     0.0001     0.0002  improvement
llm         task/reward         +62.07%     0.0025     0.0033  improvement
ngram       task/fill_accuracy  +28.19%     0.0001     0.0002  improvement
llm         task/fill_accuracy  +18.44%     0.0195     0.0195  improvement
```

No autofill specific flag appears in `ingest` or `compare`, which is acceptance
criterion 1 again, this time for the task metrics rather than for the sweep.

### `--no-browser`, and why it is a real test

`--no-browser` parses the same page with the standard library's `html.parser`
and puts a dictionary where the DOM was. Everything after extraction is the same
code: the same `PageDriver` protocol, the same classification, the same policy,
the same scoring. That is what makes the CI run a test of the demo rather than
of a second implementation, and the browser test asserts the join directly: the
live DOM and the parser have to report the same fields for the same page, and
the two runs have to produce the same score, field by field.

The browser tests are marked `browser` and skip on a machine with no Chromium,
which is the same arrangement `docs/llm.md` describes for Ollama.

## Reproducing all of it

```text
triage autofill generate --out data --seed 0
triage autofill sweep    --data data --out runs
triage autofill evaluate --data data --weights runs/best.npz --locale de_DE --bootstrap 5 --out eval
triage autofill evaluate --data data --weights runs/best.npz --locale de_DE --bootstrap 5 --split test --out eval_test
triage ingest  runs      --database sweep.db
triage compare --database sweep.db --baseline lr0.03_l20
triage ingest  eval/runs --database eval.db
triage compare --database eval.db --baseline rules_de_DE_val --tags eval/macro_f1
triage report  --database sweep.db --baseline lr0.03_l20 --autofill eval --output autofill.html
```

And the demo, which needs `pip install "ml-experiment-triage[agentic]"` and a
Chromium (`--no-browser` needs neither):

```text
triage autofill evaluate --data data --weights runs/best.npz --locale all --bootstrap 5 --out eval_all
triage autofill agentic  --pages data/pages --out task --policy heuristic --decisions eval_all/policy.json
triage autofill agentic  --pages data/pages --out task --policy model --weights runs/best.npz --decisions eval_all/policy.json
triage autofill agentic  --pages data/pages --out task --policy llm --data data --decisions eval_all/policy.json --database llm.db
triage ingest  task/runs --database task.db
triage compare --database task.db --baseline rules
```

Everything above is deterministic given the seeds, with two labelled exceptions:
`latency_us` in the outcomes file is a measured wall clock mean and moves with
the machine, and the `llm` engine is a local model's answers, cached in
`--database` so that a second run reproduces the first exactly.
