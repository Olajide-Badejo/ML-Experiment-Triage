# Engineering log

Dated entries, one per problem worth remembering. Each records the symptom, what
actually caused it, what the options were, what I did and why, and how I know it
is fixed. The debug report PDF is built from this file.

---

## 2026-08-05: no Python 3.13 on the machine

**Symptom.** `py -0p` listed only 3.14. The build specification asks for 3.13,
falling back to 3.12 only if a dependency refuses.

**Root cause.** Nothing broken; the machine simply had a newer interpreter than
the project targets.

**Options.** (a) Build against 3.14 and see what breaks. (b) Install 3.13.
(c) Fall back to 3.12.

**Fix and why.** Installed 3.13.14 with `py install 3.13`. Building against 3.14
would have meant betting on the wheel situation for tensorboard and protobuf,
which are the two dependencies here most likely to lag a new interpreter, and
any failure would then have been ambiguous between my code and a missing wheel.
Option (c) was available but there was no reason to go backwards.

**Verification.** The venv reports 3.13.14; every pinned package installed from
a wheel; a TensorBoard write and read round trip ran before any parser code was
written, so a later parser failure could not be blamed on the toolchain.

---

## 2026-08-05: the dash guard flagged a file that is not in the repository

**Symptom.** The very first run of `scripts/check_no_dashes.py` failed on an em
dash at line 208 of the build specification file sitting in the working
directory.

**Root cause.** The guard walked the filesystem. The specification file is
ignored and is an input to the work rather than part of the project, so
filesystem scope was answering the wrong question. The same bug would later have
flagged anything inside `.venv` or a build directory.

**Options.** (a) Add path exclusions and keep adding them forever. (b) Scope the
guard to what git would ship.

**Fix and why.** Option (b): the guard now asks
`git ls-files --cached --others --exclude-standard`, which is exactly the set of
files that are or will become part of the repository. Exclusion lists rot;
asking git does not.

**Verification.** The guard passes on a clean tree and still fails on a planted
dash in a tracked file. It runs in `make check-style` and twice in `make all`,
once before the PDFs exist and once after, so the compiled PDFs are checked too.

**Commit.** Phase 0, `5ef7208`.

---

## 2026-08-05: the window block mode was running at 12 percent type I error

The most important entry here. The calibration suite existed for exactly this
and found it on its first full run.

**Symptom.** With a nominal alpha of 0.05 and a true effect of exactly zero, the
single run window block mode rejected the null on **12.0 percent** of cases,
against a gate of [2, 8]. The seed replicated mode passed on its first run at 4.6
percent, so the defect was specific to the block machinery.

**Investigation.** The diagnostic printed the block count, block length and
autocorrelation time per case, then swept type I error against window length,
autocorrelation level, block length multiplier, and whether the statistic was
smoothed. Two things fell out of that grid.

**Root cause, part one.** The autocorrelation time was estimated on the two
windows *concatenated*. Joining two independent runs end to end puts a step
change at the join, and the estimator reads a step change as long range
dependence. Measured, it inflated tau by roughly a factor of two. It was also
masked: the inflated tau was then clipped by a target block count, so the net
effect was neither reliably conservative nor reliably wrong, which is the worst
kind of bug to have in a statistic.

**Root cause, part two.** The block length was one tau. At one autocorrelation
time the block means are still visibly correlated with their neighbours. The
observed split is contiguous (one run's blocks against the other's), but a
permuted split scatters neighbouring blocks across both groups, which breaks that
correlation and produces a null distribution narrower than the truth. A null that
is too narrow is precisely how a test becomes anticonservative.

Smoothing made this worse in a way I had not anticipated. The 9 point moving
average that the statistic uses raises the measured autocorrelation time of the
window from about 8 to about 14, so the blocks needed to be longer than the raw
series would have suggested.

**Options.**
(a) Keep one tau and accept the error rate. Rejected: a calibration miss is a
stop the line defect, and the whole claim of the project is that these p values
mean something.
(b) Raise the multiplier and clip the block length when the window is too short.
Rejected after measuring it: clipping reintroduces exactly the failure, and the
measured error rate at short windows stayed above 10 percent.
(c) Raise the multiplier and refuse when the window cannot support it.
(d) Abandon the single run mode entirely. Rejected: sometimes one run is all
there is, and a labelled weak answer beats no answer.

**Fix and why.** Option (c). Tau is now estimated per window with the larger
taken, the block is three times tau, and if the window cannot supply eight such
blocks the comparison raises a `ComparisonError` naming the concrete remedies
(log more steps, widen the window, or supply seed replicates) rather than
returning a number. Three was chosen by measurement, not taste: across window
lengths from 1200 to 8000 steps and autocorrelation coefficients from 0 to 0.9,
one tau gave 9 to 14 percent, two gave 7 to 10, three landed on nominal wherever
the precondition was comfortably satisfied.

Refusing rather than clipping is the part I would defend hardest. A p value that
is quietly miscalibrated is worse than no p value, because it spends credibility
that nothing later can recover.

**Verification.** `test_block_mode_type_one_error_is_near_nominal` measures 5.81
percent over 1928 null cases, with 72 refused as too short.
`test_block_mode_refuses_a_window_it_cannot_calibrate` pins the refusal.
`test_block_mode_power_on_a_large_effect` confirms the fix did not simply make
the test deaf: power is 100 percent on a large effect.

**Commit.** Phase 2, `4e6eb8e`.

---

## 2026-08-05: a selection effect in my own measurement

**Symptom.** After the fix above, the measured type I error still looked high in
some cells of the sweep: 20 percent at 3000 steps with an autocorrelation
coefficient of 0.9, against 5.6 percent at 8000 steps.

**Root cause.** Not the estimator. The refusal rule decides using tau estimated
from the same data it then tests. In the cells where most cases were refused, the
survivors were exactly the cases whose tau happened to be underestimated, which
are exactly the cases that get blocks that are too short. Conditioning on
survival selected for the failure. The cells with a refusal rate under 10 percent
all measured between 4.4 and 5.8 percent, and the error rate rose monotonically
with the refusal rate, which is the signature of a selection effect rather than a
broken statistic.

**Options.** (a) Refuse using something independent of the data, which for an
unknown autocorrelation time is not available. (b) Report the conditional rate
and say so. (c) Set the calibration suite inside the region where the
precondition is comfortably satisfied, so the measured rate is effectively
unconditional.

**Fix and why.** Option (c), with (b) written down. The calibration suite runs at
8000 steps, where under 4 percent of cases are refused, and the suite asserts
that refusal rate stays below 10 percent so the measurement cannot silently drift
into the biased regime. The honest statement, which is in the methodology, is
that the measured rate is conditional on the precondition being met, and that the
tool refuses rather than guesses when it is not.

**Verification.** The suite asserts both the error rate and the refusal rate.

---

## 2026-08-05: every HTML report differed from the last

**Symptom.** `test_report_is_byte_identical_when_rerun` failed immediately on
being written. Two renderings from the same database, at the same seed, differed.

**Root cause.** Plotly assigns each figure a freshly generated UUID as the div id
when it renders to HTML. Nothing about the data changed; the container changed.

**Why it mattered more than it looked.** The design says reports are pure
functions of the database plus a recorded permutation seed, and the README says
so. A file that differs on every run makes that claim unverifiable: nobody can
diff two reports to confirm nothing moved, which is the practical way anyone
would ever check it.

**Options.** (a) Strip generated ids after rendering, which is fragile against
Plotly's own internal references to them. (b) Pass an explicit `div_id`.

**Fix and why.** Option (b), with the id derived from the metric name, which is
already unique per figure and is also more readable in the output.

**Verification.** Two renderings now agree line for line, ignoring only the
generation timestamp, which is the one thing that legitimately differs.

**Commit.** Phase 3, `198dfab`.

---

## 2026-08-05: I was wrong about the learning rate, and the tool was right

**Symptom.** `test_learning_rate_is_the_strongest_influence_on_the_loss` failed.
Batch size correlated more strongly with the validation loss than learning rate
did, which is not what I expected from a sweep whose largest effects are all
driven by the learning rate.

**Root cause.** Not a bug. The sweep gives learning rate an optimum in the middle
of its range: 0.0003 is worse than 0.001, 0.003 is the best of the four, and 0.01
is by a distance the worst. That is a U shape. Spearman rank correlation measures
monotone association, and the rank correlation of a U shape is near zero no
matter how large the effect. Batch size was swept monotonically over the same
runs, so it correlates more strongly while driving a smaller effect.

**Options.** (a) Change the demo sweep to make learning rate monotone, so the
test passes. Rejected, and it is worth saying why: it would have made the demo
flatter the method by hiding the exact case where the method is blind, and a
learning rate with an optimum is the realistic shape, not the awkward one.
(b) Add a nonlinear measure such as distance correlation or mutual information.
Rejected for now as scope; noted as future work.
(c) Keep the behaviour, document the limitation, and pin it with a test.

**Fix and why.** Option (c). The limitation is now in the `sensitivity` module
docstring, in the methodology, in the README's "what it will not tell you"
section, and printed under the sensitivity table in the HTML report. The test was
renamed `test_rank_correlation_is_blind_to_the_non_monotone_learning_rate` and
now asserts the blindness deliberately, so that a future reader who sees a weak
learning rate correlation does not "fix" it. A second assertion confirms the
effect the correlation cannot see is genuinely there and correctly sized: the
tool recovers a gap of 0.193 between the best and worst conditions against a
designed 0.190.

**Verification.** Two tests, one at unit level on an isolated U shape and one at
integration level on the real sweep.

**Commit.** Phase 3, `198dfab`.

---

## 2026-08-05: two test failures that were the test's fault, not the code's

Recorded because being wrong in this direction is worth noticing too.

**The smoothing bleed.** `test_window_statistic_is_the_mean_of_the_tail` built a
step function, 900 zeros then 100 fives, and asserted the statistic was 5.0
within 0.05. It measured 4.9444. The 9 point moving average reaches back across
the step, so the four points at the start of the window each borrow some of the
value before it, a deficit of `4 * (4+3+2+1) / 9` spread over 100 points, which
is exactly 0.0556. The code was right and my tolerance was wrong. The test now
asserts the derived value to six decimal places and a second test covers the
clean case, which is a stronger check than the loose one it replaced.

**The self containment check.** I asserted that the string `cdn.plot.ly` never
appears in the HTML. It does, inside the inlined Plotly bundle, as a default
topojson host for choropleths this report never draws, alongside map tile
attributions. None of it is fetched. The test now strips the inlined scripts and
checks the remaining markup for loading tags, which is what actually causes a
network request.

**Fix and why.** Both tests were rewritten to assert the real property rather
than a proxy for it. A test that fails on correct code is not a strict test, it
is a broken one, and loosening the assertion would have been the wrong repair in
both cases.

---

## 2026-08-05: string hashing would have broken reproducibility silently

**Symptom.** None. Caught by reading, before it could produce a wrong result.

**Root cause.** The synthetic sweep seeded each run's generator with
`hash(condition.name)`. Python randomises string hashing per process unless
`PYTHONHASHSEED` is set, so the sweep would have been different on every
invocation while looking entirely deterministic in the code.

**Why it would have been nasty.** Every test in the repository would still have
passed, because they all regenerate the sweep in the same process. The failure
would have surfaced only as the committed verdict record never matching, and the
cause would have been very hard to see from that symptom.

**Fix and why.** `zlib.crc32` of the name, which is stable across processes and
versions. Keying on the name rather than an index also means adding a condition
later cannot shift the numbers of the conditions before it.

**Verification.** `make verify-demo` rebuilds the sweep from its seeds and checks
all twelve verdicts against the committed record to four decimal places.

---

## 2026-08-30: the published calibration was measuring one design out of six

**Symptom.** None visible, which is the point. Every gate was green and every
number in the README, the report and the HTML footer was reproducible. The suite
was measuring five seeds a side, one spread, Gaussian noise: the one design a
permutation test is exact in, and therefore the design least able to notice that
the statistic underneath it had just changed.

**Root cause.** Two of them, and they compound. The arms had been chosen when the
raw mean difference was the permuted statistic and balanced designs were all that
was tested, so nothing in the suite could see the failure that motivated
studentizing (D12). Separately, the block length multiplier of three had been
fixed by measurement against an autocorrelation time that part 03 then corrected:
tau had been estimated on the smoothed window, where a nine point moving average
leaves white noise looking correlated over nine points. Correcting tau left the
multiplier attached to nothing.

**Options.** (a) Publish the new statistics against the old arms. Rejected: it
would have meant claiming a calibration for designs never measured. (b) Add the
arms and gate them all at [2, 8] percent. Rejected after measuring: three runs of
a wide condition against seven of a narrow one measures 12.92 percent, and a gate
it cannot meet is either a permanently red build or, worse, an invitation to
widen the gate quietly later. (c) Add the arms, gate each regime at what its own
measurement supports, and publish every rate whether or not it flatters. Taken.

**Fix and why.** Six new arms, 1200 null cases per design cell, 1000 paired and
500 on macro F1, all deterministic. Before is the number published in v1.0.0 or,
where the arm is new, the rate the unstudentized statistic measured in the same
cell as recorded in the defect register.

| Arm | Before | After |
|---|---|---|
| Type I, seed replicated, 5v5 | 4.53 percent | 4.53 percent |
| Type I across seed sigma 0.005 to 0.05 | 4.40, 4.70, 5.00 | 4.40, 4.70, 5.00 |
| Power, seed replicated | 96.25 percent | 96.25 percent |
| Type I, window block | 5.81 percent, 1928 cases, 72 refused | 6.28 percent, 1989 cases, 11 refused |
| Power, window block | 100 percent, 291 cases | 100 percent, 300 cases |
| Null p uniformity at 0.05/0.10/0.25/0.50 | 0.058, 0.099, 0.241, 0.485 | 0.0580, 0.0993, 0.2407, 0.4847 |
| Weak mode cost at seed sigma 0.01/0.02/0.04 | 53.2, 77.0, 87.4 percent | 54.3, 76.6, 87.7 percent |
| Seed replicated on the same seed variance | 4.4, 4.7, 5.0 (a different arm) | 4.1, 3.8, 4.1 percent |
| Unequal spread, 5 narrow against 5 wide | 8.25 percent, unstudentized | 7.83 percent |
| Unequal spread, 3 narrow against 7 wide | 1.08 percent, unstudentized | 2.08 percent |
| Unequal spread, 7 narrow against 3 wide | 17.92 percent, unstudentized | 12.92 percent |
| Unequal counts only, 7 against 3 | not measured | 4.25 percent |
| Heavy tailed noise, Student t at 3 df | not measured | 5.00 percent |
| Paired clustered, 10 clusters of 4 | not measured | 4.90 percent |
| The same data, clustering ignored | not measured | 12.10 percent |
| Paired clustered on macro F1 | not measured | 2.00 percent |

Three gates, because the three regimes differ and one gate would have had to be
either dishonest about the worst cell or useless on the best. Equal spreads and
heavy tails keep [2, 8] percent. Unequal spreads with five or more runs a side
get [2, 10]: studentizing is asymptotic in the number of runs and five a side
lands near nominal rather than on it. Unequal spreads with fewer than five runs
on one side get [1, 15], set so the defect it exists to catch fails it, since the
raw mean difference measured 17.92 percent there. The macro F1 arm gates on its
upper bound alone: a statistic that discrete cannot produce a uniform p value, so
it is conservative, and conservative is the safe direction.

The two numbers in the weak mode cost table were measured at different seed
deviations from each other until now. The strong column was borrowed from the
seed variance sweep, at 0.005, 0.02 and 0.05, and printed against rows labelled
0.01, 0.02 and 0.04. The numbers were close and the mislabelling was invisible,
which is exactly the kind of thing this suite exists to stop; both columns now
come out of one arm at one set of deviations.

**The block length multiplier, re-derived.** Swept over 2000 null cases a point
with the corrected tau estimate, against the refusal rate on runs of a realistic
length at rho 0.8:

| Multiplier | Type I | Refused of 2000 | Refused at 4000 steps |
|---|---|---|---|
| 2 tau | 8.85 percent | 1 | 0 percent |
| 3 tau | 7.02 percent | 5 | 6 percent |
| 4 tau | 6.28 percent | 11 | 20 percent |
| 5 tau | 6.09 percent | 46 | 40 percent |
| 6 tau | 6.03 percent | 108 | 78 percent |

The error rate is asymptotic at about 6 percent, so four tau takes three quarters
of the correction that is available at all and every longer block takes almost
nothing while refusing far more comparisons. Two tau is outside the gate
altogether. Four is what the mode now uses. It is the one constant here that
should be expected to move again if the window statistic changes.

**Verification.** `pytest tests/statistics -q -s`, 17 passed in 110 s on the
machine in the README, every gate green on its own measurement. The full suite is
303 passing in 130 seconds. Every published number is rendered from `triage/calibration.py`:
the LaTeX tables through `scripts/gen_report_assets.py` and the Markdown through
`scripts/render_calibration_docs.py`, whose `--check` mode runs in CI so the two
sides cannot drift apart again.

**A defect found on the way.** The dash guard failed on the rebuilt main report.
`RegressionConfig.describe()` prints the flag `--fdr`, and TeX sets two hyphens
as an en dash, so the abstract of the PDF carried a banned character and, worse,
the wrong flag. `escape()` in the asset generator now breaks the ligature. The
guard caught it in the compiled PDF, which is the reason it reads PDFs at all.

---

## 2026-08-30: a 400 from a running Ollama was reported as "no Ollama"

**Symptom.** The first end to end run of `make demo-autofill` against the real
service printed

    error: no Ollama at http://localhost:11434 (HTTP Error 400: Bad Request).
    Start it with `ollama serve`

while Ollama was running, answering `/api/tags`, and holding two models in VRAM.
The advice was to restart a service that was working.

**Root cause.** Two bugs stacked, and the outer one hid the inner one.

`urllib.error.HTTPError` subclasses `URLError` which subclasses `OSError`, so the
`except (URLError, OSError)` that turns a refused connection into
`LlmUnavailableError` was also catching every 4xx and 5xx the server sent. A
response is the opposite of an absent service, and the message threw away the
body, which is where Ollama had put the actual explanation.

The 400 underneath it was real: this build of Ollama serves an `/api/embed`
batch of about 288 inputs and fails above roughly 300, with the runner reporting
`Post "http://127.0.0.1:PORT/tokenize": connection refused`. Its embedding runner
subprocess dies and restarts, and the next smaller batch succeeds. Verified
repeatedly at 300, 512 and 900 inputs, each time with a 128 input batch
succeeding immediately afterwards, and at 128, 256, 257 and 288 succeeding
throughout. The batch size here was 512, chosen on the reasoning that the
endpoint takes an array and one request beats nine hundred, with no measurement
behind the specific number.

**Options.** (a) Catch `HTTPError` first and lower the batch size. (b) Retry a
failed batch at half the size, which would hide a server that is unwell behind a
slower path that usually works. (c) Send one text per request, which is the
design this endpoint exists to avoid and would have made the corpus embedding
463 round trips instead of four.

**Fix and why.** Option (a), both halves. `HTTPError` is now caught ahead of
`OSError`: a 404 is an `LlmUnavailableError` naming `ollama pull`, and any other
status is an `LlmProtocolError` carrying Ollama's own `{"error": ...}` text and
saying plainly that restarting the server will not help. And
`DEFAULT_BATCH_SIZE` is 128, which is comfortably under the measured boundary and
still makes the de_DE training split four requests rather than 463. The number
is now a measurement with the measurement written beside it.

**Verification.** Three unit tests drive a fake transport that raises a real
`HTTPError`: a 400 whose message survives and does not mention `ollama serve`, a
404 that names the pull command, and a non JSON error body that is still
reported. The demo then ran end to end against the real service.

---

## 2026-08-30: the Section 2 co residency question, answered

**The question.** Section 2 left one thing unverified: whether Ollama keeps the
12B chat model and the embedder simultaneously resident on a 12,227 MiB card.
The arithmetic said yes, 7.5 GB plus 0.6 GB plus caches, and the fallback if it
did not was sequential loading, which the batch design already permits.

**Measured**, during `triage llm annotate` on the real service, with
`/api/ps` and `nvidia-smi` sampled while the annotation was running:

| Model | Resident | In VRAM |
|---|---|---|
| `mistral-nemo:12b-instruct-2407-q4_K_M` | 7.88 GB | 7.88 GB |
| `embeddinggemma:300m` | 0.68 GB | 0.68 GB |

`nvidia-smi` reported **9,207 MiB of 12,227 MiB** in use at the same moment, with
both models fully on the GPU and neither offloaded. No eviction was observed
across a 460 second annotation pass that alternated embedding and generation.

**Conclusion.** Co residency holds on this card at the default 4096 token
context. The sequential loading fallback is not needed and is not implemented as
a special case; the batch design that would have made it work is there anyway,
because embedding the corpus in one pass before any generation is simply the
faster order.

---

## 2026-08-30: the LLM layer, measured

Recorded because Section 6.6 asks for the real service run to be in the log
rather than only in the docs. Machine: RTX 5070, 12,227 MiB; Ollama 0.32.14;
`mistral-nemo:12b-instruct-2407-q4_K_M` and `embeddinggemma:300m`.

**The ablation, which is the claim.** `triage llm annotate --ablation` scored 90
de_DE validation fields twice and had the comparison layer judge the difference:

| Arm | Macro F1 | Requests | Wall clock |
|---|---|---|---|
| Zero shot | 0.8007 | 90 | 221.1 s |
| Retrieval augmented, k = 8 | 0.8879 | 0 (90 cache hits) | 0.0 s |

Verdict **improvement, p 0.0079, adjusted p 0.0212**, +0.0872 macro F1 over 5
bootstrap replicates. Retrieval augmented few shot selection is worth having on
this task, and the number is a verdict from `compare_all` rather than a claim.

**Against the other engines**, same split: keyword rules 0.7420 macro F1, n gram
model 0.9381, LLM 0.8879. Both learned engines beat the rules at p 0.0079. The
pure numpy classifier beats the 12B model and answers in microseconds against
its 5.04 seconds a field, which is worth stating plainly: the LLM is a useful
third condition here, not the best one.

**The cache round trip.** The retrieval arm above made zero HTTP calls because a
previous run had asked those exact questions. 90 of 90 served from
`llm_responses`, identical answers, 0.0 s. That is the reproducibility property
of 6.3 observed rather than asserted.

**Retrieval latency**, 463 de_DE training vectors at 256 dimensions, a 0.47 MB
matrix: exact brute force search at k = 8 costs a mean of **15.5 us** (median
14.9, p95 18.6, n = 2000), reading the whole cached corpus out of SQLite costs
1.7 ms, and the HTTP round trip that produces the query vector costs 2,233 ms.
The search is about 144,000 times cheaper than embedding the query it searches
with. That is the whole argument against a vector database at this scale, and it
is in `docs/llm.md` in one line.

**Wall clocks.** Corpus generation and the sweep of 18 runs: 3.7 s. Model and
heuristic evaluation: under a second. LLM annotation of 90 fields on a cold
cache: 459.1 s (5.1 s a field), and 0.0 s on a warm one. The whole
`make demo-autofill` chain: 460.5 s with the LLM step, 16.5 s without it.

---

## 2026-08-30: the agentic demo, in a real browser

Section 5.4 asks for the demo to be run on a real machine and the wall clock
recorded here rather than only in the docs. Machine: Windows 11, Chrome
151.0.7922.174 found by choreographer 1.3.0, headless, one browser process for
all six pages; Ollama 0.32.14 with `mistral-nemo:12b-instruct-2407-q4_K_M` for
the third arm.

**What was run.** A 4,000 field corpus at seed 0 (six pages, three templates in
two locales, 71 inputs between them), the thirty run sweep, one evaluation over
both locales to export `policy.json`, then `triage autofill agentic` three
times into one output directory, once per engine, filling under the exported
`per_type_threshold` policy.

| Step | Wall clock |
|---|---|
| `autofill generate` (4,000 fields) plus `autofill sweep` (30 runs) | 21 s |
| `autofill evaluate --locale all --bootstrap 5` | 1 s |
| `autofill agentic --policy heuristic` (6 pages, browser) | 0.7 s |
| `autofill agentic --policy model` (6 pages, browser, plus the screenshot) | 0.8 s |
| `autofill agentic --policy llm` (6 pages, browser, cold cache) | 434.1 s |
| The same LLM arm on a warm cache | 0.9 s |
| `ingest` 90 run directories plus `compare` | under 2 s |

**The result table**, 71 fields over six pages, scored against `data-truth`
read back out of the DOM:

| Engine | Filled | Fill accuracy | Reward per field | Correction cost |
|---|---:|---:|---:|---:|
| Keyword rules | 30 | 0.8667 | 0.2394 | 0.0563 |
| n gram model | 64 | 1.0000 | 0.9014 | 0.0000 |
| Local LLM, kNN k = 8 | 42 | 0.9286 | 0.4789 | 0.0423 |

Handed to `triage compare` with no autofill specific flag, baseline `rules`:
`ngram` +151.40% on `task/reward` (p 0.0001, adjusted 0.0002) and +28.19% on
`task/fill_accuracy` (adjusted 0.0002); `llm` +62.07% and +18.44% (adjusted
0.0033 and 0.0195). Both learned engines beat the keyword baseline on the task
metric, and the pure numpy model beats the 12B model at six orders of magnitude
less wall clock a field, which is the same ordering the validation split gave.

**Two things the browser found that a dictionary would not have.** The first
run refused every page with "relative path can't be expressed as a file URI",
because a `file://` URI is absolute by definition and `--pages
experiments/results/.../pages` is not; the fix is one `resolve()` and there is
now a test that opens a page by a relative path. The second was cosmetic and
still worth having: the capture was the emulated 1000 by 900 window rather than
the page, so a third of the committed PNG was white. The capture now crops to
the body's box, which is D35a's argument applied to a screenshot nobody had
taken yet.

**The screenshot.** `assets/images/agentic-demo.png`, the en_US checkout page
after the model filled it: 1000 pixels wide at device scale factor 1 (D35a),
recompressed at zlib level 9, **21.8 KB**. For comparison, the same capture at
the full window height before the crop was 23.5 KB and at scale 2 would have
been about four times that.

**Cache round trip.** The LLM arm was run twice. The second pass served 71 of 71
answers from `llm_responses`, made zero HTTP requests, took 0.9 s against
434.1 s, and wrote a byte identical summary. That is the reproducibility
property of 6.3 holding for a browser driven run as well as for a split.
