# The ecosystem: three repositories, one author, one taxonomy

Three of my projects touch form field classification and training run analysis,
and two of them are mature. That is a situation with two failure modes. The
first is silent duplication: three generators, three taxonomies, three Ollama
clients, none of which can read each other's files. The second is a namespace
collision that makes the duplication invisible until an import breaks. This page
states what each repository is for, what this one deliberately does not compete
with, and what the shared contracts are.

## Who does what

| Repository | What it is | Relationship to this one |
| --- | --- | --- |
| [`Autofill_audit`](https://github.com/Olajide-Badejo/Autofill_audit) (dist `autofill-audit`, import `autofill_audit`) | The production grade form field auditor: a Playwright driven crawler and a classification ladder of a 392 rule engine across 9 locale vocabularies, a calibrated ONNX n gram model, and an optional Ollama fallback. | **A consumer.** It depends on this package and imports the statistics layer in exactly one module, `evaluate/triage_bridge.py`, with an AST test that fails if a second importer appears. Its five filed issues are the consumer contract this release closes. |
| [`PyTorch-Performance-and-Health-Toolkit`](https://github.com/Olajide-Badejo/PyTorch-Performance-and-Health-Toolkit) (dist `torch-perf-toolkit`, import `tpt`) | The production grade instrumentation engine: a PyTorch profiler feeding a health monitor, a throughput sweeper and an inference evaluator, plus a 0.57M parameter transformer field classifier and a 3 locale generator. | **An interop partner, not a dependent.** It does not import this package and cannot easily: its Python floor is 3.11 and this one's is 3.12. What connects them is its log schema, which this tool now parses (E6), and its `tpt.triage` explainer, whose grounding design this tool adopted with credit. |
| **This repository** (dist `ml-experiment-triage`, import `triage`) | Ranked, significance tested comparison of training runs, and a self contained autofill demo workload built to exercise it. | The statistics layer is the product. The autofill vertical is a **demo workload and CI fixture**, not a competitor to either sibling. |

The [autofill vertical](autofill.md) here is pure numpy and the standard
library, trains in seconds, and exists so that this tool has a workload with
known ground truth to be pointed at end to end. `Autofill_audit` is the auditor
you would actually run against a real site; TPT is the trainer you would
actually use to fit a real model. Both are stated as such rather than compared
against, because the comparison would be dishonest in this repository's favour
on speed and in theirs on everything that matters.

## The `TriageReport` collision, and why nothing was renamed

TPT ships a `tpt.triage` subpackage (an Ollama grounded explainer, about 2,164
lines), a `tpt-triage` console script, and a `tpt.triage.report.TriageReport`
class. This repository ships the distribution `ml-experiment-triage`, the import
package `triage`, a `triage` console script, and
`triage.analysis.regression.TriageReport`.

Nothing collides at import time. `tpt.triage` is a subpackage of `tpt` and is
never reachable as a top level `triage`, so both can be installed into one
environment and both import correctly. The collision is entirely in a reader's
head, and in one line of code:

```python
from triage.analysis.regression import TriageReport  # this package
from tpt.triage.report import TriageReport  # shadows the first
```

**Decision: this package keeps the name `TriageReport`** (E7a). It is in the
published `__all__`, a downstream project constructs it, and renaming a class to
avoid a collision that the import system does not have would break a real
consumer to fix a documentation problem. The rule is therefore a documentation
rule, and it is short: **never import both in one example, one docstring or one
test.** Where both have to be discussed, they are spelled with their packages, as
they are above.

The console scripts do not collide either: `triage` and `tpt-triage` are
different names, which is a piece of luck rather than a plan and is recorded
here so it stays true.

## One value space for field labels

Three projects that disagree about how to spell a postal code cannot exchange a
single evaluation file. So there is one spelling, and it is not mine:

- `Autofill_audit` ships a 37 token `autofill_audit.taxonomy.Label` StrEnum
  whose values are **WHATWG HTML autocomplete tokens**, validated against a 960
  form corpus.
- TPT ships a 17 label taxonomy in `snake_case`.
- This repository's `triage.autofill.taxonomy.FieldType` adopts the WHATWG
  hyphenated value space **verbatim** for every token it includes: 18 tokens
  plus `unknown`, a strict subset of `Autofill_audit`'s (E7b).

Because `FieldType` is a `StrEnum`, a member *is* its token: it serialises to
JSON as `"postal-code"`, compares equal to the string `Autofill_audit` wrote,
and needs no `.value` at any boundary. An evaluation file written by one project
is readable by the other without a translation table.

TPT's `snake_case` labels are the one case that does need translating, and the
bridge is documented rather than inferred:

```python
from triage.autofill.taxonomy import from_tpt_label

from_tpt_label("postal_code")  # FieldType.POSTAL_CODE, "postal-code"
from_tpt_label("postal-code")  # already correct, returned unchanged
from_tpt_label("shoe_size")  # KeyError naming the label and the mapping
```

An unrecognised label raises rather than mapping to `unknown`, because mapping
it would turn a taxonomy mismatch into an accuracy figure that silently counts
the wrong class as a correct refusal.

`unknown` is a real class here and the last index by convention, so a caller
that only fills the tokens it recognises can slice the head down to a prefix.

## Reading TPT's logs

TPT is a second, independent producer of training logs, which is the most useful
thing one repository can be to another: it is the only test of whether this
tool's parsers work on somebody else's file format rather than on its own
fixtures. Its JSONL schema v2 has three properties that broke the parser here,
all fixed in 1.1.0 and all covered by a committed fixture (E6):

- every file opens with an environment header line carrying no `step`, and a
  resumed sweep writes several of them;
- non finite numbers are written as the strings `"NaN"`, `"Infinity"` and
  `"-Infinity"`;
- `null` means "not measured", never zero.

`triage ingest` reads a `health/healthy_steps.jsonl` shaped file directly. TPT's
`sweep_results.jsonl` is one row per configuration with no step axis, which is
`Outcomes` shaped and reads through `--outcomes`, joined on `config_key`.

The header line convention is worth stating in TPT's `docs/schema.md` as a
parser contract rather than as an implementation detail, and an issue saying so
is owed upstream: a consumer that infers the schema from the first record gets
`can_parse` true followed by a parse error, which is precisely the defect this
release fixed on its own side.

## What is deliberately not shared

- **No shared Ollama client library.** All three repositories talk to a local
  Ollama. This one's client is one `urllib` module against two documented
  endpoints, and a shared library would create a versioning
  relationship between three projects to save less code than the coordination
  costs (E7d). Out of scope, stated once, here.
- **No shared generator.** TPT has a 3 locale generator; `Autofill_audit` has a
  6 locale by 4 tier one. This repository's is a byte deterministic demo fixture
  whose noise knobs are recorded in `meta.json` so that a published number can
  be traced to the data that produced it. Three generators is the correct number
  when they exist for three different purposes.
- **No shared classifier.** TPT trains a transformer, `Autofill_audit` ships
  ONNX, and this one is a numpy softmax head that fits in seconds so the CI
  fixture stays a CI fixture.

## What was borrowed, with credit

The grounding pass in `triage/llm/summarizer.py` is adopted from TPT's
`tpt.triage` explainer (E7d): extract every number from a generated summary,
check each against a table computed in Python from the report's own numbers, and
delete any sentence containing a number the table does not support. The credit
is in the module docstring as well as here, because that is where somebody
reading the implementation will be.

## Version relationship

`Autofill_audit` pins this package by git tag in its dev extra. Once
`ml-experiment-triage` is on PyPI (E1) that pin becomes an ordinary version
range, which is the point of the packaging split: its core install is numpy and
scipy, so an auditor that wants a permutation test does not also install pandas,
plotly, jinja2 and tensorboard.
