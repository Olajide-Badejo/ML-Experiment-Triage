# The local LLM and retrieval layer

Everything in `triage/llm/` talks to an Ollama server on `localhost:11434` over
plain HTTP with the standard library. There is no code path in the package, in
its tests, or in CI that reaches a hosted model, and none of the three verbs
falls back to one when the local service is absent: they say what to type and
stop.

Four capabilities, in the order they depend on each other.

| Module | What it does |
|---|---|
| `ollama_client.py` | Four endpoints over `urllib.request`, temperature 0, fixed seed |
| `embeddings.py` | Vectors cached in the triage database, searched by exact brute force |
| `annotator.py` | Retrieval augmented few shot field classification, with an ablation |
| `summarizer.py`, `ask.py` | Generated prose whose every number is verified or deleted |

## The models, and why these

Measured on the development machine: an RTX 5070 with 12,227 MiB of VRAM,
running Ollama 0.32.14.

| Role | Model | Size | Why |
|---|---|---|---|
| Chat, default | `mistral-nemo:12b-instruct-2407-q4_K_M` | 7.5 GB | Fully GPU resident at a 4096 to 8192 token context |
| Chat, fallback | `qwen2.5:7b-instruct-q4_K_M` | 4.7 GB | Faster; used automatically when the default is not pulled |
| Embedding, default | `embeddinggemma:300m` | 622 MB | Multilingual, which a de_DE corpus requires |
| Embedding, opt in | `bge-m3` | 1.2 GB | Larger quality tier; not pulled by anything here |

`nomic-embed-text` is rejected rather than merely unused: v1.5 is English only,
and half of this project's reference corpus is German. The Q5_K_M variant of the
12B chat model is on disk on the development machine and is deliberately not the
default: 8.7 GB plus a long context spills past 12 GB into CPU offload, which is
the failure mode this whole configuration exists to avoid.

    ollama pull mistral-nemo:12b-instruct-2407-q4_K_M
    ollama pull embeddinggemma:300m

## The prompt prefixes are not optional

`embeddinggemma` is trained asymmetrically. A query is embedded behind
`"task: search result | query: "` and a document behind `"title: none | text: "`,
each including its trailing space. Getting this wrong raises nothing at all: it
degrades retrieval quietly, by an amount nobody notices without an ablation.

So no call site in this package ever writes a prefix. Call sites name a model,
and the model's config record in `triage/llm/embeddings.py` knows what its own
prefixes are. `bge-m3` uses none, and that difference is data in the same table
rather than a branch somewhere else. A model with no record is refused rather
than embedded with empty prefixes, because empty is correct for one of the two
supported models and silently wrong for the other.

## Cache keys, and why the digest is in them

Two caches live in the ordinary triage database, in two tables added without a
schema version bump because they are additive:

* embeddings, keyed by `(model_name, model_digest, sha256(prefixed text))`;
* generated responses, keyed by `sha256(model + digest + prompt)`.

The digest is in both keys because an embedding is deterministic for one model
at one quantisation and is **not** deterministic across Ollama versions, driver
versions, or a repull of the same tag. A cache keyed on the name alone would
serve one model build's vectors to a query embedded by another, and the only
symptom would be retrieval getting worse.

The text is hashed after the prefix is applied, which is what keeps the query
and document embeddings of one string in two different rows: they are two
different inputs to the model.

The consequence a user sees is that a second `triage llm annotate` over the same
fields makes zero HTTP calls, an interrupted pass resumes, and a report built
twice from one database is the same report.

## No vector database, and the measurement behind that

The corpus is the labelled training split of the reference workload: 463 vectors
of 256 dimensions for de_DE at the demo corpus size, which is a 0.47 MB float32
matrix. One `matrix @ query` over it plus an `argpartition` is a single BLAS
call. Measured on the development machine:

| Step | Measured |
|---|---|
| Exact search, k = 8, over 463 x 256 | mean 15.5 us, median 14.9 us, p95 18.6 us (n = 2000) |
| Reading the whole cached corpus back out of SQLite | 1.7 ms |
| Embedding ONE query over HTTP (cache miss) | 2,233 ms |
| One chat completion, 12B at Q4_K_M | about 5 s |

The search is 15 microseconds. The HTTP round trip that produces the query
vector it searches with is 2.2 seconds, about 144,000 times longer, and the
generation that consumes the result is longer still. Nothing an index could do
to the 15 microseconds is observable.

An approximate index would add a dependency, a build step, a recall parameter
and a second thing that can be stale, in exchange for being slower at this size
and no longer exact. That is the whole justification, and it is written down
because a reader is entitled to know that the absence of a vector database here
is a decision rather than an omission.

## What the annotator does

For each field: render its signals to one line, embed that line, retrieve the k
nearest labelled training fields (default 8), build a few shot prompt with the
nearest example last, and ask for exactly `{"field_type": "..."}`. The parse is
strict; a failure is retried once with a corrective instruction; a second
failure is recorded as `unknown` and counted.

The confidence attached to an answer is a TIER, not a probability. A chat model
emits a token, not a posterior. First pass parses get 0.80, retried ones 0.55,
`unknown` 0.20, and the docstring says so, because calling that a calibrated
probability is exactly the dishonesty `triage/autofill/calibration.py` argues
against.

### The ablation: does retrieval actually help?

`triage llm annotate --ablation` scores the same de_DE split twice, zero shot
and retrieval augmented, writes both as ordinary triage run directories, ingests
them and compares them with the same seed replicated permutation test as every
other claim this tool makes. The lift is a verdict with a p value rather than a
sentence.

Measured on the development machine:

| Arm | de_DE val macro F1 | Requests | Wall clock |
|---|---|---|---|
| Zero shot | 0.8007 | 90 | 221.1 s |
| Retrieval augmented, k = 8 | 0.8879 | 0 (90 cache hits) | 0.0 s |

**Verdict: improvement, p 0.0079, adjusted p 0.0212**, over 5 bootstrap
replicates on 90 de_DE validation fields, `mistral-nemo:12b-instruct-2407-q4_K_M`
with `embeddinggemma:300m` retrieval. The lift is +0.0872 macro F1, or +10.9
percent relative. That verdict came out of `compare_all` on run directories the
ablation wrote, not out of arithmetic in this package.

The retrieval arm cost nothing on this run because a previous `triage llm
annotate` had already asked those exact questions, which is the response cache
doing its job: 90 cache hits, zero HTTP calls, identical answers.

### Measured against the other two engines

The same 90 row de_DE validation split, scored by all three engines:

| Engine | Accuracy | Macro F1 | ECE | Per field |
|---|---|---|---|---|
| Keyword rules | 0.7444 | 0.7420 | 0.1833 | microseconds |
| n gram model | 0.9333 | 0.9381 | 0.0512 | microseconds |
| Local LLM, kNN k = 8 | 0.9000 | 0.8879 | 0.1400 | 5.04 s |

Both learned engines beat the keyword baseline with a seed replicated verdict at
p 0.0079: the n gram model by +34.73 percent macro F1 and the LLM by +24.68
percent. The pure numpy classifier that trains in seconds is ahead of the 12B
model on this task and answers about a million times faster, which is the
comparison worth having and is the reason this repository ships it as the
reference workload rather than the LLM.

Wall clock: 90 fields in 459.1 s, 90 requests, 0 retried, 0 unparsed, so 5.1
seconds a field on a cold cache. Every one of those requests was a cache miss;
the second pass over the same fields makes zero HTTP calls and takes 0.0 s.

## Grounded generation

`summarizer.py` renders the whole `TriageReport` into the prompt, with no
retrieval: a few dozen findings fit in a 4096 token context, and retrieving
parts of them would add a silent omission failure mode this cannot otherwise
have. It then runs a GROUNDING PASS, a pattern adopted from the PyTorch
Performance Toolkit's `tpt.triage` explainer with credit (E7d): extract every
number from the generated text, verify each against a Python computed table of
the report's own numbers, and DELETE any sentence containing a number the table
does not support, recording how many were dropped.

Deletion rather than a warning, because a summary is read by somebody who is not
going to check it, which is the entire reason it exists. A hallucinated p value
with a caveat attached is still a hallucinated p value.

The check is tolerant to rounding in the direction a writer rounds and in no
other: a number matches when it agrees with an entry to the precision it was
WRITTEN at, and a probability may be spelled as a percentage. It will not accept
a number that is merely close, because 0.041 and 0.049 sit on either side of a
gate.

`triage report --llm-summary` embeds the result under "Automated summary (local
LLM: model name)". The flag is OFF by default and that is a correctness
property: without it the page is a byte identical function of the database,
which is what the reproducibility gate checks. With it, determinism holds
through the response cache.

## `triage llm ask`

Retrieval runs over the PROSE corpus only, the Markdown documents, the README
and the engineering log, chunked at headings. Numbers are never retrieved: every
figure in an answer is computed from the database by a SQL template chosen by
intent, and the answer is then checked against exactly those figures.

Four intents are answerable: verdicts, refusals, sensitivity, and what the
database holds. The first three need a `--baseline`, for the same reason
`triage compare` does. A question outside them is REFUSED, naming what can be
answered instead, and that refusal is the feature: a question answering box that
guesses when it does not know is worse than no box.

Both spellings work, because the specification uses both:

    triage llm ask "which conditions regressed?" --baseline lr0.0010_bs32
    triage ask "which conditions regressed?" --baseline lr0.0010_bs32

## Testing

Every test in CI runs against an injected fake transport with canned chat,
embedding and tags responses, including malformed ones: a body that is not JSON,
a chat turn answering prose where JSON was demanded, a token outside the
taxonomy, an embedding batch that comes back the wrong length, and a poisoned
summary that fabricates numbers. Nothing in CI opens a socket.

One file, `tests/integration/test_ollama_service.py`, is marked `ollama` and
talks to the real service. It skips itself automatically when `health()` fails,
so it disappears on any machine without a local Ollama. Run it deliberately with

    .venv/Scripts/python.exe -m pytest -m ollama -v -s

The result of the last such run, including the co residency measurement Section
2 left open, is recorded in `docs/ENGINEERING_LOG.md`.
