"""The local LLM and retrieval layer: Ollama over localhost, and nothing else.

Section 6. Everything in this package talks to an Ollama server on
`localhost:11434` over plain HTTP with `urllib.request`, and there is no code
path anywhere in it, or in its tests, or in CI, that reaches a hosted model.
That is a design constraint rather than a preference: the corpus this operates
on is an evaluation of somebody's experiments, and the tool should not be the
reason any of it leaves the machine.

Four capabilities, in the order they depend on each other:

* `ollama_client` speaks the four endpoints this needs and turns every failure
  into an exception that names what to type.
* `embeddings` caches vectors in the existing triage database, keyed by model
  name AND model digest, and searches them by exact brute force.
* `annotator` classifies form fields with retrieval augmented few shot prompts,
  which is the one place retrieval demonstrably earns its keep here.
* `summarizer` and `ask` generate prose and then DELETE any sentence whose
  numbers the report does not support.

Nothing here is imported by `triage/cli.py` at parse time, and nothing outside
this package imports it at module scope, so a core install that never asks for
an LLM never pays for one.
"""

from __future__ import annotations
