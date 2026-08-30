"""This repository's reference workload: browser form field classification.

Deciding that an input is a given name, a postal code or a card security code,
across locales whose address formats disagree, is the machine learning problem
behind browser autofill. It is here because a tool that ranks training runs
needs a training run of its own to rank, and because this particular problem
exercises every axis the tool advertises: a dataset with a definition, a sweep
with seeds, two conditions to compare, calibrated probabilities to check, and a
cross sectional evaluation output that is not a time series.

**What this is not.** It is not a competitor to `Autofill_audit`, which is the
production grade auditor with a 392 rule engine, nine locale vocabularies and a
calibrated ONNX model behind a Playwright driver; and it is not a competitor to
the PyTorch Performance Toolkit's transformer field classifier, which is the
production grade trainer. It is pure numpy, it trains in seconds, and its whole
job is to be a workload this repository can be honest about end to end. The
taxonomy is value compatible with both of them by construction (E7b), so a
label written here means the same thing there.

**Dependencies.** numpy, scipy and the standard library, and nothing else, so
this package imports on the core install that the statistics consumer has. No
torch, no onnx, no browser and no network anywhere in it.
"""

from __future__ import annotations
