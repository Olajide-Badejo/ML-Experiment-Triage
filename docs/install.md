# Install

```bash
pip install ml-experiment-triage
```

That is the whole install for the statistics API. Python 3.12 or later.

**Until the PyPI registration completes that line 404s, and so does every other
`pip install ml-experiment-triage` on this page.** Install from the tag in the
meantime, which resolves the same dependencies and gives you the same `triage`
command. Any extra named below works in the same position:

```bash
pip install "ml-experiment-triage @ git+https://github.com/Olajide-Badejo/ML-Experiment-Triage.git@v1.1.0"
pip install "ml-experiment-triage[cli] @ git+https://github.com/Olajide-Badejo/ML-Experiment-Triage.git@v1.1.0"
```

This paragraph and its code block go away the day the package is on PyPI.

## What the core install is, and why it is small

The core dependencies are **numpy and scipy, and nothing else**. That is a
deliberate split (E1), filed as an issue by the one downstream project that
depends on this one: it uses the statistics layer over rows of evaluation
outcomes, and pandas, tensorboard, plotly and jinja2 would have been most of the
install and none of the value for that use.

So a core install gives you:

- the whole `triage` public API for comparison, flagging, ranking and
  sensitivity;
- the JSONL and MLflow parsers, which need only the standard library;
- `paired_permutation`, `Outcomes` and the rest of the consumer surface.

Everything else is opt in.

## Extras

| Extra | Installs | For |
| --- | --- | --- |
| *(none)* | numpy, scipy | the statistics API, JSONL and MLflow ingest |
| `parsers` | pandas, tensorboard, tqdm | CSV in either shape, TensorBoard event files, the progress bar |
| `report` | plotly, jinja2 | the self contained HTML report |
| `cli` | `parsers` plus `report` | the `triage` command, doing everything it claims |
| `agentic` | choreographer | the browser driven form fill demo |
| `all` | `cli` plus `agentic` | everything |

```bash
pip install "ml-experiment-triage[cli]"     # the command line tool
pip install "ml-experiment-triage[all]"     # and the browser demo
```

A module that needs an extra imports it at the point of use and raises a message
naming the extra to install, rather than failing at import time with a
traceback about a package you never asked for. `tqdm` is the one exception that
is not an error at all: without it the progress bar degrades to plain lines,
which is what a log wants anyway.

## Verifying the install

```bash
triage --version
triage demo
```

`triage demo` synthesises a 31 run sweep in a temporary directory, ingests it,
compares every condition against the baseline and writes an HTML report, so a
fresh install can demonstrate itself with no clone and no data of your own. It
needs the `cli` extra, because it writes a report.

## Python and platform

- **Python 3.12 and 3.13.** Both are tested on ubuntu, windows and macos in CI.
- **CPU only.** Nothing here uses a GPU at any point, including the autofill
  model and the LLM layer's embeddings, which run against a local Ollama.
- **A TeX installation** is needed only to rebuild the two PDF reports, which
  are committed.

## Developing on the repository

The user path is `pip install`. Development and CI install the committed PEP 751
`pylock.toml`, which is the exact resolution every number this project publishes
was measured under.

```bash
git clone https://github.com/Olajide-Badejo/ML-Experiment-Triage
cd ML-Experiment-Triage
nox -s env        # a .venv from pylock.toml, plus this package editable
nox -s test       # the full suite
nox              # lint, typecheck and test: what CI runs first
```

`nox` is the canonical runner on every platform, and the Makefile is a thin
wrapper that forwards to it. `nox -l` lists every session. `pip install -r
pylock.toml` works too, on pip 25.1 and later.

To build this site locally:

```bash
nox -s docs-site           # mkdocs build --strict
nox -s docs-site -- serve  # live reload on localhost:8000
```

See `CONTRIBUTING.md` in the repository for the commit hooks, the inner loop and
what the build will refuse.
