# Quickstart

Three verbs against your own runs, one directory per run:

```bash
triage ingest  path/to/runs --database triage.db
triage compare --database triage.db --baseline my_baseline_variant
triage report  --database triage.db --baseline my_baseline_variant --output report.html
```

`ingest` is the only verb that touches log files. `compare` and `report` read
SQLite, so they take milliseconds and can be rerun freely, and both are pure
functions of the database plus the recorded permutation seed. Re-ingesting an
unchanged source does no work, and an interrupted ingest loses at most the run
in flight.

If you have no runs of your own yet:

```bash
triage demo
```

## The layout it expects

```text
runs/
  lr0.001_bs32_seed0/
    config.json                       # {"learning_rate": 0.001, "batch_size": 32, "seed": 0}
    events.out.tfevents.1700000000.host
  lr0.001_bs32_seed1/
    config.json
    metrics.csv                       # step,train/loss,val/accuracy
  lr0.003_bs32_seed0/
    config.json
    metrics.jsonl                     # {"step": 0, "val/loss": 2.30}
```

Runs that differ only in their `seed` are grouped into one condition
automatically, which is what makes the strong comparison mode available. A run's
identity is its path relative to the ingest root, so two `seed0` directories
under different sweeps are two runs and not one.

## What comes out

```text
candidate            metric           change     p adj  verdict
---------------------------------------------------------------
lr0.0100_bs32        val/loss        +38.67%    0.0198  regression
lr0.0003_bs32        val/loss        +20.27%    0.0198  regression
lr0.0100_bs32        val/accuracy     -5.03%    0.0198  regression
lr0.0030_bs32        val/loss        -19.52%    0.0198  improvement
lr0.0030_bs128_seed0 val/loss        -16.64%    0.0001  improvement  [weaker mode]
lr0.0030_bs64        val/loss        -12.59%    0.0340  improvement
```

Every p value carries the mode it came from. A row marked `[weaker mode]` was
computed without seed replicates, which the [methodology](methodology.md) shows
is a much weaker claim, and the two modes are corrected in separate Benjamini
Hochberg families so neither inflates the other's denominator.

`triage report` writes the same content as a single self contained HTML file
with the seed spread drawn as a band around each curve.

## Exit codes

A CI gate needs to tell these cases apart, so they are five codes and not two.
`triage --help` prints the same table.

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | a bug in triage: an unexpected exception, with its traceback |
| 2 | a usage error, or a failure the tool foresaw, reported as `error: ...` |
| 3 | the work was done, but at least one run failed to parse |
| 4 | the run finished having performed zero comparisons |

Code 4 is the one worth wiring into a gate. Through 1.0.0 a `compare` that
compared nothing exited 0, so a build could go green because every comparison
had been refused.

## Useful flags

| Flag | What it does |
| --- | --- |
| `--fdr RATE` | the Benjamini Hochberg false discovery rate, 0.05 by default, and operative since 1.1.0 |
| `--practical-threshold-absolute X` | the practical gate in the metric's own units, for a metric whose baseline can be zero |
| `--higher-is-better TAG` / `--lower-is-better TAG` | state a metric's direction when its name does not imply one |
| `--outcomes` | read step free JSONL as cross sectional `Outcomes` rows rather than as a time series |
| `--quiet` | the count summary only; provenance lines stay |
| `--log-level LEVEL` | ingest diagnostics on stderr |

## Reading MLflow runs

Both of MLflow's backends are read directly, with no MLflow installed and no
extra needed:

```bash
triage ingest path/to/mlflow.db --database triage.db   # the SQLite backend
triage ingest path/to/mlruns    --database triage.db   # the file store
```

A tracking database holds many runs, so each becomes its own run named
`<database>/<run_uuid>`; a file store run keeps its directory path. MLflow params
become the run config, with numbers read as numbers, so an MLflow sweep reaches
the sensitivity ranking like any other.

The four SQLite tables read here (`runs`, `metrics`, `params`, `tags`) are
unchanged in these fields from MLflow 1.x through 3.x. The `mlruns/` file store
is read as MLflow froze it: that backend went into maintenance mode when SQLite
became the default in 3.7, December 2025.

## The other verbs

| Verb | What it is |
| --- | --- |
| `triage demo` | the synthetic sweep end to end, in a temporary directory |
| `triage autofill` | [this repository's reference workload](autofill.md): generate, train, sweep, evaluate, agentic |
| `triage llm` / `triage ask` | [the local model layer](llm.md): annotate, summarize, ask, against a local Ollama |

## Next

- [Library tutorial](library.md), to call the statistics from your own code.
- [Methodology](methodology.md), for what the p values mean and what they do not.
