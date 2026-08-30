"""The autofill vertical end to end, in one command.

    python -m examples.demo_autofill
    python -m examples.demo_autofill --n-fields 4000 --seeds 5

**Run it as a module, not as a path**, for the same reason
`examples.demo_workflow` says so: running it by path puts `examples/` on
`sys.path` instead of the repository root and the first import fails.

The point of the chain is that everything after step four is the ordinary tool.
`triage ingest`, `triage compare` and `triage report` are handed the run
directories the autofill sweep and the evaluations wrote, and are given no
autofill specific flag at any point, which is acceptance criterion 1 of Section
5.6 demonstrated rather than asserted.

**A database per question, and the tool insisted.** The sweep compares learning
rates on `val/macro_f1`; each evaluation compares an engine against the keyword
baseline on `eval/macro_f1`. Those are different questions with different
baselines, and one database would make each comparison list the other's
conditions as refusals for the honest reason that they share no metric.

The evaluations are also kept apart from each other, and that one was not a
choice: every evaluation scores the rules baseline beside whatever engine is
under study, so two of them write a run directory called
`rules_de_DE_val_seed0`, and `triage ingest` refuses to store two different
source paths under one run id (D4). It is right to. The two `rules` runs happen
to hold the same numbers here, and a store that quietly accepted them would
equally quietly accept two that did not.

**The LLM step skips rather than fails.** Step four asks a local Ollama to
classify the same split. With no Ollama, or with the models unpulled, it prints
why and the demo carries on; every other step is a pure function of the seeds
and writes the same bytes either way. That is the same discipline that keeps the
LLM summary out of the default report, and a demo that fell over on a machine
with no model would be a poor advertisement for it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from triage.cli import EXIT_OK
from triage.cli import main as triage

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "experiments" / "autofill_demo"

#: How many fields the demo generates. Smaller than the 4000 the published
#: numbers were measured at, because this exists to be run by somebody who has
#: just cloned the repository and it should finish while they are still looking
#: at it. `--n-fields 4000 --seeds 5` reproduces the documented table.
DEFAULT_FIELDS = 1200

#: The sweep condition everything else is read against: the lowest learning rate
#: with no L2, which is the first point of the grid.
SWEEP_BASELINE = "lr0.03_l20"

#: The engine every other engine is compared against, which is the keyword
#: baseline. Comparing the model against the rules is the question 5.6 asks.
ENGINE_BASELINE = "rules_de_DE_val"

#: Which locale the comparison is drawn on. de_DE, because that is where the
#: model earns its margin over the keyword baseline (5.6 criterion 2) and
#: therefore where a demo has something to show.
LOCALE = "de_DE"

#: How many eval rows the LLM step classifies. A 12B model on a consumer card
#: answers a few fields a second, so the whole split would be minutes of wall
#: clock in a demo. The evaluation is over exactly the rows that were classified
#: and records that it was.
LLM_LIMIT = 120


def step(number: int, title: str) -> None:
    print(f"\n=== {number}. {title} ===", flush=True)


def run(arguments: list[str]) -> int:
    print(f"$ triage {' '.join(arguments)}", flush=True)
    return triage(arguments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="demo-autofill",
        description="Generate, sweep, evaluate, ingest, compare and report the autofill vertical.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where everything goes")
    parser.add_argument("--n-fields", type=int, default=DEFAULT_FIELDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds", type=int, default=3, help="how many of the five sweep seeds to train"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="do not attempt the local model step at all"
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    corpus = out / "corpus"
    runs = out / "runs"
    sweep_database = out / "sweep.db"
    engine_database = out / "engines.db"
    llm_database = out / "llm.db"
    cache = out / "llm_cache.db"
    started = time.perf_counter()

    step(1, "generate the synthetic corpus")
    code = run(
        ["autofill", "generate", "--out", str(corpus),
         "--n-fields", str(args.n_fields), "--seed", str(args.seed), "--quiet"]
    )  # fmt: skip
    if code != EXIT_OK:
        return code

    step(2, "sweep the grid")
    code = run(
        ["autofill", "sweep", "--data", str(corpus), "--out", str(runs),
         "--seeds", str(args.seeds), "--quiet"]
    )  # fmt: skip
    if code != EXIT_OK:
        return code

    step(3, f"evaluate the model against the heuristic on {LOCALE}")
    code = run(
        ["autofill", "evaluate", "--data", str(corpus), "--out", str(out / "eval"),
         "--policy", "model", "--weights", str(runs / "best.npz"), "--locale", LOCALE,
         "--seed", str(args.seed), "--quiet"]
    )  # fmt: skip
    if code != EXIT_OK:
        return code

    step(4, "evaluate the local model on the same split")
    if args.skip_llm:
        print("skipped: --skip-llm was given")
    else:
        annotate = [
            "llm", "annotate", "--data", str(corpus), "--out", str(out / "llm_eval"),
            "--database", str(cache), "--locale", LOCALE, "--limit", str(LLM_LIMIT),
            "--seed", str(args.seed), "--quiet",
        ]  # fmt: skip
        # Its exit code is deliberately not propagated. The LLM condition is an
        # addition to this demo rather than a step it rests on, and a machine
        # with no Ollama should still get the other six steps.
        if run(annotate) != EXIT_OK:
            print(
                "the local model step did not run, and nothing after it depends on it: "
                "start Ollama with `ollama serve` and pull the models named in docs/llm.md "
                "to include the LLM condition",
                file=sys.stderr,
            )

    step(5, "ingest every run directory, with no autofill specific flag")
    code = run(["ingest", str(runs), "--database", str(sweep_database), "--quiet"])
    if code != EXIT_OK:
        return code
    ingested = []
    for source, target in ((out / "eval", engine_database), (out / "llm_eval", llm_database)):
        if (source / "runs").is_dir():
            code = run(["ingest", str(source / "runs"), "--database", str(target), "--quiet"])
            if code != EXIT_OK:
                return code
            ingested.append(target)

    step(6, "compare the sweep, then compare each engine against the rules")
    code = run(
        ["compare", "--database", str(sweep_database), "--baseline", SWEEP_BASELINE, "--quiet"]
    )
    if code != EXIT_OK:
        return code
    for target in ingested:
        code = run(["compare", "--database", str(target), "--baseline", ENGINE_BASELINE, "--quiet"])
        if code != EXIT_OK:
            return code

    step(7, "write the HTML report, with the autofill section attached")
    code = run(
        ["report", "--database", str(sweep_database), "--baseline", SWEEP_BASELINE,
         "--autofill", str(out / "eval"), "--output", str(out / "autofill_report.html"),
         "--quiet"]
    )  # fmt: skip
    if code != EXIT_OK:
        return code

    print(f"\ndemo-autofill finished in {time.perf_counter() - started:.1f} s under {out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
