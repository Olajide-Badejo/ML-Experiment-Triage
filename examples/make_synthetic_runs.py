"""Write the synthetic hyperparameter sweep to disk, in three log formats.

The sweep itself, its ground truth and its writers live in `triage.demo`, and
this script is the command line over them:

    python -m examples.make_synthetic_runs --output experiments/results/demo_sweep

The move was `triage demo`'s doing (D32). A tool that can only demonstrate
itself from a git checkout cannot demonstrate itself to somebody who ran
`pip install`, so the sweep had to become part of the package; keeping a second
copy here would have been two demos free to drift apart. The names below are re
exported because the integration tests and `examples/demo_workflow.py` import
them from this module, and moving code should not move an import line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from triage.demo import (
    ACCURACY_TAG,
    BASELINE,
    CONDITIONS,
    KNOWN_BEST,
    LOSS_TAG,
    N_SEEDS,
    N_STEPS,
    SWEEP_SEED,
    WRITERS,
    Condition,
    accuracy_spec,
    generate,
    loss_spec,
    write_config,
    write_csv,
    write_jsonl,
    write_tensorboard,
)

__all__ = [
    "ACCURACY_TAG",
    "BASELINE",
    "CONDITIONS",
    "KNOWN_BEST",
    "LOSS_TAG",
    "N_SEEDS",
    "N_STEPS",
    "SWEEP_SEED",
    "WRITERS",
    "Condition",
    "accuracy_spec",
    "generate",
    "loss_spec",
    "main",
    "write_config",
    "write_csv",
    "write_jsonl",
    "write_tensorboard",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="experiments/results/demo_sweep",
        help="directory to write the sweep into (replaced if it exists)",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress bar")
    args = parser.parse_args()

    root = Path(args.output)
    written = generate(root, show_progress=not args.quiet)

    print(f"wrote {len(written)} runs across {len(CONDITIONS)} conditions to {root}")
    for condition in CONDITIONS:
        print(
            f"  {condition.name:<16} {condition.log_format:<12} "
            f"{condition.n_seeds} seed(s)  loss effect {condition.loss_effect:+.3f}"
            f"  {condition.note}"
        )
    print(f"baseline: {BASELINE}; known best: {KNOWN_BEST}; sweep seed {SWEEP_SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
