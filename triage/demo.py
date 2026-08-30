"""The synthetic sweep this tool demonstrates itself on, and how it is written.

The sweep is fully determined by `SWEEP_SEED`, so the demo, the integration
tests and the committed reports all reproduce the same verdict table on any
machine. It is deliberately built to exercise every path the tool has:

* **Three formats in one sweep.** Some conditions log to TensorBoard event
  files, some to CSV, some to JSONL. The comparison layer cannot tell, which is
  the point of having one model behind three parsers.
* **A known best condition,** so an integration test can assert the tool finds
  the thing that is actually there.
* **A clear regression,** so the two gate rule has something to fire on.
* **A real but negligible difference,** below the practical threshold, so the
  practical gate has something to reject that the statistical gate accepts.
* **One condition with a single seed,** so the weaker window block mode appears
  in the report beside the strong one and is visibly labelled as weaker.

The ground truth is in `CONDITIONS` below, and nothing downstream reads it.

**This lives in the package rather than in `examples/` because `triage demo`
does.** A tool that can only demonstrate itself from a git checkout cannot
demonstrate itself to somebody who ran `pip install`, which is everybody who has
not already decided to trust it. `examples/make_synthetic_runs.py` is now a thin
script over this module, so the committed demo and the installed one are the
same code and cannot drift.
"""

from __future__ import annotations

import json
import shutil
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from triage.progress import track
from triage.synthetic import CurveSpec, generate_curve

SWEEP_SEED = 20260805
N_STEPS = 6000
N_SEEDS = 5

LOSS_TAG = "val/loss"
ACCURACY_TAG = "val/accuracy"


@dataclass(frozen=True)
class Condition:
    """One point of the sweep, with its ground truth effect on the loss."""

    name: str
    learning_rate: float
    batch_size: int
    loss_effect: float
    log_format: str
    n_seeds: int = N_SEEDS
    note: str = ""


# Effects are on the validation loss, where lower is better. The accuracy curve
# is generated with the opposite sign so the two metrics tell a consistent story
# and the false discovery correction has correlated metrics to work on, which is
# the situation it was chosen for.
CONDITIONS = (
    Condition("lr0.0010_bs32", 0.0010, 32, 0.000, "tensorboard", note="baseline"),
    Condition("lr0.0003_bs32", 0.0003, 32, +0.055, "tensorboard", note="undertrained, worse"),
    Condition("lr0.0030_bs32", 0.0030, 32, -0.060, "csv", note="known best"),
    Condition("lr0.0100_bs32", 0.0100, 32, +0.130, "csv", note="clear regression"),
    Condition("lr0.0010_bs64", 0.0010, 64, -0.004, "jsonl", note="real but negligible"),
    Condition("lr0.0030_bs64", 0.0030, 64, -0.042, "jsonl", note="better"),
    Condition(
        "lr0.0030_bs128",
        0.0030,
        128,
        -0.050,
        "csv",
        n_seeds=1,
        note="single seed, forces the weaker mode",
    ),
)

BASELINE = "lr0.0010_bs32"
KNOWN_BEST = "lr0.0030_bs32"


def loss_spec(condition: Condition) -> CurveSpec:
    return CurveSpec(
        n_steps=N_STEPS,
        floor=0.35,
        amplitude=2.0,
        decay=90.0,
        noise_sigma=0.030,
        rho=0.8,
        seed_sigma=0.020,
        effect=condition.loss_effect,
    )


def accuracy_spec(condition: Condition) -> CurveSpec:
    return CurveSpec(
        n_steps=N_STEPS,
        floor=0.91,
        amplitude=-0.75,
        decay=110.0,
        noise_sigma=0.010,
        rho=0.8,
        seed_sigma=0.007,
        effect=-condition.loss_effect * 0.30,
    )


def write_config(directory: Path, condition: Condition, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "variant": condition.name,
        "learning_rate": condition.learning_rate,
        "batch_size": condition.batch_size,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "seed": seed,
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def write_tensorboard(directory: Path, steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    writer = EventFileWriter(str(directory))
    for index, step in enumerate(steps):
        for tag, values in curves.items():
            summary = summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(values[index]))]
            )
            writer.add_event(
                event_pb2.Event(step=int(step), wall_time=1.7e9 + index, summary=summary)
            )
    writer.close()


def write_csv(directory: Path, steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    tags = list(curves)
    lines = ["step," + ",".join(tags)]
    lines.extend(
        f"{step}," + ",".join(f"{curves[tag][index]:.7g}" for tag in tags)
        for index, step in enumerate(steps)
    )
    (directory / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(directory: Path, steps: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    lines = [
        json.dumps({"step": int(step), **{tag: float(curves[tag][index]) for tag in curves}})
        for index, step in enumerate(steps)
    ]
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


WRITERS: dict[str, Callable[[Path, np.ndarray, dict[str, np.ndarray]], None]] = {
    "tensorboard": write_tensorboard,
    "csv": write_csv,
    "jsonl": write_jsonl,
}


def generate(root: Path, show_progress: bool = True) -> list[Path]:
    """Write the whole sweep, replacing anything already there."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    plan = [(condition, seed) for condition in CONDITIONS for seed in range(condition.n_seeds)]
    written: list[Path] = []
    steps = np.arange(N_STEPS, dtype=np.int64)

    for condition, seed in track(
        plan, "synthesise", enabled=show_progress, label=lambda pair: pair[0].name
    ):
        # One generator per run, keyed on the condition and the seed, so adding
        # a condition later cannot shift the numbers of the ones before it.
        # crc32 rather than hash(): Python randomises string hashing per
        # process, which would make the sweep irreproducible between runs.
        name_key = zlib.crc32(condition.name.encode("utf-8"))
        rng = np.random.default_rng([SWEEP_SEED, name_key, seed])
        curves = {
            LOSS_TAG: generate_curve(loss_spec(condition), rng),
            ACCURACY_TAG: np.clip(generate_curve(accuracy_spec(condition), rng), 0.0, 1.0),
        }

        directory = root / f"{condition.name}_seed{seed}"
        write_config(directory, condition, seed)
        WRITERS[condition.log_format](directory, steps, curves)
        written.append(directory)

    return written
