"""The demo grid: six conditions, five seeds, one directory `triage ingest` reads.

The grid is the one 5.5 names, `lr {0.03, 0.1, 0.3}` crossed with
`l2 {0, 1e-4}` over seeds zero to four, and it exists to be RANKED rather than
to find a good model: five seeds per condition is what makes the seed replicated
permutation mode available, which is the mode this whole tool is built around,
and two learning rates whose difference is smaller than the seed spread is
exactly the case a single run comparison gets wrong.

The corpus is read and hashed ONCE and the thirty runs share it. Featurising is
a per corpus cost, not a per run one, and paying it thirty times would be most of
the wall clock. That is also why the sweep keeps only the best head: thirty
weight files are tens of megabytes of scaffolding that the comparison layer never
opens, so `best.npz` is written beside `sweep.json` and the rest are discarded.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage.autofill.features import featurise
from triage.autofill.generator import load_split
from triage.autofill.model import TrainConfig, TrainResult, train, write_run_dir
from triage.progress import track

SWEEP_LEARNING_RATES: tuple[float, ...] = (0.03, 0.1, 0.3)
SWEEP_L2: tuple[float, ...] = (0.0, 1e-4)
SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: Every point of the grid, as `(learning_rate, l2, seed)`. Ordered seed
#: slowest so that a sweep stopped early still holds whole conditions.
SWEEP_GRID: tuple[tuple[float, float, int], ...] = tuple(
    (learning_rate, l2, seed)
    for seed in SWEEP_SEEDS
    for learning_rate in SWEEP_LEARNING_RATES
    for l2 in SWEEP_L2
)


@dataclass(frozen=True)
class SweepResult:
    """What the sweep wrote and how long it took."""

    root: Path
    runs: list[Path]
    seconds: float
    best_variant: str
    best_seed: int
    best_macro_f1: float
    weights: Path
    summary: Path


def locale_mix(records: list[Any]) -> str:
    """`en_US:0.52,de_DE:0.48`, recorded in every run's config.

    The mix is a property of the corpus rather than of the run, and a comparison
    across runs trained on different mixes is a comparison of two things at once,
    so it goes in the config where a reader of the report can see it.
    """
    if not records:
        return ""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.locale] = counts.get(record.locale, 0) + 1
    return ",".join(f"{name}:{count / len(records):.2f}" for name, count in sorted(counts.items()))


def sweep(
    data_dir: Path | str,
    out_dir: Path | str,
    epochs: int = TrainConfig.epochs,
    batch_size: int = TrainConfig.batch_size,
    n_seeds: int = len(SWEEP_SEEDS),
    show_progress: bool = True,
) -> SweepResult:
    """Train the whole grid into `out_dir`, one run directory per point.

    `n_seeds` and `epochs` are here so a smoke test can run a reduced grid; the
    defaults are the grid the specification names and the ones every number in
    `docs/autofill.md` was measured at.
    """
    root = Path(out_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    train_records = load_split(data_dir, "train")
    val_records = load_split(data_dir, "val")
    train_matrix = featurise(train_records)
    val_matrix = featurise(val_records)
    mix = locale_mix(train_records)

    plan = [point for point in SWEEP_GRID if point[2] < n_seeds]
    started = time.perf_counter()
    written: list[Path] = []
    best: tuple[float, TrainResult, TrainConfig] | None = None

    for learning_rate, l2, seed in track(
        plan,
        "train",
        enabled=show_progress,
        label=lambda point: f"lr{point[0]:g} l2{point[1]:g} seed{point[2]}",
    ):
        config = TrainConfig(
            learning_rate=learning_rate, l2=l2, seed=seed, epochs=epochs, batch_size=batch_size
        )
        result = train(train_matrix, val_matrix, config)
        written.append(
            write_run_dir(
                root / f"{config.variant}_seed{seed}",
                result,
                config,
                locale_mix=mix,
                extra_config={"n_train": result.n_train, "n_val": result.n_val},
                save_weights=False,
            )
        )
        score = result.history[-1]["val/macro_f1"]
        if best is None or score > best[0]:
            best = (score, result, config)

    seconds = time.perf_counter() - started
    assert best is not None  # the grid is never empty: n_seeds is at least one
    score, result, config = best
    weights = result.model.save(root / "best.npz")

    summary = root / "sweep.json"
    summary.write_text(
        json.dumps(
            {
                "n_runs": len(written),
                "seconds": seconds,
                "grid": {
                    "learning_rate": list(SWEEP_LEARNING_RATES),
                    "l2": list(SWEEP_L2),
                    "seeds": list(SWEEP_SEEDS[:n_seeds]),
                },
                "epochs": epochs,
                "batch_size": batch_size,
                "locale_mix": mix,
                "n_train": len(train_records),
                "n_val": len(val_records),
                "best": {
                    "variant": config.variant,
                    "seed": config.seed,
                    "val_macro_f1": score,
                    "weights": weights.name,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return SweepResult(
        root=root,
        runs=written,
        seconds=seconds,
        best_variant=config.variant,
        best_seed=config.seed,
        best_macro_f1=score,
        weights=weights,
        summary=summary,
    )
