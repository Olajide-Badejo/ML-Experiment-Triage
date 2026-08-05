"""Regenerate the committed parser fixtures under `tests/fixtures`.

The fixtures are committed rather than built during the test run, so that a
parser regression is caught against bytes that have not moved since the parser
was written. This script exists to rebuild them deliberately, for example after
a TensorBoard format change, and its output is fully determined by the seed
below.

Every fixture carries the same underlying series in a different container, so
the unit suite can assert that all three parsers agree on the numbers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SEED = 424242
N_STEPS = 60


def reference_series() -> dict[str, np.ndarray]:
    """The one series every fixture encodes, as float32 to match the model."""
    rng = np.random.default_rng(SEED)
    steps = np.arange(N_STEPS, dtype=np.int64)
    loss = (2.5 * np.exp(-steps / 18.0) + 0.20 + rng.normal(0, 0.02, N_STEPS)).astype(np.float32)
    accuracy = (0.95 - 0.8 * np.exp(-steps / 14.0) + rng.normal(0, 0.01, N_STEPS)).astype(
        np.float32
    )
    return {"steps": steps, "train/loss": loss, "val/accuracy": accuracy}


CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adamw",
    "seed": 0,
}


def write_config(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8")


def make_tensorboard(series: dict[str, np.ndarray]) -> None:
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    directory = FIXTURES / "tensorboard" / "tb_run"
    if directory.exists():
        shutil.rmtree(directory)
    write_config(directory)

    writer = EventFileWriter(str(directory))
    for index, step in enumerate(series["steps"]):
        for tag in ("train/loss", "val/accuracy"):
            summary = summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(series[tag][index]))]
            )
            writer.add_event(
                event_pb2.Event(step=int(step), wall_time=1700000000.0 + index, summary=summary)
            )
    writer.close()

    # The writer names the file after the host and process, which would make the
    # fixture machine specific. Rename it to something stable and committable.
    written = sorted(directory.glob("*tfevents*"))
    if len(written) != 1:
        raise RuntimeError(f"expected one event file, found {len(written)}")
    written[0].rename(directory / "events.out.tfevents.1700000000.fixture")


def make_csv_wide(series: dict[str, np.ndarray]) -> None:
    directory = FIXTURES / "csv_wide" / "csv_wide_run"
    write_config(directory)
    lines = ["step,wall_time,train/loss,val/accuracy"]
    for index, step in enumerate(series["steps"]):
        wall = 1700000000.0 + index
        lines.append(
            f"{step},{wall:.1f},{series['train/loss'][index]:.9g},"
            f"{series['val/accuracy'][index]:.9g}"
        )
    (directory / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_csv_long(series: dict[str, np.ndarray]) -> None:
    directory = FIXTURES / "csv_long" / "csv_long_run"
    write_config(directory)
    lines = ["step,tag,value"]
    for index, step in enumerate(series["steps"]):
        for tag in ("train/loss", "val/accuracy"):
            lines.append(f"{step},{tag},{series[tag][index]:.9g}")
    (directory / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_jsonl(series: dict[str, np.ndarray]) -> None:
    directory = FIXTURES / "jsonl" / "jsonl_run"
    write_config(directory)
    lines = []
    for index, step in enumerate(series["steps"]):
        lines.append(
            json.dumps(
                {
                    "step": int(step),
                    "wall_time": 1700000000.0 + index,
                    "train/loss": float(series["train/loss"][index]),
                    "val/accuracy": float(series["val/accuracy"][index]),
                }
            )
        )
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_truncated_jsonl(series: dict[str, np.ndarray]) -> None:
    """A JSONL log cut off mid line, which is what a killed job leaves behind."""
    directory = FIXTURES / "jsonl_truncated" / "killed_run"
    write_config(directory)
    lines = [
        json.dumps({"step": int(step), "train/loss": float(series["train/loss"][index])})
        for index, step in enumerate(series["steps"][:30])
    ]
    text = "\n".join(lines) + '\n{"step": 30, "train/lo'
    (directory / "metrics.jsonl").write_text(text, encoding="utf-8")


def main() -> int:
    series = reference_series()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.save(
        FIXTURES / "reference_series.npy", np.stack([series["train/loss"], series["val/accuracy"]])
    )
    make_tensorboard(series)
    make_csv_wide(series)
    make_csv_long(series)
    make_jsonl(series)
    make_truncated_jsonl(series)
    print(f"fixtures written under {FIXTURES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
