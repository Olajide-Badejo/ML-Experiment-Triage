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


def make_tpt_jsonl(series: dict[str, np.ndarray]) -> None:
    """A second producer's shape: an environment header line, then step rows.

    The PyTorch Performance and Health Toolkit writes its schema v2 logs this
    way, and a resumed sweep writes the header again partway through. Neither
    header carries a step, so a parser that infers its schema from the first
    record alone claims the file and then fails on it. Both headers must be
    skipped and counted, and the eight step rows must ingest cleanly.
    """
    directory = FIXTURES / "tpt_jsonl" / "healthy_steps"
    write_config(directory)
    header = {
        "type": "environment",
        "run_id": "tpt-healthy-0",
        "row_schema_version": 2,
        "torch_version": "2.9.0",
        "device": "cuda:0",
    }
    lines = [json.dumps(header)]
    for index in range(8):
        if index == 4:
            lines.append(json.dumps(header))
        lines.append(json.dumps({"step": index, "loss": float(series["train/loss"][index])}))
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


#: The field types the outcomes fixture classifies, spelled as the WHATWG
#: autocomplete token values that `Autofill_audit`'s taxonomy uses, so the two
#: repositories agree on the strings without either importing the other (E7b).
LABELS = ("postal-code", "email", "tel", "given-name", "cc-number")

#: How many form templates the outcomes fixture spreads its fields over. The
#: clustered paired test's smallest attainable p value is 2 / 2**n_clusters, so
#: six templates can reach 0.031 and no lower: small enough to enumerate
#: exhaustively, large enough for the result to clear alpha 0.05 at all.
N_TEMPLATES = 6
FIELDS_PER_TEMPLATE = 4


def make_outcomes_jsonl() -> None:
    """`Autofill_audit`'s `run.jsonl`, in their filed schema exactly (E5).

    One row per classified field per engine, no step anywhere. The keys and
    their spelling are copied from the schema in their issue and must not drift:
    this fixture is the contract test for the shape they hand us, and a rename
    here would pass while their file stopped parsing.

    The two engines score the SAME fields, which is what makes the comparison a
    paired one, and the fields are clustered by `template_id`, which is what
    makes the pairs non independent: fields on one form template share their
    markup, their locale and their author. The rules engine is given a real
    disadvantage on two of the six templates rather than a uniform one, so the
    cluster structure matters to the answer instead of being decoration.
    """
    rng = np.random.default_rng(SEED)
    directory = FIXTURES / "outcomes" / "autofill_run"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for template in range(N_TEMPLATES):
        # Two templates the rule engine reads badly, four it reads about as well
        # as the model: a per cluster shift, which is the thing the clustered
        # swap exists to respect.
        rules_rate = 0.45 if template < 2 else 0.86
        ngram_rate = 0.90
        for field_number in range(FIELDS_PER_TEMPLATE):
            label = LABELS[(template + field_number) % len(LABELS)]
            form_id = f"form-{template:02d}-{field_number:02d}"
            for engine, rate in (("rules", rules_rate), ("ngram", ngram_rate)):
                correct = bool(rng.random() < rate)
                rows.append(
                    {
                        "schema_version": "1.0.0",
                        "run_id": "autofill-eval-0",
                        "engine": engine,
                        "split": "test",
                        "form_id": form_id,
                        "template_id": f"template-{template:02d}",
                        "true_label": label,
                        "pred_label": label if correct else "off",
                        "correct": correct,
                        "confidence": round(float(rng.uniform(0.55, 0.99)), 4),
                        "latency_us": round(float(rng.uniform(120.0, 900.0)), 1),
                    }
                )
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (directory / "run.jsonl").write_text(text, encoding="utf-8")


def make_tpt_sweep_jsonl() -> None:
    """TPT's `sweep_results.jsonl`: one row per configuration, no step (E6 tail).

    The throughput sweeper writes a row per config rather than a series, so it
    is outcomes shaped and not a training log. It declares no `schema_version`
    on its result rows, which is exactly why this file needs `--outcomes` to be
    read: the strict recognition that claims the autofill file above refuses
    this one rather than guessing at an undeclared format.

    Two runners are measured on the same configurations, joined on `config_key`.
    """
    rng = np.random.default_rng(SEED + 1)
    directory = FIXTURES / "tpt_sweep" / "sweep_results"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_size in (8, 16, 32, 64):
        for workers in (2, 4):
            key = f"bs{batch_size}_w{workers}"
            base = 220.0 * np.log2(batch_size) + 40.0 * workers
            for runner, gain in (("eager", 1.0), ("compiled", 1.18)):
                rows.append(
                    {
                        "config_key": key,
                        "runner": runner,
                        "batch_size": batch_size,
                        "num_workers": workers,
                        "throughput_samples_per_s": round(
                            float(base * gain + rng.normal(0.0, 12.0)), 3
                        ),
                        "peak_memory_mb": round(float(180.0 * batch_size / 8 + workers * 15), 1),
                        "status": "ok",
                    }
                )
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    (directory / "sweep_results.jsonl").write_text(text, encoding="utf-8")


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
    make_tpt_jsonl(series)
    make_outcomes_jsonl()
    make_tpt_sweep_jsonl()
    print(f"fixtures written under {FIXTURES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
