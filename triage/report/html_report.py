"""One self contained HTML file: verdicts, curves, sensitivity, methodology.

Self contained is the requirement that shapes this module. The output is a
single file with the Plotly runtime inlined, so it survives being emailed,
dropped in a bucket, or opened from a laptop with no network. Nothing is
fetched at view time.

**On colour.** The palette is the validated reference categorical set, used
unchanged and in its documented slot order, which is certified for the adjacent
pair case that line charts fall under. The page commits to a single light
surface rather than following the viewer's theme. That is deliberate: this
report is evidence, it gets screenshotted into other documents, and it should
look identical everywhere it is opened. Identity never rests on colour alone,
because every series is in the legend and every number in the charts is also in
the verdict table below them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

from triage.analysis.comparison import (
    ComparisonConfig,
    ComparisonRefusal,
    group_by_variant,
    window_statistic,
)
from triage.analysis.regression import RegressionConfig, TriageReport
from triage.analysis.sensitivity import SensitivityResult
from triage.calibration import SUMMARY
from triage.core.experiment import Experiment

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Reference categorical palette, light mode, in documented slot order.
SERIES_COLOURS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
BASELINE_COLOUR = "#52514e"  # secondary ink: the baseline is a reference, not a series
SURFACE = "#fcfcfb"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MAX_PLOT_POINTS = 1200


@dataclass
class ReportContext:
    """Everything the template needs, assembled once and passed in whole."""

    title: str
    baseline: str
    generated_at: str
    database: str
    experiments: list[Experiment]
    triage: TriageReport
    sensitivity: list[SensitivityResult]
    comparison_config: ComparisonConfig
    regression_config: RegressionConfig
    calibration: dict[str, str]
    #: Comparisons the analysis declined to make, rendered as their own block.
    #: A report that shows only what could be computed is not a report of what
    #: was asked for.
    refusals: tuple[ComparisonRefusal, ...] = ()


def _rgba(hex_colour: str, alpha: float) -> str:
    hex_colour = hex_colour.lstrip("#")
    red, green, blue = (int(hex_colour[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _downsample(steps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin long curves so the file stays small without changing what is seen."""
    if steps.size <= MAX_PLOT_POINTS:
        return steps, values
    stride = int(np.ceil(steps.size / MAX_PLOT_POINTS))
    return steps[::stride], values[::stride]


def _variant_curve(
    runs: list[Experiment], tag: str, config: ComparisonConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean smoothed curve across seeds, with the min and max envelope.

    The envelope is the point of the chart. Plotting one line per condition
    hides the very thing this tool exists to account for, so the spread across
    seeds is drawn as a band and a reader can see immediately whether two
    conditions are separated by more than their own run to run noise.
    """
    usable = [run for run in runs if run.has(tag)]
    reference = max(usable, key=lambda run: len(run.series(tag))).series(tag)
    grid = reference.steps.astype(np.float64)

    curves = []
    for run in usable:
        series = run.series(tag)
        curves.append(
            np.interp(
                grid, series.steps.astype(np.float64), series.smoothed(config.smoothing_window)
            )
        )
    stacked = np.vstack(curves)
    return grid, stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0)


def build_metric_figure(
    experiments: list[Experiment],
    tag: str,
    baseline_key: str,
    config: ComparisonConfig,
) -> str:
    """Overlaid smoothed curves per condition, with the seed spread as a band."""
    variants = group_by_variant(experiments)
    figure = go.Figure()

    ordered = [baseline_key] + [key for key in variants if key != baseline_key]
    colour_index = 0
    for variant_key in ordered:
        runs = variants.get(variant_key, [])
        if not any(run.has(tag) for run in runs):
            continue
        is_baseline = variant_key == baseline_key
        if is_baseline:
            colour = BASELINE_COLOUR
        else:
            colour = SERIES_COLOURS[colour_index % len(SERIES_COLOURS)]
            colour_index += 1

        grid, mean, low, high = _variant_curve(runs, tag, config)
        seeds = sum(1 for run in runs if run.has(tag))

        if seeds > 1:
            band_x, band_low = _downsample(grid, low)
            _, band_high = _downsample(grid, high)
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([band_x, band_x[::-1]]),
                    y=np.concatenate([band_high, band_low[::-1]]),
                    fill="toself",
                    fillcolor=_rgba(colour, 0.13),
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{variant_key} range",
                )
            )

        line_x, line_y = _downsample(grid, mean)
        label = f"{variant_key}{' (baseline)' if is_baseline else ''}"
        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name=f"{label}, {seeds} seed{'s' if seeds != 1 else ''}",
                line={"color": colour, "width": 2, "dash": "dot" if is_baseline else "solid"},
                hovertemplate=(
                    f"<b>{label}</b><br>step %{{x:,.0f}}<br>{tag} %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    window_start = _window_start(experiments, tag, config)
    if window_start is not None:
        figure.add_vrect(
            x0=window_start,
            x1=_last_step(experiments, tag),
            fillcolor=_rgba(MUTED_INK, 0.10),
            line_width=0,
            annotation_text="final window",
            annotation_position="top left",
            annotation_font={"size": 11, "color": MUTED_INK},
        )

    figure.update_layout(
        template="none",
        height=380,
        margin={"l": 60, "r": 24, "t": 16, "b": 48},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": "#52514e"},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 12},
        },
    )
    axis_style = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": AXIS,
        "tickfont": {"color": MUTED_INK, "size": 11},
        "title_font": {"color": MUTED_INK, "size": 12},
    }
    figure.update_xaxes(title_text="training step", **axis_style)
    figure.update_yaxes(title_text=tag, **axis_style)

    # An explicit div id matters more than it looks. Left to itself Plotly mints
    # a fresh UUID per figure, which makes two runs over identical data produce
    # different files and quietly breaks the promise that a report is a pure
    # function of the database and the seed.
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=f"figure-{tag.replace('/', '-').replace(' ', '-')}",
        config={"displaylogo": False},
    )


def _window_start(
    experiments: list[Experiment], tag: str, config: ComparisonConfig
) -> float | None:
    candidates = [run.series(tag) for run in experiments if run.has(tag)]
    if not candidates:
        return None
    longest = max(candidates, key=len)
    size = longest.window_size(config.window_fraction, config.window_minimum)
    return float(longest.steps[-size])


def _last_step(experiments: list[Experiment], tag: str) -> float:
    return float(max(run.series(tag).steps[-1] for run in experiments if run.has(tag)))


def summarise_runs(experiments: list[Experiment], config: ComparisonConfig) -> list[dict[str, Any]]:
    """One row per condition for the run inventory table."""
    variants = group_by_variant(experiments)
    rows = []
    for variant_key, runs in variants.items():
        tags = sorted({tag for run in runs for tag in run.tags})
        row: dict[str, Any] = {
            "variant": variant_key,
            "n_seeds": len(runs),
            "run_ids": [run.run_id for run in runs],
            "formats": sorted({run.source_format for run in runs}),
            "statistics": {},
        }
        for tag in tags:
            values = [window_statistic(run.series(tag), config) for run in runs if run.has(tag)]
            if values:
                row["statistics"][tag] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
        rows.append(row)
    return rows


def render(context: ReportContext, output_path: str | Path) -> Path:
    """Write the single file report and return where it landed."""
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["signed"] = lambda value: f"{value:+.3f}"
    environment.filters["signed_pct"] = lambda value: f"{value:+.2f}"
    environment.filters["p"] = _format_p

    figures = {
        tag: build_metric_figure(
            context.experiments, tag, context.baseline, context.comparison_config
        )
        for tag in sorted({result.tag for result in context.triage.findings})
    }

    html = environment.get_template("report.html").render(
        context=context,
        figures=figures,
        runs=summarise_runs(context.experiments, context.comparison_config),
        plotly_js=get_plotlyjs(),
        n_runs=len(context.experiments),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def _format_p(value: float | None) -> str:
    """Small p values as a bound rather than a fake precision.

    `None` is a p value the tool declined to compute, which is a different thing
    from a large one and is printed as a refusal rather than as a number.
    """
    if value is None:
        return "not reported"
    if value < 0.0001:
        return "below 0.0001"
    return f"{value:.4f}"


def build_context(
    experiments: list[Experiment],
    triage: TriageReport,
    sensitivity: list[SensitivityResult],
    baseline: str,
    database: str,
    comparison_config: ComparisonConfig,
    regression_config: RegressionConfig,
    # The measured error rates, defaulted rather than required (D32). They live
    # in `triage.calibration`, which is the single source the README, the PDFs,
    # the figures and the CLI all read, and a caller assembling a context should
    # not have to know that in order to get the footer this tool is built around.
    calibration: dict[str, str] = SUMMARY,
    refusals: Sequence[ComparisonRefusal] = (),
    title: str = "ML Experiment Triage",
) -> ReportContext:
    return ReportContext(
        title=title,
        baseline=baseline,
        generated_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        database=database,
        experiments=experiments,
        triage=triage,
        sensitivity=sensitivity,
        comparison_config=comparison_config,
        regression_config=regression_config,
        calibration=calibration,
        refusals=tuple(refusals),
    )
