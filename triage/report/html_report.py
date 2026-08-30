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

**plotly and jinja2 are imported on first use, not at module load** (E1). They
are about 5 MB of reporting stack, they are the `report` extra rather than the
core install, and `triage/__init__.py` resolves `build_context` and `render`
lazily for the same reason. Importing this module therefore costs nothing until
something actually draws; the first attribute access on either dependency
raises `MissingExtraError` naming the extra if it is absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from triage._extras import LazyModule
from triage.analysis.comparison import (
    ComparisonConfig,
    ComparisonRefusal,
    group_by_variant,
    window_statistic,
)
from triage.analysis.regression import RegressionConfig, TriageReport
from triage.analysis.sensitivity import SensitivityReport, SensitivityResult
from triage.calibration import SUMMARY
from triage.core.experiment import Experiment

if TYPE_CHECKING:
    import jinja2
    import plotly.graph_objects as go
    import plotly.offline as plotly_offline
else:
    _REPORT = "drawing the HTML report"
    go = LazyModule("plotly.graph_objects", extra="report", purpose=_REPORT)
    plotly_offline = LazyModule("plotly.offline", extra="report", purpose=_REPORT)
    jinja2 = LazyModule("jinja2", extra="report", purpose=_REPORT)

LOGGER = logging.getLogger("triage.report")

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: The size past which the report stops being the thing it is built to be. This
#: page exists to be emailed, dropped in a bucket and opened offline, and a
#: 14.5 MB attachment does none of those well. Past this the tool says so rather
#: than silently handing over something that will bounce.
LARGE_REPORT_BYTES = 10 * 1024 * 1024

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
#: Line styles the candidate series cycle through once the palette wraps. Eight
#: colours drawn solid means condition nine is pixel identical to condition one,
#: on the chart and in the legend swatch alike, so the dash advances with every
#: full turn of the palette. `dot` is absent on purpose: it is the baseline's,
#: and the baseline has to stay the one line a reader can find without counting.
SERIES_DASHES = ("solid", "dash", "longdash", "dashdot")
BASELINE_COLOUR = "#52514e"  # secondary ink: the baseline is a reference, not a series
SURFACE = "#fcfcfb"
#: Muted ink, mirrored into `--muted` in the template. Measured at 5.03:1 on
#: `SURFACE`, which clears WCAG AA for the normal size text it styles: table
#: headers, tile labels, captions, and every axis on every chart. The value it
#: replaced was 3.50:1, which does not.
MUTED_INK = "#6f6d68"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MAX_PLOT_POINTS = 1200

#: How many candidate conditions are drawn as their own line, over and above the
#: baseline. A 500 run sweep produced a 14.5 MB page carrying fifty overlaid
#: curves, which is not a chart and is not readable however the colours are
#: chosen. Twelve is two full turns of the eight slot palette minus a little
#: headroom, and it is the point past which a legend stops fitting on one line.
#: The conditions that do not fit are not dropped: they are drawn as one
#: envelope, and the legend says how many went into it.
MAX_PLOTTED_CONDITIONS = 12

#: How far seed lengths inside one condition may differ before the figure says
#: so. Runs never stop on exactly the same step, so a tolerance of nothing would
#: print the caveat on every honest sweep and teach the reader to ignore it.
RAGGED_TOLERANCE = 0.05

#: The caveat itself. Plain ASCII, because it is interpolated into the Plotly
#: JSON payload and read back out by a test.
RAGGED_NOTE = "seeds of one condition end at different steps; curves stop where their seeds do"

#: How the footer and the header spell the moment the report was built.
GENERATED_AT_FORMAT = "%Y-%m-%d %H:%M %Z"

#: The reproducible builds convention, honoured so that a caller who cannot pass
#: an argument, a Makefile or a CI job for instance, can still pin the stamp.
SOURCE_DATE_EPOCH = "SOURCE_DATE_EPOCH"

#: Everything a figure div id may contain. Metric tags arrive from CSV headers,
#: JSONL keys and TensorBoard tags, so they are third party data and are
#: whitelisted rather than blacklisted on the way into an HTML attribute.
UNSAFE_IN_SLUG = re.compile(r"[^A-Za-z0-9_-]")


def _slug(text: str) -> str:
    """A metric tag reduced to characters that cannot end an HTML attribute."""
    return UNSAFE_IN_SLUG.sub("-", text)


def resolve_generated_at(generated_at: str | datetime | None = None) -> str:
    """The moment to stamp on the report, in order of who gets to decide.

    An explicit argument first, then `SOURCE_DATE_EPOCH`, then the wall clock.
    Only the last of those is non reproducible, and it is the one a caller who
    cares about reproducibility never reaches.

    The environment value is read as UTC rather than as local time. The old
    footer used `datetime.now().astimezone()`, so it carried a locale dependent
    zone name and two machines building from one database disagreed about the
    contents of the file, which is precisely what a reproducibility gate exists
    to catch and precisely what the masking in that gate hid.
    """
    if generated_at is not None:
        if isinstance(generated_at, datetime):
            return generated_at.strftime(GENERATED_AT_FORMAT)
        return generated_at

    raw = os.environ.get(SOURCE_DATE_EPOCH)
    if raw:
        try:
            seconds = int(raw.strip())
        except ValueError:
            LOGGER.warning(
                "%s is set to %r, which is not an integer number of seconds; "
                "falling back to the wall clock, so this report will not be reproducible",
                SOURCE_DATE_EPOCH,
                raw,
            )
        else:
            return datetime.fromtimestamp(seconds, tz=UTC).strftime(GENERATED_AT_FORMAT)

    return datetime.now().astimezone().strftime(GENERATED_AT_FORMAT)


def _plotly_text(text: str) -> str:
    """Escape a label on its way into a Plotly `name` or `hovertemplate`.

    Plotly does not render these as plain text. It runs them through a converter
    that interprets a documented subset of HTML, so an unescaped `<` in a metric
    tag is markup inside the chart even though the JSON payload around it is
    escaped. Ampersand first, or the escaping would eat its own output.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;")


@dataclass(frozen=True)
class AutofillSection:
    """The reference workload's calibration numbers, as plain data.

    Plain data, and read from the JSON `triage autofill evaluate` wrote, so that
    the reporting layer imports nothing from `triage.autofill` and the numbers on
    the page are the ones that were measured rather than a second computation
    that could disagree with them. The file is the contract; this is its shape.
    """

    temperature: float
    ece_pre: float
    ece_post: float
    mce_pre: float
    mce_post: float
    n_bins: int
    n_rows: int
    #: `(mean confidence, accuracy, count)` per bin, before and after scaling.
    bins_pre: tuple[tuple[float, float, int], ...] = ()
    bins_post: tuple[tuple[float, float, int], ...] = ()
    #: One row per scored engine, from `evaluation.json`. Empty when only the
    #: calibration file was found, which is a legitimate half of the output.
    engines: tuple[dict[str, Any], ...] = ()
    locale: str = ""
    split: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "ece_pre": self.ece_pre,
            "ece_post": self.ece_post,
            "mce_pre": self.mce_pre,
            "mce_post": self.mce_post,
            "n_bins": self.n_bins,
            "n_rows": self.n_rows,
            "bins_pre": [list(item) for item in self.bins_pre],
            "bins_post": [list(item) for item in self.bins_post],
            "engines": [dict(row) for row in self.engines],
            "locale": self.locale,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutofillSection:
        return cls(
            temperature=float(data["temperature"]),
            ece_pre=float(data["ece_pre"]),
            ece_post=float(data["ece_post"]),
            mce_pre=float(data["mce_pre"]),
            mce_post=float(data["mce_post"]),
            n_bins=int(data["n_bins"]),
            n_rows=int(data["n_rows"]),
            bins_pre=tuple(_bin_triples(data.get("bins_pre", ()))),
            bins_post=tuple(_bin_triples(data.get("bins_post", ()))),
            engines=tuple(dict(row) for row in data.get("engines", ())),
            locale=str(data.get("locale", "")),
            split=str(data.get("split", "")),
        )


def _bin_triples(rows: Any) -> list[tuple[float, float, int]]:
    """Bins as `(confidence, accuracy, count)`, from either spelling.

    The calibration file writes them as objects, and `to_dict` writes them back
    as triples, so a section that has been round tripped through JSON has to read
    the same either way.
    """
    triples: list[tuple[float, float, int]] = []
    for row in rows:
        if isinstance(row, dict):
            triples.append((float(row["confidence"]), float(row["accuracy"]), int(row["count"])))
        else:
            confidence, accuracy, count = row
            triples.append((float(confidence), float(accuracy), int(count)))
    return triples


def load_autofill_section(path: str | Path) -> AutofillSection | None:
    """Read an evaluation output directory, or `None` when there is nothing there.

    An absence is not an error: `--autofill` pointed at a directory that holds no
    calibration file means the evaluation ran with the heuristic policy, which
    produces no probabilities to calibrate. A report that refused to render over
    that would be refusing over a legitimate result.
    """
    directory = Path(path)
    calibration_file = directory if directory.is_file() else directory / "calibration.json"
    if not calibration_file.exists():
        return None
    payload = json.loads(calibration_file.read_text(encoding="utf-8"))
    summary = dict(payload.get("calibration", payload))

    engines: tuple[dict[str, Any], ...] = ()
    locale = split = ""
    evaluation = calibration_file.parent / "evaluation.json"
    if evaluation.exists():
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        engines = tuple(dict(row) for _, row in sorted(report.get("engines", {}).items()))
        locale = str(report.get("locale", ""))
        split = str(report.get("split", ""))

    return AutofillSection.from_dict(
        {**summary, "engines": engines, "locale": locale, "split": split}
    )


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
    #: The reference workload's probability calibration, when the caller passed
    #: one. `None` renders nothing at all, so a report of an ordinary sweep is
    #: byte identical to what it was before this section existed.
    autofill: AutofillSection | None = None

    @property
    def sensitivity_caveat(self) -> str:
        """The caveat the analysis wrote, rather than a second copy of it.

        The template used to hand duplicate this prose, which made it two things
        to keep true and, worse, a fixed sentence where the real caveat carries
        the numbers: how few conditions the weakest correlation actually rests
        on. A template cannot know that and the analysis already does.
        """
        return SensitivityReport(results=list(self.sensitivity)).caveat()


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

    Outside its own step range a seed contributes nothing rather than its last
    value. `np.interp` clamps by default, so a run killed at step 200 of 1000
    used to be extended as a flat line to step 1000 and folded into both the
    mean and the envelope, inside the very window the statistic is taken over.
    The band there was drawn from a number no seed ever measured. Aggregating
    with the `nan` aware reductions makes each point an average of the seeds
    that actually reached it, and the band collapses onto the survivor rather
    than straddling it and a corpse.

    The grid comes from the longest run, so at least one seed is alive at every
    grid point and no column is empty.
    """
    usable = [run for run in runs if run.has(tag)]
    reference = max(usable, key=lambda run: len(run.series(tag))).series(tag)
    grid = reference.steps.astype(np.float64)

    curves = []
    for run in usable:
        series = run.series(tag)
        curves.append(
            np.interp(
                grid,
                series.steps.astype(np.float64),
                series.smoothed(config.smoothing_window),
                left=np.nan,
                right=np.nan,
            )
        )
    stacked = np.vstack(curves)
    return (
        grid,
        np.nanmean(stacked, axis=0),
        np.nanmin(stacked, axis=0),
        np.nanmax(stacked, axis=0),
    )


def _is_ragged(runs: list[Experiment], tag: str) -> bool:
    """Whether the seeds of one condition disagree about how long a run is."""
    lengths = [len(run.series(tag)) for run in runs if run.has(tag)]
    if len(lengths) < 2:
        return False
    return min(lengths) < (1.0 - RAGGED_TOLERANCE) * max(lengths)


def _prominence(triage: TriageReport | None, tag: str) -> dict[str, tuple[float, float]]:
    """How loudly each condition speaks on this metric, for choosing what to draw.

    Severity first, because it is the ranking the verdict table already uses and
    the thing a reader came for. Size of the effect second, because severity is
    defined as zero for everything that is not a regression, and a page that
    drew twelve arbitrary conditions out of fifty whenever the sweep contained
    no regression would be choosing by accident.
    """
    if triage is None:
        return {}
    scores: dict[str, tuple[float, float]] = {}
    for finding in triage.findings:
        if finding.result.tag != tag:
            continue
        candidate = finding.result.candidate
        score = (finding.severity, abs(finding.result.relative_effect_pct))
        scores[candidate] = max(scores.get(candidate, (0.0, 0.0)), score)
    return scores


def _plotted_conditions(
    variants: dict[str, list[Experiment]],
    tag: str,
    baseline_key: str,
    triage: TriageReport | None,
) -> tuple[list[str], list[str]]:
    """Split the conditions into the ones drawn as lines and the ones folded in.

    The baseline is never folded: it is the reference every other line is read
    against. Ties break on the condition name so that two runs over one database
    choose the same twelve.
    """
    present = [key for key, runs in variants.items() if any(run.has(tag) for run in runs)]
    candidates = [key for key in present if key != baseline_key]
    scores = _prominence(triage, tag)
    ranked = sorted(
        candidates, key=lambda key: (*(-value for value in scores.get(key, (0.0, 0.0))), key)
    )
    drawn = ranked[:MAX_PLOTTED_CONDITIONS]
    folded = ranked[MAX_PLOTTED_CONDITIONS:]
    ordered = ([baseline_key] if baseline_key in present else []) + [
        key for key in present if key != baseline_key and key in set(drawn)
    ]
    return ordered, sorted(folded)


def _folded_envelope(
    variants: dict[str, list[Experiment]],
    folded: list[str],
    tag: str,
    config: ComparisonConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The min and max across every condition that is not drawn individually."""
    curves = [_variant_curve(variants[key], tag, config) for key in folded]
    grid = max((curve[0] for curve in curves), key=len)
    lows, highs = [], []
    for own_grid, _mean, low, high in curves:
        lows.append(np.interp(grid, own_grid, low, left=np.nan, right=np.nan))
        highs.append(np.interp(grid, own_grid, high, left=np.nan, right=np.nan))
    return grid, np.nanmin(np.vstack(lows), axis=0), np.nanmax(np.vstack(highs), axis=0)


def build_metric_figure(
    experiments: list[Experiment],
    tag: str,
    baseline_key: str,
    config: ComparisonConfig,
    index: int = 0,
    triage: TriageReport | None = None,
) -> str:
    """Overlaid smoothed curves per condition, with the seed spread as a band.

    `index` is the position of this figure on the page. It prefixes the div id
    so that two tags whose slugs collide, `val/loss` and `val loss` for
    instance, still get distinct ids without the id having to carry any
    character from the tag that an HTML attribute cannot hold.

    `triage` supplies the severity ranking that decides which conditions are
    drawn as their own line when a sweep carries more of them than a chart can
    hold. Without it the choice falls back to the size of the effect, and with
    no findings at all to the condition name.
    """
    variants = group_by_variant(experiments)
    figure = go.Figure()
    safe_tag = _plotly_text(tag)

    ordered, folded = _plotted_conditions(variants, tag, baseline_key, triage)
    colour_index = 0
    ragged = False
    for variant_key in ordered:
        runs = variants.get(variant_key, [])
        if not any(run.has(tag) for run in runs):
            continue
        ragged = ragged or _is_ragged(runs, tag)
        is_baseline = variant_key == baseline_key
        if is_baseline:
            colour, dash = BASELINE_COLOUR, "dot"
        else:
            colour = SERIES_COLOURS[colour_index % len(SERIES_COLOURS)]
            dash = SERIES_DASHES[(colour_index // len(SERIES_COLOURS)) % len(SERIES_DASHES)]
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
                    name=f"{_plotly_text(variant_key)} range",
                )
            )

        line_x, line_y = _downsample(grid, mean)
        label = f"{_plotly_text(variant_key)}{' (baseline)' if is_baseline else ''}"
        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name=f"{label}, {seeds} seed{'s' if seeds != 1 else ''}",
                line={"color": colour, "width": 2, "dash": dash},
                hovertemplate=(
                    f"<b>{label}</b><br>step %{{x:,.0f}}<br>{safe_tag} %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    # Everything that did not fit, as one envelope. Dropping these conditions
    # would make the chart disagree with the table below it about how many
    # conditions the sweep has, so they are shown, in ink that does not compete
    # with the lines, with their count on the legend.
    if folded:
        grid, low, high = _folded_envelope(variants, folded, tag, config)
        band_x, band_low = _downsample(grid, low)
        _, band_high = _downsample(grid, high)
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([band_x, band_x[::-1]]),
                y=np.concatenate([band_high, band_low[::-1]]),
                fill="toself",
                fillcolor=_rgba(MUTED_INK, 0.10),
                line={"width": 0},
                hoverinfo="skip",
                showlegend=True,
                name=f"{len(folded)} further conditions, range",
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

    # A reader looking at a band that narrows to nothing deserves to be told
    # why rather than left to infer that the seeds converged.
    if ragged:
        figure.add_annotation(
            text=RAGGED_NOTE,
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.30,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font={"size": 11, "color": MUTED_INK},
        )

    figure.update_layout(
        template="none",
        height=380,
        margin={"l": 60, "r": 24, "t": 16, "b": 72 if ragged else 48},
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
    figure.update_yaxes(title_text=safe_tag, **axis_style)

    # An explicit div id matters more than it looks. Left to itself Plotly mints
    # a fresh UUID per figure, which makes two runs over identical data produce
    # different files and quietly breaks the promise that a report is a pure
    # function of the database and the seed. Plotly writes this id straight into
    # an HTML attribute and into a JavaScript string literal without escaping
    # either, so what goes in has to already be safe in both.
    # `str(...)` rather than a cast: plotly ships no type information, so the
    # return is Any, and this function's contract is that it hands back HTML.
    return str(
        figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=f"figure-{index}-{_slug(tag)}",
            config={"displaylogo": False},
        )
    )


def build_reliability_figure(section: AutofillSection) -> str:
    """A reliability diagram: predicted confidence against measured accuracy.

    The diagonal is perfect calibration, and it is drawn first so that the two
    curves read against it rather than against each other. A point ABOVE the
    diagonal is a model that is better than it claims; a point below is one that
    is worse, which is the direction that costs a decision layer money.

    The bins hold equal MASS rather than equal width (see
    `triage.autofill.calibration`), so the points are unevenly spaced along the
    x axis on purpose: that spacing is where the model's confidences actually
    live, and equal width bins would hide it by putting most of the data in one.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="perfect calibration",
            line={"color": AXIS, "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    for label, bins, colour in (
        ("before scaling", section.bins_pre, SERIES_COLOURS[1]),
        ("after scaling", section.bins_post, SERIES_COLOURS[0]),
    ):
        figure.add_trace(
            go.Scatter(
                x=[confidence for confidence, _, _ in bins],
                y=[accuracy for _, accuracy, _ in bins],
                mode="lines+markers",
                name=label,
                line={"color": colour, "width": 2},
                marker={"color": colour, "size": 7},
                customdata=[count for _, _, count in bins],
                hovertemplate=(
                    "confidence %{x:.3f}<br>accuracy %{y:.3f}<br>%{customdata} rows<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 60, "r": 20, "t": 10, "b": 50},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": MUTED_INK},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis={
            "title": "mean predicted confidence in the bin",
            "range": [0.0, 1.02],
            "gridcolor": GRID,
            "linecolor": AXIS,
        },
        yaxis={
            "title": "measured accuracy in the bin",
            "range": [0.0, 1.02],
            "gridcolor": GRID,
            "linecolor": AXIS,
        },
    )
    return str(
        figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id="figure-reliability",
            config={"displaylogo": False},
        )
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
    environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        # Unconditional rather than by file extension. Every value this template
        # interpolates is either third party data or derived from it, and an
        # autoescape rule that depends on what a template file is named is a
        # rule that silently stops applying when somebody renames one.
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["signed"] = lambda value: f"{value:+.3f}"
    environment.filters["signed_pct"] = lambda value: f"{value:+.2f}"
    environment.filters["p"] = _format_p

    # Keyed off what the database holds, not off what could be compared. Keying
    # figures and inventory columns off FINDINGS meant a metric nobody could
    # test against the baseline, which is precisely the case a reader most needs
    # to look at, silently had no curve and no column at all.
    tags = sorted({tag for run in context.experiments for tag in run.tags if run.has(tag)})
    compared = {finding.result.tag for finding in context.triage.findings}
    figures = {
        tag: build_metric_figure(
            context.experiments,
            tag,
            context.baseline,
            context.comparison_config,
            index,
            context.triage,
        )
        for index, tag in enumerate(tags)
    }

    reliability = (
        build_reliability_figure(context.autofill)
        if context.autofill is not None and context.autofill.bins_pre
        else ""
    )

    html = environment.get_template("report.html").render(
        context=context,
        figures=figures,
        compared=compared,
        runs=summarise_runs(context.experiments, context.comparison_config),
        reliability=reliability,
        # Fetched only when there is something for it to draw. `get_plotlyjs`
        # reads the bundle off disk, so this also keeps an empty report cheap
        # to build and not merely cheap to store.
        plotly_js=plotly_offline.get_plotlyjs() if figures or reliability else "",
        n_runs=len(context.experiments),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    size = output.stat().st_size
    if size > LARGE_REPORT_BYTES:
        LOGGER.warning(
            "report %s is %.1f MB, larger than the %.0f MB this format carries "
            "comfortably: consider reporting on fewer conditions or metrics at once",
            output,
            size / 1024 / 1024,
            LARGE_REPORT_BYTES / 1024 / 1024,
        )
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
    # Optional and defaulted off, so every existing caller renders exactly the
    # page it rendered before (D27: the report is a byte identical function of
    # the database, and a new section is exactly what quietly breaks that).
    autofill: AutofillSection | None = None,
    # The one input that is not the database. Left to the wall clock the report
    # cannot be a pure function of what it was built from, so a caller who needs
    # it to be says what moment to stamp; `SOURCE_DATE_EPOCH` does the same for
    # a caller that only has an environment to work with.
    generated_at: str | datetime | None = None,
) -> ReportContext:
    return ReportContext(
        title=title,
        baseline=baseline,
        generated_at=resolve_generated_at(generated_at),
        database=database,
        experiments=experiments,
        triage=triage,
        sensitivity=sensitivity,
        comparison_config=comparison_config,
        regression_config=regression_config,
        calibration=calibration,
        refusals=tuple(refusals),
        autofill=autofill,
    )
