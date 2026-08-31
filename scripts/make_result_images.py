"""Render the result charts shown on the repository landing page.

Two variants of every chart are written, one for a light page and one for a
dark one, so the README reads correctly whichever theme a viewer has. The
palette is the validated reference categorical set: its light steps on the light
surface and its dark steps on the dark one, which is a selected pairing rather
than an automatic inversion.

This script needs `kaleido` for static export, which is a development extra
rather than a runtime dependency, so it is not part of `make all`. The images it
produces are committed. Regenerate them deliberately with `make images`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from triage.analysis.comparison import ComparisonConfig, group_by_variant
from triage.calibration import NOMINAL_ALPHA, WEAK_MODE_COST
from triage.core.store import Store

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "assets" / "images"
DEFAULT_DATABASE = ROOT / "experiments" / "demo" / "triage.db"
BASELINE = "lr0.0010_bs32"

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
MAX_POINTS = 700


@dataclass(frozen=True)
class Theme:
    """Surfaces, ink and series colours for one rendering mode."""

    name: str
    surface: str
    ink: str
    muted: str
    grid: str
    axis: str
    baseline: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    baseline="#52514e",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    baseline="#c3c2b7",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"),
)

THEMES = (LIGHT, DARK)


def rgba(colour: str, alpha: float) -> str:
    colour = colour.lstrip("#")
    red, green, blue = (int(colour[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def style_axes(figure: go.Figure, theme: Theme, x_title: str, y_title: str) -> None:
    common = {
        "showgrid": True,
        "gridcolor": theme.grid,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": theme.axis,
        "tickfont": {"color": theme.muted, "size": 13},
        "title_font": {"color": theme.muted, "size": 14},
    }
    figure.update_xaxes(title_text=x_title, **common)
    figure.update_yaxes(title_text=y_title, **common)
    figure.update_layout(
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        font={"family": FONT, "color": theme.ink, "size": 13},
        margin={"l": 70, "r": 28, "t": 56, "b": 56},
    )


#: Export scale. The four curve charts came out at 630 to 650 KB each at scale
#: 2, which put 17.5 MB of binary churn into an eleven commit history for
#: figures a README displays at about 880 pixels wide. At scale 1 they are 1180
#: pixels across, still wider than they are ever drawn, and about a fifth of the
#: bytes. Sharpness on a high density display is unaffected in practice because
#: the image is downscaled to fit the column either way.
EXPORT_SCALE = 1


def save(figure: go.Figure, stem: str, theme: Theme, width: int, height: int) -> Path:
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = IMAGES / f"{stem}-{theme.name}.png"
    figure.write_image(str(path), width=width, height=height, scale=EXPORT_SCALE)
    return optimise(path)


def optimise(path: Path) -> Path:
    """Rewrite a PNG with maximum lossless compression.

    Plotly writes through kaleido at the default compression level. Re encoding
    the same pixels at level 9 with the filter search on costs a second per
    image and takes another fifth off, and every byte here is committed.
    """
    from PIL import Image

    with Image.open(path) as image:
        pixels = image.copy()
    pixels.save(path, format="PNG", optimize=True, compress_level=9)
    return path


# ------------------------------------------------------------- curve figures


def variant_band(runs: list, tag: str, smoothing: int):
    """Mean, minimum and maximum across seeds of the smoothed curve.

    `smoothing` is a display choice and is wider here than the nine points the
    statistic uses. Lightly smoothed, the ribbon is dominated by within run
    measurement noise; smoothed harder, what remains is the seed to seed spread,
    which is the quantity the chart exists to put beside the differences between
    conditions. The captions say the figure is smoothed for display.
    """
    usable = [run for run in runs if run.has(tag)]
    reference = max(usable, key=lambda run: len(run.series(tag))).series(tag)
    grid = reference.steps.astype(np.float64)
    stacked = np.vstack(
        [
            np.interp(
                grid,
                run.series(tag).steps.astype(np.float64),
                run.series(tag).smoothed(smoothing),
            )
            for run in usable
        ]
    )
    stride = max(1, int(np.ceil(grid.size / MAX_POINTS)))
    return (
        grid[::stride],
        stacked.mean(axis=0)[::stride],
        stacked.min(axis=0)[::stride],
        stacked.max(axis=0)[::stride],
        len(usable),
    )


def curve_figure(
    experiments: list,
    tag: str,
    theme: Theme,
    config: ComparisonConfig,
    y_title: str,
    x_start: int = 600,
    display_smoothing: int = 101,
) -> go.Figure:
    """Overlaid smoothed curves, with the spread across seeds drawn as a band.

    The band is the whole point of the chart. One line per condition hides the
    run to run variance that this project exists to account for; with the band
    drawn, two conditions that are not actually separated look like it.

    The view starts after the initial transient. Drawn from step zero, the drop
    from 2.5 to 0.35 owns the whole vertical range and squashes every band into
    a few pixels at the bottom, which hides the one thing the chart is for. The
    comparison is decided in the converged region, so that is what is shown, and
    the axis says so.
    """
    figure = go.Figure()
    variants = group_by_variant(experiments)
    order = [BASELINE] + [key for key in variants if key != BASELINE]

    span: list[tuple[float, float]] = []
    index = 0
    for key in order:
        runs = variants.get(key, [])
        if not any(run.has(tag) for run in runs):
            continue
        is_baseline = key == BASELINE
        colour = theme.baseline if is_baseline else theme.series[index % len(theme.series)]
        if not is_baseline:
            index += 1

        steps, mean, low, high, seeds = variant_band(runs, tag, display_smoothing)
        keep = steps >= x_start
        steps, mean, low, high = steps[keep], mean[keep], low[keep], high[keep]
        span.append((low.min(), high.max()))
        if seeds > 1:
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate([high, low[::-1]]),
                    fill="toself",
                    fillcolor=rgba(colour, 0.20),
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                name=f"{key}{' (baseline)' if is_baseline else ''}",
                line={
                    "color": colour,
                    "width": 2.5,
                    "dash": "dash" if is_baseline else "solid",
                },
            )
        )

    longest = max((run.series(tag) for run in experiments if run.has(tag)), key=len)
    window = longest.window_size(config.window_fraction, config.window_minimum)
    figure.add_vrect(
        x0=float(longest.steps[-window]),
        x1=float(longest.steps[-1]),
        fillcolor=rgba(theme.muted, 0.14),
        line_width=0,
        annotation_text="final window",
        annotation_position="top left",
        annotation_font={"size": 12, "color": theme.muted},
    )

    style_axes(
        figure, theme, f"training step (from {x_start:,}, after the initial descent)", y_title
    )
    low_edge = min(pair[0] for pair in span)
    high_edge = max(pair[1] for pair in span)
    padding = 0.12 * (high_edge - low_edge)
    figure.update_layout(
        xaxis={"range": [x_start, float(longest.steps[-1])]},
        yaxis={"range": [low_edge - padding, high_edge + padding]},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 12},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return figure


# -------------------------------------------------------- calibration figure


def weak_mode_figure(theme: Theme) -> go.Figure:
    """The headline result: what comparing single runs costs.

    Both modes on data with a true effect of exactly zero. Every rejection is a
    false positive, so the nominal 5 percent line is where an honest test sits.
    """
    sigmas = [f"{sigma:.2f}" for sigma, _, _ in WEAK_MODE_COST]
    weak = [value * 100 for _, value, _ in WEAK_MODE_COST]
    strong = [value * 100 for _, _, value in WEAK_MODE_COST]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=sigmas,
            y=weak,
            name="Single run comparison",
            marker_color=theme.series[1],
            text=[f"{value:.1f}%" for value in weak],
            textposition="outside",
            textfont={"color": theme.ink, "size": 14},
            cliponaxis=False,
        )
    )
    figure.add_trace(
        go.Bar(
            x=sigmas,
            y=strong,
            name="Seed replicated comparison",
            marker_color=theme.series[0],
            text=[f"{value:.1f}%" for value in strong],
            textposition="outside",
            textfont={"color": theme.ink, "size": 14},
            cliponaxis=False,
        )
    )
    # The reference line goes in the legend rather than being annotated in the
    # plot: an in plot label at either end collided with a bar value, and the
    # legend row was already there.
    figure.add_trace(
        go.Scatter(
            x=sigmas,
            y=[NOMINAL_ALPHA * 100] * len(sigmas),
            mode="lines",
            name="nominal 5% false positive rate",
            line={"color": theme.muted, "width": 2, "dash": "dot"},
            hoverinfo="skip",
        )
    )

    style_axes(figure, theme, "seed to seed standard deviation", "false positive rate")
    figure.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.12,
        yaxis={"range": [0, 100], "ticksuffix": "%"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 13},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return figure


def uniformity_figure(theme: Theme) -> go.Figure:
    """Measured rejection rate against nominal, at four thresholds.

    A calibrated test lies on the diagonal everywhere, not only at 0.05.
    """
    from triage.calibration import UNIFORMITY

    nominal = [threshold for threshold, _ in UNIFORMITY]
    measured = [value for _, value in UNIFORMITY]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 0.55],
            y=[0, 0.55],
            mode="lines",
            name="perfect calibration",
            line={"color": theme.muted, "width": 2, "dash": "dot"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=nominal,
            y=measured,
            mode="markers+text",
            name="measured",
            marker={
                "color": theme.series[0],
                "size": 14,
                "line": {"color": theme.surface, "width": 2},
            },
            text=[f"  {value:.3f}" for value in measured],
            textposition="middle right",
            textfont={"color": theme.ink, "size": 13},
        )
    )
    style_axes(figure, theme, "nominal threshold", "measured rejection rate")
    figure.update_layout(
        xaxis={"range": [0, 0.58]},
        yaxis={"range": [0, 0.58]},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 13},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return figure


CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)


def screenshot_html_report(height: int = 1450) -> Path | None:
    """Capture the generated HTML report with a headless browser.

    Optional: it needs Chrome or Edge, and it is only used to produce the
    preview image on the landing page. Where no browser is found this returns
    None and says so rather than failing the run.
    """
    import shutil
    import subprocess

    report = ROOT / "experiments" / "results" / "triage_report.html"
    if not report.exists():
        print("no HTML report found; run `make demo` first to include the preview")
        return None

    browser = next(
        (path for path in CHROME_CANDIDATES if Path(path).exists()),
        shutil.which("chrome") or shutil.which("chromium") or shutil.which("google-chrome"),
    )
    if browser is None:
        print("no Chrome or Edge found; skipping the HTML report preview")
        return None

    IMAGES.mkdir(parents=True, exist_ok=True)
    output = IMAGES / "html-report.png"
    result = subprocess.run(
        [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--virtual-time-budget=20000",
            f"--window-size=1400,{height}",
            f"--screenshot={output}",
            report.as_uri(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if not output.exists():
        print(f"screenshot failed: {result.stderr.strip()[:300]}")
        return None
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=str(DEFAULT_DATABASE))
    parser.add_argument("--no-screenshot", action="store_true", help="skip the HTML report preview")
    args = parser.parse_args()

    database = Path(args.database)
    if not database.exists():
        raise SystemExit(f"no database at {database}; run `make demo` first")

    config = ComparisonConfig()
    with Store(database) as store:
        experiments = store.load_all()

    written: list[Path] = []
    for theme in THEMES:
        written.append(
            save(
                curve_figure(experiments, "val/loss", theme, config, "validation loss"),
                "val-loss-curves",
                theme,
                1180,
                560,
            )
        )
        written.append(
            save(
                curve_figure(experiments, "val/accuracy", theme, config, "validation accuracy"),
                "val-accuracy-curves",
                theme,
                1180,
                560,
            )
        )
        written.append(save(weak_mode_figure(theme), "weak-mode-cost", theme, 1000, 560))
        written.append(save(uniformity_figure(theme), "calibration-uniformity", theme, 900, 540))

    if not args.no_screenshot:
        preview = screenshot_html_report()
        if preview is not None:
            written.append(preview)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"{len(written)} images written to {IMAGES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
