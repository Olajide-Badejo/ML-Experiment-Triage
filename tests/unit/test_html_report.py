"""The report renderer, driven directly rather than through the CLI.

The renderer ships the headline artifact of this tool and had no unit test at
all, which is how a stored cross site scripting hole survived in it: metric tag
names arrive from CSV headers, JSONL keys and TensorBoard tags, all of which are
third party data, and every one of them was interpolated into the emitted page.

So the fixtures here are deliberately hostile or degenerate rather than
representative. A run whose metric is named with an HTML payload, a condition
whose seeds died at different steps, a sweep with no metrics in it at all: these
are the inputs a real sweep produces on a bad day, and the renderer has to keep
its two promises under all of them, namely that the page is inert and that the
numbers on it are a pure function of the database.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pytest
from markupsafe import escape

from triage.analysis.comparison import ComparisonConfig, ComparisonRefusal, compare_all
from triage.analysis.regression import RegressionConfig, TriageReport, classify, rank
from triage.analysis.sensitivity import analyse
from triage.core.experiment import Experiment, MetricSeries
from triage.report import html_report
from triage.report.html_report import (
    MAX_PLOTTED_CONDITIONS,
    RAGGED_NOTE,
    _variant_curve,
    build_context,
    render,
)

#: The proof of concept payload from the specification. It closes the `id`
#: attribute the tag was interpolated into and opens an image element whose
#: `onerror` handler fires without any user interaction.
HOSTILE_TAG = 'loss"><img src=q onerror=alert(1)>'

#: A condition name that tries to break out of the inlined Plotly script block.
HOSTILE_VARIANT = "</script><script>alert(2)</script>"

SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL)
GRAPH_DIV = re.compile(r'<div id="([^"]*)" class="plotly-graph-div"')


def make_run(
    run_id: str,
    variant: str,
    seed: int,
    tags: list[str],
    *,
    offset: float = 0.0,
    n_points: int = 60,
    first_step: int = 0,
    noise_seed: int = 0,
    config: dict | None = None,
) -> Experiment:
    """One synthetic run: a falling curve plus reproducible noise.

    The noise is drawn from an explicitly seeded generator rather than from a
    hash of the identifiers, because several tests here assert byte identity
    across two renders and `hash()` is salted per process.
    """
    rng = np.random.default_rng(noise_seed)
    steps = np.arange(first_step, first_step + n_points)
    metrics = {
        tag: MetricSeries(
            tag=tag,
            steps=steps,
            values=1.0 - 0.005 * steps + offset + rng.normal(0.0, 0.01, n_points),
        )
        for tag in tags
    }
    full_config: dict = {"variant": variant, "seed": seed}
    full_config.update(config or {})
    return Experiment(
        run_id=run_id,
        source_path=f"/fixture/{run_id}",
        source_format="csv",
        config=full_config,
        metrics=metrics,
    )


def two_conditions(tags: list[str], *, baseline: str = "base", candidate: str = "cand") -> list:
    """Three seeds each of two conditions, separated enough to be a finding."""
    runs = []
    for index in range(3):
        runs.append(make_run(f"{baseline}-{index}", baseline, index, tags, noise_seed=index))
        runs.append(
            make_run(
                f"{candidate}-{index}",
                candidate,
                index,
                tags,
                offset=0.4,
                noise_seed=100 + index,
            )
        )
    return runs


def build(
    experiments: list[Experiment],
    baseline: str,
    *,
    config: ComparisonConfig | None = None,
    with_analysis: bool = True,
    **kwargs,
):
    """A context assembled the way the CLI assembles one, from real analysis.

    `with_analysis=False` skips the comparison layer, which is what the very
    large fixtures want: the unit under test here is the renderer, and running
    ten thousand permutations per condition to reach it measures something else.
    """
    comparison_config = config or ComparisonConfig()
    regression_config = RegressionConfig()
    refusals: tuple[ComparisonRefusal, ...] = ()
    findings: list = []
    sensitivity: list = []
    if experiments and with_analysis:
        results = compare_all(experiments, baseline, config=comparison_config)
        refusals = results.refusals
        findings = rank(classify(list(results), regression_config))
        sensitivity = analyse(experiments, config=comparison_config)
    triage = TriageReport(findings=findings, config=regression_config, baseline=baseline)
    return build_context(
        experiments=experiments,
        triage=triage,
        sensitivity=sensitivity,
        baseline=baseline,
        database="fixture.db",
        comparison_config=comparison_config,
        regression_config=regression_config,
        refusals=refusals,
        **kwargs,
    )


def render_html(tmp_path: Path, experiments: list, baseline: str, name: str = "r.html", **kwargs):
    path = render(build(experiments, baseline, **kwargs), tmp_path / name)
    return path.read_text(encoding="utf-8")


def markup_of(html: str) -> str:
    """The page with every script block removed: what the browser parses as HTML."""
    return SCRIPT_BLOCK.sub("", html)


# --------------------------------------------------------- D3: hostile names


def test_a_metric_name_carrying_a_payload_lands_inert(tmp_path: Path) -> None:
    """D3, the proof of concept: the tag closed its own attribute and injected.

    The tag reached `div_id` with only `/` and space replaced, so the emitted
    markup contained a live `<img onerror=...>` element. Nothing about the fix
    is cosmetic: the same string also broke the chart, because an unescaped
    double quote inside a JavaScript string literal is a syntax error.
    """
    html = render_html(tmp_path, two_conditions([HOSTILE_TAG]), "base")
    markup = markup_of(html)

    assert "<img" not in markup, "the payload opened an element in the emitted markup"
    assert '"><img' not in html, "the raw payload must not survive anywhere in the page"
    # Inert is not the same as hidden: the reader still has to be able to see
    # what the metric in their database is actually called.
    assert str(escape(HOSTILE_TAG)) in markup, "the name is still shown, escaped"


def test_every_figure_div_id_is_a_safe_slug(tmp_path: Path) -> None:
    """The id is now the index plus a slug, so it cannot carry markup or collide."""
    tags = [HOSTILE_TAG, "val/loss", "val loss"]
    html = render_html(tmp_path, two_conditions(tags), "base")

    ids = GRAPH_DIV.findall(html)
    assert len(ids) == len(tags)
    assert len(set(ids)) == len(tags), "distinct tags must not slug onto one id"
    for div_id in ids:
        assert re.fullmatch(r"figure-\d+-[A-Za-z0-9_-]*", div_id), div_id


def test_the_charts_still_render_with_a_hostile_metric_name(tmp_path: Path) -> None:
    """Escaping is only a fix if the figure survives it."""
    html = render_html(tmp_path, two_conditions([HOSTILE_TAG]), "base")

    assert "Plotly.newPlot(" in html
    assert len(GRAPH_DIV.findall(html)) == 1


def test_plotly_labels_carry_no_unescaped_markup(tmp_path: Path) -> None:
    """Plotly renders `name` and `hovertemplate` through a tag interpreting
    converter, so a `<` reaching them is markup in the chart itself even though
    the surrounding JSON is escaped."""
    html = render_html(tmp_path, two_conditions([HOSTILE_TAG]), "base")

    # Plotly spells a literal `<` as `<` inside its JSON payload, so this
    # is the shape an unescaped angle bracket would take on the way to the
    # converter. `&lt;` is what it looks like once escaped.
    assert "\\u003cimg" not in html
    assert "&lt;img" in html


def test_a_hostile_condition_name_cannot_close_the_script_block(tmp_path: Path) -> None:
    html = render_html(tmp_path, two_conditions([HOSTILE_TAG], candidate=HOSTILE_VARIANT), "base")

    assert "<script>alert(2)</script>" not in html
    assert "Plotly.newPlot(" in html


# ----------------------------------------------------- D9: ragged run lengths


def ragged_pair(tag: str) -> list[Experiment]:
    """One seed that ran to the end and one that was killed a fifth of the way in."""
    return [
        make_run("long", "cand", 0, [tag], n_points=200, noise_seed=1),
        make_run("short", "cand", 1, [tag], n_points=40, noise_seed=2),
    ]


def test_a_run_that_died_early_contributes_nothing_past_its_last_step() -> None:
    """D9: the band in the final window was fabricated from a clamped curve.

    `np.interp` clamps by default, so a seed killed at step 40 of 200 was
    extended as a flat line to step 200 and folded into the mean and the min max
    envelope, exactly inside the shaded window the statistic is taken over. The
    figure then showed a spread across seeds that no seed ever measured.
    """
    tag = "val/loss"
    runs = ragged_pair(tag)
    grid, mean, low, high = _variant_curve(runs, tag, ComparisonConfig())

    assert grid.size == 200, "the grid still comes from the longest run"
    assert np.isfinite(mean).all() and np.isfinite(low).all() and np.isfinite(high).all()

    survivor = np.interp(
        grid,
        runs[0].series(tag).steps.astype(np.float64),
        runs[0].series(tag).smoothed(ComparisonConfig().smoothing_window),
    )
    # Past step 40 exactly one seed has data, so the band has to collapse onto
    # that seed rather than straddle it and the dead one.
    assert high[-1] == low[-1] == mean[-1]
    assert mean[-1] == survivor[-1]
    assert high[-1] - low[-1] == 0.0


def test_the_seeds_that_do_overlap_are_still_averaged() -> None:
    """The fix must not throw away the region where both seeds are alive."""
    tag = "val/loss"
    runs = ragged_pair(tag)
    _, mean, low, high = _variant_curve(runs, tag, ComparisonConfig())

    assert high[0] > low[0], "both seeds are alive at step 0, so there is a spread"
    assert low[0] <= mean[0] <= high[0]


def test_a_figure_over_ragged_seeds_says_so(tmp_path: Path) -> None:
    tag = "val/loss"
    runs = [
        *[make_run(f"base-{i}", "base", i, [tag], noise_seed=i) for i in range(3)],
        make_run("cand-0", "cand", 0, [tag], offset=0.4, n_points=200, noise_seed=10),
        make_run("cand-1", "cand", 1, [tag], offset=0.4, n_points=200, noise_seed=11),
        make_run("cand-2", "cand", 2, [tag], offset=0.4, n_points=40, noise_seed=12),
    ]
    html = render_html(tmp_path, runs, "base")
    assert RAGGED_NOTE in html


def test_a_figure_over_even_seeds_is_not_annotated(tmp_path: Path) -> None:
    """The annotation is a caveat, and a caveat printed always is noise."""
    html = render_html(tmp_path, two_conditions(["val/loss"]), "base")
    assert RAGGED_NOTE not in html


# ------------------------------------------- D23: the runtime is not free


def test_a_report_with_no_figures_does_not_inline_the_plotly_runtime(tmp_path: Path) -> None:
    """D23: 4.9 MB of charting runtime shipped even with nothing to chart."""
    path = render(build([], "base"), tmp_path / "empty.html")

    assert path.stat().st_size < 100_000, "nothing to draw, so nothing to draw it with"
    html = path.read_text(encoding="utf-8")
    assert "Plotly.newPlot(" not in html
    assert "<script>" not in html, "the guarded block must not be emitted empty either"


def test_a_report_with_figures_still_inlines_the_runtime(tmp_path: Path) -> None:
    """Self contained is the promise the whole module is shaped around."""
    html = render_html(tmp_path, two_conditions(["val/loss"]), "base")

    assert len(html) > 1_000_000
    assert "Plotly.newPlot(" in html
    assert "<script src=" not in html


# ------------------------------------ D24: every metric in the database


def baseline_is_missing_a_metric() -> list[Experiment]:
    """A metric only the candidate logged, so no comparison can be built on it."""
    runs = []
    for index in range(3):
        runs.append(make_run(f"base-{index}", "base", index, ["val/loss"], noise_seed=index))
        runs.append(
            make_run(
                f"cand-{index}",
                "cand",
                index,
                ["val/loss", "val/extra"],
                offset=0.4,
                noise_seed=100 + index,
            )
        )
    return runs


def test_a_metric_with_no_finding_still_gets_a_curve(tmp_path: Path) -> None:
    """D24: figures were keyed off findings, so an uncompared metric vanished.

    A metric the baseline never logged cannot be compared against the baseline,
    which is exactly when a reader most wants to see what it did.
    """
    html = render_html(tmp_path, baseline_is_missing_a_metric(), "base")

    assert len(GRAPH_DIV.findall(html)) == 2, "both metrics in the database get a curve"
    assert "val/extra" in html


def test_a_metric_with_no_finding_says_why(tmp_path: Path) -> None:
    html = render_html(tmp_path, baseline_is_missing_a_metric(), "base")
    assert "not compared against the baseline" in markup_of(html)


def test_the_inventory_covers_every_metric_and_marks_the_gaps(tmp_path: Path) -> None:
    html = render_html(tmp_path, baseline_is_missing_a_metric(), "base")
    markup = markup_of(html)

    inventory = markup.split("Run inventory")[1]
    assert inventory.count("val/extra") >= 1, "the column is keyed off the union of tags"
    assert "&mdash;" in inventory, "the baseline has no value for that metric and says so"


def test_a_report_with_no_metrics_at_all_says_so(tmp_path: Path) -> None:
    path = render(build([], "base"), tmp_path / "empty.html")
    assert "no metric" in path.read_text(encoding="utf-8").lower()


# ------------------------------------ D25: telling many conditions apart


LINE_STYLE = re.compile(r'"line":\{"color":"(#[0-9a-f]{6})","dash":"([a-z]+)","width":2\}')


def many_conditions(count: int, tag: str = "val/loss", seeds: int = 3) -> list[Experiment]:
    """`count` conditions including the baseline, separated by a graded effect."""
    runs = [make_run(f"base-{seed}", "base", seed, [tag], noise_seed=seed) for seed in range(seeds)]
    for index in range(1, count):
        for seed in range(seeds):
            runs.append(
                make_run(
                    f"c{index:02d}-{seed}",
                    f"c{index:02d}",
                    seed,
                    [tag],
                    offset=0.02 * index,
                    noise_seed=1000 * index + seed,
                )
            )
    return runs


def test_condition_nine_is_not_pixel_identical_to_condition_one(tmp_path: Path) -> None:
    """D25: the palette has eight slots and the line style wrapped with it.

    Condition nine took colour slot one back and drew it as the same solid line,
    so two conditions were indistinguishable on the chart and in the legend swatch
    alike. The dash pattern now advances when the colour wraps.
    """
    html = render_html(tmp_path, many_conditions(12), "base")

    styles = LINE_STYLE.findall(html)
    assert len(styles) == 12, "one line per condition, baseline included"
    assert len(set(styles)) == 12, "no two conditions may share a colour and a dash"


def test_only_the_most_severe_conditions_are_drawn_individually(tmp_path: Path) -> None:
    """D25: at 500 runs the page was 14.5 MB of curves nobody can read.

    A chart with fifty overlaid lines is not a chart. The conditions that carry
    the findings are drawn, the rest are folded into one envelope that still
    shows where they went, and the count of what was folded is on the legend so
    the reader knows the chart is not the whole sweep.
    """
    total = MAX_PLOTTED_CONDITIONS + 4  # baseline plus three more than fit
    html = render_html(tmp_path, many_conditions(total, seeds=5), "base")

    styles = LINE_STYLE.findall(html)
    assert len(styles) == MAX_PLOTTED_CONDITIONS + 1, "the cap, plus the baseline"
    assert len(set(styles)) == len(styles)

    folded = total - 1 - MAX_PLOTTED_CONDITIONS
    assert f"{folded} further conditions" in html

    worst = f"c{total - 1:02d}"
    mildest = "c01"
    assert f'"name":"{worst},' in html, "the largest effect must be one of the drawn lines"
    assert f'"name":"{mildest},' not in html, "the mildest condition is the one folded away"


def test_the_baseline_is_always_drawn_however_many_conditions_there_are(
    tmp_path: Path,
) -> None:
    """The baseline is the reference every other line is read against."""
    html = render_html(tmp_path, many_conditions(MAX_PLOTTED_CONDITIONS + 6, seeds=3), "base")
    assert '"name":"base (baseline),' in html


def test_the_table_headers_stay_put_while_a_long_table_scrolls(tmp_path: Path) -> None:
    """503 unpaginated rows with the header off screen is a table of numbers
    whose columns nobody can name."""
    html = render_html(tmp_path, two_conditions(["val/loss"]), "base")
    assert "position: sticky" in html


def test_a_report_over_the_size_limit_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten megabytes of evidence does not survive being emailed, which is the
    one thing this report is built to do."""
    monkeypatch.setattr(html_report, "LARGE_REPORT_BYTES", 1024)
    with caplog.at_level(logging.WARNING, logger="triage.report"):
        render_html(tmp_path, two_conditions(["val/loss"]), "base")
    assert any("larger than" in record.getMessage() for record in caplog.records)


def test_a_report_under_the_size_limit_says_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="triage.report"):
        render(build([], "base"), tmp_path / "empty.html")
    assert not caplog.records


# ----------------------------------------- D26: the page a screen reader gets


def relative_luminance(colour: str) -> float:
    """WCAG 2.1 relative luminance of a `#rrggbb` colour."""
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(one: str, other: str) -> float:
    darker, lighter = sorted((relative_luminance(one), relative_luminance(other)))
    return (lighter + 0.05) / (darker + 0.05)


def test_the_muted_ink_clears_wcag_aa_for_the_text_it_styles(tmp_path: Path) -> None:
    """D26, measured: `#898781` on `#fcfcfb` is 3.50:1, and it styles normal
    size body text: table headers, tile labels, captions and every axis on every
    chart. AA wants 4.5:1 for that."""
    html = render_html(tmp_path, two_conditions(["val/loss"]), "base")

    muted = re.search(r"--muted:\s*(#[0-9a-f]{6});", html).group(1)
    surface = re.search(r"--surface:\s*(#[0-9a-f]{6});", html).group(1)

    assert contrast_ratio(muted, surface) >= 5.0
    # The charts are drawn in Python and the page in CSS, so the same ink is
    # written twice and can drift. It must not.
    assert muted == html_report.MUTED_INK
    assert surface == html_report.SURFACE


def every_section() -> list[Experiment]:
    """A sweep that fills all four tables: verdicts, refusals, inventory, sensitivity.

    Six conditions over a swept learning rate give the rank correlation enough
    to work with, and one condition too short for any calibrated test puts a row
    in the refusals table.
    """
    runs = []
    for index in range(6):
        name = "base" if index == 0 else f"c{index:02d}"
        for seed in range(3):
            runs.append(
                make_run(
                    f"{name}-{seed}",
                    name,
                    seed,
                    ["val/loss"],
                    offset=0.02 * index,
                    noise_seed=1000 * index + seed,
                    config={"lr": 0.001 * (index + 1)},
                )
            )
    runs.append(
        make_run("tiny-0", "tiny", 0, ["val/loss"], n_points=6, noise_seed=7, config={"lr": 0.05})
    )
    return runs


def test_every_table_names_itself_and_scopes_its_headers(tmp_path: Path) -> None:
    markup = markup_of(render_html(tmp_path, every_section(), "base"))

    tables = len(re.findall(r"<table\b", markup))
    assert tables == 4, "verdicts, refusals, run inventory and sensitivity"
    assert len(re.findall(r"<caption\b", markup)) == tables

    headers = re.findall(r"<th\b[^>]*>", markup)
    assert headers
    assert all('scope="col"' in header for header in headers)


def test_the_page_has_one_main_landmark(tmp_path: Path) -> None:
    markup = markup_of(render_html(tmp_path, two_conditions(["val/loss"]), "base"))
    assert len(re.findall(r"<main\b", markup)) == 1
    assert "</main>" in markup


def test_every_figure_is_labelled_for_a_reader_who_cannot_see_it(tmp_path: Path) -> None:
    """A chart is a canvas of nothing to anyone not looking at it, so it has to
    say what it is."""
    html = render_html(tmp_path, baseline_is_missing_a_metric(), "base")
    markup = markup_of(html)

    figures = re.findall(r'<div class="figure"[^>]*>', markup)
    assert len(figures) == len(GRAPH_DIV.findall(html)) == 2
    for figure in figures:
        assert 'role="img"' in figure
        assert "aria-label=" in figure
    assert "val/loss" in markup


# ------------------------------------------------- D27: a stamp you can fix


#: An arbitrary but fixed moment. `1700000000` is the same instant in UTC.
FIXED_GENERATED_AT = "2023-11-14 22:13 UTC"
FIXED_EPOCH = "1700000000"


def test_the_page_stamps_the_moment_it_was_given(tmp_path: Path) -> None:
    html = render_html(
        tmp_path, two_conditions(["val/loss"]), "base", generated_at=FIXED_GENERATED_AT
    )
    assert FIXED_GENERATED_AT in html


def test_two_builds_with_a_fixed_stamp_are_byte_identical(tmp_path: Path) -> None:
    """D27: the determinism gate used to mask the timestamp and pass by luck.

    Masking is not a fix, it is a way of not testing the thing. With the moment
    supplied rather than read off the wall clock, the report is a pure function
    of its inputs and the two files can be compared as bytes.
    """
    runs = two_conditions(["val/loss"])
    first = render(build(runs, "base", generated_at=FIXED_GENERATED_AT), tmp_path / "first.html")
    second = render(build(runs, "base", generated_at=FIXED_GENERATED_AT), tmp_path / "second.html")

    assert first.read_bytes() == second.read_bytes()


def test_source_date_epoch_fixes_the_stamp_in_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reproducible builds convention, so a caller that cannot pass an
    argument, `make` for instance, can still pin the stamp. UTC on purpose: the
    local zone name is what made the old footer differ between two machines
    building from one database."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", FIXED_EPOCH)
    assert build([], "base").generated_at == FIXED_GENERATED_AT


def test_an_explicit_stamp_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", FIXED_EPOCH)
    assert build([], "base", generated_at="whenever").generated_at == "whenever"


def test_an_unparseable_source_date_epoch_falls_back_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed environment variable must not cost somebody their report."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not a number")
    with caplog.at_level(logging.WARNING, logger="triage.report"):
        stamp = build([], "base").generated_at
    assert stamp
    assert FIXED_GENERATED_AT not in stamp
    assert any("SOURCE_DATE_EPOCH" in record.getMessage() for record in caplog.records)


# ------------------------------------------------ D28: the ends of the range


def test_a_single_run_still_renders_a_report(tmp_path: Path) -> None:
    """One run is a legitimate database. Nothing can be compared against it, and
    the page has to say that rather than fall over on an empty table."""
    only = [make_run("only", "base", 0, ["val/loss"], noise_seed=3)]
    html = render_html(tmp_path, only, "base")

    assert len(GRAPH_DIV.findall(html)) == 1
    assert "not compared against the baseline" in markup_of(html)
    assert ">1<" in html, "the run count tile"


def test_a_five_hundred_run_sweep_renders_a_readable_page(tmp_path: Path) -> None:
    """D25 and D28: the case that produced a 14.5 MB page of fifty curves.

    The analysis is skipped on purpose; the renderer is what is under test, and
    the number of runs is what is being varied.
    """
    runs = []
    for condition in range(20):
        name = "base" if condition == 0 else f"c{condition:02d}"
        for seed in range(25):
            runs.append(
                make_run(
                    f"{name}-{seed}",
                    name,
                    seed,
                    ["val/loss"],
                    offset=0.01 * condition,
                    noise_seed=1000 * condition + seed,
                )
            )
    assert len(runs) == 500

    path = render(build(runs, "base", with_analysis=False), tmp_path / "big.html")
    html = path.read_text(encoding="utf-8")

    assert len(LINE_STYLE.findall(html)) == MAX_PLOTTED_CONDITIONS + 1
    assert "7 further conditions" in html
    assert path.stat().st_size < html_report.LARGE_REPORT_BYTES
    # Every condition is still accounted for in the inventory, which is the
    # table the chart must not silently disagree with.
    assert markup_of(html).count("<tr>") >= 20
