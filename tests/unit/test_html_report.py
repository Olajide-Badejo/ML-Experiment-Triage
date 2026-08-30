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

import re
from pathlib import Path

import numpy as np
from markupsafe import escape

from triage.analysis.comparison import ComparisonConfig, ComparisonRefusal, compare_all
from triage.analysis.regression import RegressionConfig, TriageReport, classify, rank
from triage.analysis.sensitivity import analyse
from triage.core.experiment import Experiment, MetricSeries
from triage.report.html_report import RAGGED_NOTE, _variant_curve, build_context, render

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
    **kwargs,
):
    """A context assembled the way the CLI assembles one, from real analysis."""
    comparison_config = config or ComparisonConfig()
    regression_config = RegressionConfig()
    refusals: tuple[ComparisonRefusal, ...] = ()
    findings: list = []
    sensitivity: list = []
    if experiments:
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
