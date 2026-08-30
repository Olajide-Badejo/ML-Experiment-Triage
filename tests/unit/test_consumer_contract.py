"""The frozen import surface a real downstream project depends on.

`Autofill_audit` (dist `autofill-audit`) pins this package and imports it from
exactly one module, `evaluate/triage_bridge.py`, with an AST test on its own side
that fails if a second importer ever appears. It uses only the statistics layer:
no parsers of ours run in its pipeline except as a probe, no reports are
rendered, no CLI is invoked. That makes this list, and not the CLI, the surface
that breaking would break somebody.

The rule these tests enforce is narrow and absolute: **these names keep importing
from these exact module paths, and keep accepting the shapes the consumer builds
them with.** Behaviour underneath may change with a CHANGELOG entry the consumer
can act on; the import lines and the constructor calls may not. A test that
imported them from `triage` instead, or that reached for whatever path a refactor
had moved them to, would pass while the consumer's build failed, so every import
here is written out literally.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# The contract, written as the consumer writes it. Any failure to import here is
# the failure the consumer would see, at the same line number they would see it.
from triage.analysis.comparison import ComparisonResult, permutation_p_value
from triage.analysis.regression import (
    VERDICT_BELOW_THRESHOLD,
    VERDICT_IMPROVEMENT,
    VERDICT_NO_CHANGE,
    VERDICT_REGRESSION,
    VERDICT_UNDERPOWERED,
    RegressionConfig,
    benjamini_hochberg,
    classify,
)
from triage.parsers import JsonlParser, ParseError

#: `module path -> names that must resolve from it`, exactly as filed.
CONTRACT = {
    "triage.analysis.comparison": ("ComparisonResult", "permutation_p_value"),
    "triage.analysis.regression": (
        "VERDICT_BELOW_THRESHOLD",
        "VERDICT_IMPROVEMENT",
        "VERDICT_NO_CHANGE",
        "VERDICT_REGRESSION",
        "VERDICT_UNDERPOWERED",
        "RegressionConfig",
        "benjamini_hochberg",
        "classify",
    ),
    "triage.parsers": ("JsonlParser", "ParseError"),
}


def a_result(
    tag: str = "macro_f1",
    p_value: float = 0.01,
    relative_pct: float = -10.0,
    mode: str = "template_clustered_paired",
    key: str = "",
) -> ComparisonResult:
    """A result shaped the way the consumer builds one: its own mode string."""
    return ComparisonResult(
        tag=tag,
        baseline="rules",
        candidate="ngram",
        mode=mode,
        test_name="paired permutation test clustered by form template",
        baseline_statistic=0.80,
        candidate_statistic=0.80 + relative_pct / 100.0,
        effect=relative_pct / 100.0,
        relative_effect_pct=relative_pct,
        effect_size=0.9,
        effect_size_name="Cohen's d",
        ci_low=-0.2,
        ci_high=-0.05,
        ci_method="cluster bootstrap",
        ci_level=0.95,
        p_value=p_value,
        n_permutations=1024,
        exact=True,
        min_attainable_p=0.002,
        n_baseline=10,
        n_candidate=10,
        window_points=0,
        higher_is_better=True,
        seed=7,
        key=key,
    )


# ------------------------------------------------------------- the import list


@pytest.mark.parametrize(
    ("module_path", "name"),
    [(path, name) for path, names in CONTRACT.items() for name in names],
)
def test_every_contracted_name_resolves_from_its_exact_module(module_path: str, name: str) -> None:
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f"{module_path}.{name} is part of the frozen contract"


def test_the_contract_is_the_size_it_was_filed_at() -> None:
    """A guard on the list itself, so a name cannot be quietly dropped from it."""
    assert sum(len(names) for names in CONTRACT.values()) == 12
    assert {ComparisonResult, RegressionConfig, JsonlParser, ParseError}
    assert callable(permutation_p_value)
    assert callable(benjamini_hochberg)
    assert callable(classify)


def test_the_verdict_strings_are_stable() -> None:
    """The consumer stores these in `analysis.json` and compares them as text."""
    assert VERDICT_REGRESSION == "regression"
    assert VERDICT_IMPROVEMENT == "improvement"
    assert VERDICT_NO_CHANGE == "no significant change"
    assert VERDICT_BELOW_THRESHOLD == "significant but below the practical threshold"
    assert VERDICT_UNDERPOWERED == "inconclusive: the design cannot reach alpha"


# ------------------------------------------------- the constructions they make


def test_the_default_construction_still_works() -> None:
    config = RegressionConfig()
    assert config.alpha == 0.05
    assert config.practical_threshold_pct == 2.0
    assert config.practical_threshold_absolute is None
    assert config.false_discovery_rate == 0.05


def test_the_keyword_construction_they_pinned_against_still_works() -> None:
    """v1.0.0's three keyword call, which must keep constructing the same gates."""
    config = RegressionConfig(alpha=0.01, practical_threshold_pct=1.0, false_discovery_rate=0.10)
    assert (config.alpha, config.practical_threshold_pct, config.false_discovery_rate) == (
        0.01,
        1.0,
        0.10,
    )
    assert "1 percent" in config.describe()


def test_the_absolute_threshold_construction_they_filed_for_works() -> None:
    """Their issue #3: a gate in the metric's own units, for latency_us and kin."""
    config = RegressionConfig(practical_threshold_pct=None, practical_threshold_absolute=50.0)
    assert config.practical_threshold_absolute == 50.0
    assert config.practical_threshold_pct is None
    assert "50" in config.describe()

    with pytest.raises(ValueError, match="exactly one"):
        RegressionConfig(practical_threshold_pct=2.0, practical_threshold_absolute=50.0)


def test_the_old_keyword_is_accepted_for_one_minor_version_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning):
        config = RegressionConfig(practical_threshold=4.0)
    assert config.practical_threshold_pct == 4.0

    # And the new spelling must not warn, or the consumer's `-W error` build
    # would fail on the fix rather than on the thing being deprecated.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RegressionConfig(practical_threshold_pct=4.0)


# --------------------------------------------------------- the calls they make


def test_classify_joins_positionally_to_the_results_it_was_given() -> None:
    """Their issue #4: the `id(finding.result)` rejoin can be retired."""
    results = [
        a_result(tag="macro_f1", p_value=0.001, relative_pct=+8.0, key="ngram|macro_f1"),
        a_result(tag="latency_us", p_value=0.4, relative_pct=+1.0, key="ngram|latency_us"),
    ]
    findings = classify(results, RegressionConfig())
    assert len(findings) == len(results)
    for finding, result in zip(findings, results, strict=True):
        assert finding.result is result
    assert [finding.result.key for finding in findings] == ["ngram|macro_f1", "ngram|latency_us"]


def test_a_result_built_with_the_consumers_own_mode_string_is_reportable() -> None:
    """Their issue #2: an unknown mode must not raise out of the reporting path."""
    result = a_result(mode="template_clustered_paired")
    assert "template_clustered_paired" in result.mode_label
    assert result.to_dict()["mode"] == "template_clustered_paired"
    assert classify([result], RegressionConfig())[0].family == "template_clustered_paired"


def test_benjamini_hochberg_still_takes_a_list_and_returns_an_array() -> None:
    adjusted = benjamini_hochberg([0.001, 0.02, 0.5], 0.05)
    assert isinstance(adjusted, np.ndarray)
    assert adjusted.shape == (3,)
    assert np.all(np.diff(np.sort(adjusted)) >= 0)


def test_permutation_p_value_still_takes_observed_null_and_exact() -> None:
    null = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert permutation_p_value(2.0, null, True) == pytest.approx(1 / 5)
    assert 0.0 < permutation_p_value(0.5, null, False) <= 1.0


def test_the_jsonl_parser_and_its_error_are_still_the_probe_they_use() -> None:
    """Their bridge hands a step free outcomes file to this parser as a probe."""
    assert issubclass(ParseError, Exception)
    assert hasattr(JsonlParser(), "can_parse")


# ------------------------------------------------- D32: the top level surface

#: `triage.__all__`, exactly as the consumer's issue #5 part 3 asked for it, in
#: the order it was filed. `dir(triage)` was empty: every one of these names was
#: reachable only through a deep module path, so the package had a public API in
#: practice and none on paper, and a consumer pinning it had nothing to pin to.
TOP_LEVEL = (
    "Experiment",
    "MetricSeries",
    "Store",
    "StoreError",
    "ingest",
    "IngestResult",
    "discover_runs",
    "ParseError",
    "ComparisonConfig",
    "ComparisonResult",
    "compare_all",
    "paired_permutation",
    "RegressionConfig",
    "classify",
    "rank",
    "TriageReport",
    "analyse",
    "build_context",
    "render",
)

#: Where each top level name really lives. The re export is a convenience and
#: the deep path is the contract (Section 1 rule 5), so both must resolve and,
#: crucially, must be the SAME object: two names for two objects would be the
#: worst of both, with a consumer's isinstance check passing or failing by which
#: import line they happened to write.
DEEP_PATHS = {
    "Experiment": "triage.core.experiment",
    "MetricSeries": "triage.core.experiment",
    "Store": "triage.core.store",
    "StoreError": "triage.core.store",
    "ingest": "triage.ingest",
    "IngestResult": "triage.ingest",
    "discover_runs": "triage.parsers",
    "ParseError": "triage.parsers",
    "ComparisonConfig": "triage.analysis.comparison",
    "ComparisonResult": "triage.analysis.comparison",
    "compare_all": "triage.analysis.comparison",
    "paired_permutation": "triage.analysis.comparison",
    "RegressionConfig": "triage.analysis.regression",
    "classify": "triage.analysis.regression",
    "rank": "triage.analysis.regression",
    "TriageReport": "triage.analysis.regression",
    "analyse": "triage.analysis.sensitivity",
    "build_context": "triage.report.html_report",
    "render": "triage.report.html_report",
}


def test_the_top_level_all_is_exactly_the_list_that_was_asked_for() -> None:
    """Membership, not order: `__all__` is sorted, the filed list was not."""
    import triage

    assert set(triage.__all__) == set(TOP_LEVEL)
    assert len(triage.__all__) == len(TOP_LEVEL) == 19
    assert set(DEEP_PATHS) == set(TOP_LEVEL)
    # And the lazy resolution table cannot drift from the advertised list.
    assert set(triage._EXPORTS) == set(triage.__all__)


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_every_top_level_name_resolves_and_is_the_deep_one(name: str) -> None:
    import triage

    from_top = getattr(triage, name)
    from_deep = getattr(importlib.import_module(DEEP_PATHS[name]), name)
    assert from_top is from_deep, f"triage.{name} must BE {DEEP_PATHS[name]}.{name}"


def test_dir_of_the_package_lists_the_public_names() -> None:
    """`dir(triage)` was empty, which is what a package with no API looks like."""
    import triage

    listed = dir(triage)
    assert set(TOP_LEVEL) <= set(listed)
    assert "__version__" in listed


def test_importing_the_package_does_not_drag_in_the_reporting_stack() -> None:
    """The consumer installs numpy and scipy and nothing else (E1).

    `build_context` and `render` are in `__all__`, and their module imports
    plotly and jinja2. Re exporting them eagerly would therefore have made
    `import triage` fail on precisely the install the statistics API exists to
    serve, so the top level names resolve on first access rather than on import.
    """
    code = (
        "import sys, triage; "
        "assert 'plotly' not in sys.modules, sorted(m for m in sys.modules if 'plotly' in m); "
        "assert 'jinja2' not in sys.modules"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_an_unknown_top_level_name_still_raises_attribute_error() -> None:
    """The lazy hook must not turn a typo into something that looks importable."""
    import triage

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        triage.nope  # noqa: B018 - the attribute access IS the assertion


def test_build_context_defaults_its_calibration_note() -> None:
    """D32: a caller building a context should not have to know where it lives."""
    import inspect

    from triage.calibration import SUMMARY
    from triage.report.html_report import build_context

    default = inspect.signature(build_context).parameters["calibration"].default
    assert default is SUMMARY


# ----------------------------------------------- E1: the core only install

#: Everything the extras carry. A core install is numpy and scipy; these five
#: are what `parsers`, `report` and the tooling around them add.
HEAVY = ("pandas", "plotly", "jinja2", "tensorboard", "tqdm")

#: Prelude that turns the running interpreter into a core only one. The heavy
#: packages ARE installed in the development environment, so the only way to
#: test the install the consumer actually has is to refuse them at import time.
#: A meta path finder is used rather than uninstalling anything, so the check
#: costs a subprocess and not an environment.
BLOCK_HEAVY = '''
import sys

BLOCKED = {"pandas", "plotly", "jinja2", "tensorboard", "tqdm"}


class CoreOnly:
    """Refuses the extras, exactly as a core install would."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, CoreOnly())
'''


def run_core_only(body: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in a subprocess that cannot import any of the extras."""
    return subprocess.run(
        [sys.executable, "-c", BLOCK_HEAVY + body],
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_core_only_install_can_import_every_module_of_the_package() -> None:
    """E1. Import must not be where a core install dies.

    `triage.ingest` is imported eagerly by `triage/__init__.py` and pulls in
    `triage.parsers`, which offers the CSV and TensorBoard readers to every
    path `discover_runs` walks. If importing those modules imported pandas and
    tensorboard, a consumer with numpy and scipy could not say `import triage`
    at all, whatever they meant to do with it.
    """
    result = run_core_only(
        "import triage, triage.cli, triage.ingest, triage.parsers, "
        "triage.report.html_report, triage.progress, triage.demo\n"
        "import sys\n"
        "loaded = sorted(m for m in ('pandas', 'plotly', 'jinja2', 'tensorboard', 'tqdm')\n"
        "                if m in sys.modules)\n"
        "assert loaded == [], loaded\n"
        "print('imported')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


def test_a_core_only_install_can_ingest_jsonl_and_run_a_comparison(tmp_path: Path) -> None:
    """E1, the whole point of the split, end to end without the extras.

    This is the consumer's runtime shape: JSONL rows in, a ranked significance
    tested comparison out, with numpy and scipy the only third party packages
    on the machine. It runs in a subprocess with the extras refused at import
    time, so nothing this test proves can be an accident of the development
    environment having pandas installed.
    """
    sweep = tmp_path / "sweep"
    seeds = 5
    for condition, offset in (("base", 0.0), ("candidate", -0.25)):
        for seed in range(seeds):
            directory = sweep / f"{condition}_seed{seed}"
            directory.mkdir(parents=True)
            # A config is what makes seed replicates group into one condition:
            # `variant_key` reads the config, not the directory name.
            (directory / "config.json").write_text(
                json.dumps({"variant": condition, "seed": seed}), encoding="utf-8"
            )
            # Deterministic seed to seed spread, so the strong mode has
            # something to permute and the comparison is not degenerate.
            jitter = 0.004 * (seed - 2)
            rows = [
                json.dumps({"step": step, "val/loss": 1.0 + offset + jitter - 0.002 * step})
                for step in range(60)
            ]
            (directory / "metrics.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    body = f"""
import sys
from triage import Store, compare_all, ingest

sweep, database = {str(sweep)!r}, {str(tmp_path / "core.db")!r}
with Store(database) as store:
    outcome = ingest(sweep, store, show_progress=False)
    assert not outcome.failed, outcome.failed
    assert len(outcome.added) == {2 * seeds}, outcome.added
    experiments = store.load_all()

results = compare_all(experiments, baseline="base")
assert len(results) == 1, [r.candidate for r in results]
assert results[0].candidate == "candidate"
assert results[0].p_value < 0.05, results[0].p_value
loaded = sorted(m for m in {HEAVY!r} if m in sys.modules)
assert loaded == [], loaded
print("compared", results[0].mode)
"""
    result = run_core_only(body)

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("compared")


@pytest.mark.parametrize(
    ("statement", "extra", "package"),
    [
        ("from triage.parsers import CsvParser; CsvParser().parse(P)", "parsers", "pandas"),
        ("from triage.report.html_report import render; render({}, P)", "report", "jinja2"),
    ],
)
def test_a_missing_extra_says_which_extra_and_what_to_type(
    tmp_path: Path, statement: str, extra: str, package: str
) -> None:
    """E1. The error a core install meets has to be actionable.

    Without this the failure is `ModuleNotFoundError: No module named 'plotly'`,
    which names the package and nothing else: not what wanted it, not that it
    is optional, not how to get it.
    """
    csv_run = tmp_path / "run"
    csv_run.mkdir()
    (csv_run / "metrics.csv").write_text("step,loss\n0,1.0\n", encoding="utf-8")

    body = f"""
from pathlib import Path
P = Path({str(csv_run)!r})
from triage._extras import MissingExtraError
try:
    {statement}
except MissingExtraError as error:
    message = str(error)
else:
    raise AssertionError("the missing extra did not raise")
assert {package!r} in message, message
assert 'ml-experiment-triage[{extra}]' in message, message
print("refused")
"""
    result = run_core_only(body)

    assert result.returncode == 0, result.stderr
    assert "refused" in result.stdout


def test_the_missing_extra_error_is_an_import_error() -> None:
    """Callers already guard optional features with `except ImportError`."""
    from triage._extras import MissingExtraError

    assert issubclass(MissingExtraError, ImportError)


def project_table() -> dict[str, Any]:
    """The `[project]` table of the pyproject this checkout builds from."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as handle:
        table: dict[str, Any] = tomllib.load(handle)["project"]
    return table


def requirement_names(requirements: list[str]) -> list[str]:
    return sorted(re.split(r"[<>=!~;\[ ]", requirement)[0] for requirement in requirements)


def test_the_core_install_is_numpy_and_scipy_and_nothing_else() -> None:
    """E1, as filed. The declaration is the promise; the imports are the proof."""
    assert requirement_names(project_table()["dependencies"]) == ["numpy", "scipy"]


def test_the_extras_are_the_split_that_was_asked_for() -> None:
    extras = project_table()["optional-dependencies"]
    assert {"parsers", "report", "cli", "agentic", "all"} <= set(extras)
    assert requirement_names(extras["parsers"]) == ["pandas", "tensorboard", "tqdm"]
    assert requirement_names(extras["report"]) == ["jinja2", "plotly"]
    assert extras["cli"] == ["ml-experiment-triage[parsers,report]"]
    assert extras["agentic"] == ["choreographer>=1.3"]
    assert extras["all"] == ["ml-experiment-triage[cli,agentic]"]


def test_the_declared_floors_are_the_versions_the_results_were_measured_on() -> None:
    """D30. A floor nothing has ever installed is a guess written as a fact.

    These four decode the numbers this package publishes: numpy and scipy do
    the statistics, pandas the CSV values, and the plotly bundle is inlined
    into the report verbatim so its version is part of the output bytes.
    """
    table = project_table()
    declared = dict(
        requirement.split(">=", 1)
        for requirement in table["dependencies"] + table["optional-dependencies"]["parsers"]
        if ">=" in requirement and "," not in requirement
    )
    assert declared["numpy"] == "2.5"
    assert declared["scipy"] == "1.18"
    assert declared["pandas"] == "3.0"
    assert "plotly>=6.9,<7" in table["optional-dependencies"]["report"]


# ---------------------------------------------------- D29: packaging metadata


def test_the_version_is_written_in_exactly_one_place() -> None:
    """D29. pyproject held one literal and triage/__init__.py held another.

    Nothing kept them in step, so a release could ship a wheel whose metadata
    said one version and whose `__version__` said the other, which is the kind
    of disagreement a consumer pinning a range discovers late.
    """
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["dynamic"] == ["version"]
    assert "version" not in pyproject["project"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "triage._version.VERSION"
    }

    # And the module setuptools reads must stay importable without the runtime
    # dependencies, because the build environment does not have them.
    source = (root / "triage" / "_version.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert len(assignments) == 1
    assert [target.id for node in assignments for target in node.targets] == ["VERSION"]  # type: ignore[attr-defined]
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    assert imports == []


def test_the_reported_version_is_the_installed_one() -> None:
    import triage
    from triage._version import VERSION

    assert triage.__version__ == importlib.metadata.version(triage.DISTRIBUTION)
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION), VERSION


def test_the_version_falls_back_to_the_source_outside_an_install() -> None:
    """A checkout with nothing installed still has a version to report."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as m, triage\n"
            "m.version = lambda name: (_ for _ in ()).throw(m.PackageNotFoundError(name))\n"
            "from triage._version import VERSION\n"
            "assert triage._resolve_version() == VERSION\n"
            "print('fell back')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fell back" in result.stdout


def test_the_package_ships_its_typing_marker() -> None:
    """E1. Their mypy override for `triage.*` exists only because this is absent."""
    root = Path(__file__).resolve().parents[2]
    assert (root / "triage" / "py.typed").is_file()

    with (root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert package_data["triage"] == ["py.typed"]
    assert "Typing :: Typed" in pyproject["project"]["classifiers"]


def test_the_license_is_declared_the_way_setuptools_77_wants_it() -> None:
    """D29. The table form and the classifier are deprecated together."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert pyproject["project"]["license"] == "MIT"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    classifiers = pyproject["project"]["classifiers"]
    assert not any(classifier.startswith("License ::") for classifier in classifiers)
    assert "setuptools>=77" in pyproject["build-system"]["requires"]


def test_the_readme_has_no_relative_links_left_to_break_on_pypi() -> None:
    """D29. PyPI serves the README from its own domain, with no repository under it.

    Every relative image and document link 404s there, which is what the
    project's front page looked like to anybody arriving from `pip`.
    """
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    targets = [match.group(2) for match in re.finditer(r'(src|href)="([^"]+)"', readme)]
    targets += re.findall(r"\]\(([^)]+)\)", readme)

    relative = sorted(
        {
            target
            for target in targets
            if not target.startswith(("https://", "http://", "#", "mailto:"))
        }
    )
    assert relative == [], relative
