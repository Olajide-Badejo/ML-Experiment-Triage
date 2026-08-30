"""The canonical way to run everything in this project, on any platform.

GNU Make is not on stock Windows, and the Makefile only worked there by
accident: Make's direct exec fast path runs a single command recipe without a
shell, so `.venv/Scripts/python.exe` with forward slashes happened to launch.
Any recipe that gains a pipe or an `&&` routes through cmd.exe instead, where
that same path does not resolve (verified). A build entry point that breaks the
first time somebody adds a pipe is not an entry point, so the recipes moved
here and the Makefile became a wrapper that forwards to these sessions.

    nox -l                       list the sessions
    nox                          lint, typecheck and the full test suite
    nox -s test -- -m "not slow" the inner loop: everything but the calibration
    nox -s package               build the wheel and prove it installs and runs

**Every session but `package` runs in the project's own `.venv`** rather than in
a virtual environment nox creates, and falls back to the interpreter nox itself
is running under when there is no `.venv`, which is the shape of a CI runner.
The environment is built once by `nox -s env` from the committed PEP 751
`pylock.toml`, and the point of a lock is that every session sees the same
versions; letting nox resolve a fresh set per session would mean the tests and
the calibration gates ran against something other than what is recorded.

`package` is the exception, and has to be: it exists to prove that a wheel
installs and runs somewhere that has never seen this source tree, so it builds
throwaway environments outside the repository and installs the built artifact
into them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nox

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
DEMO_DATABASE = ROOT / "experiments" / "demo" / "triage.db"
BASELINE = "lr0.0010_bs32"

#: What `nox` with no arguments runs: the three gates that are fast enough to
#: run before every commit. The calibration suite is deliberately not here; it
#: takes about two minutes and `nox -s test_stats` is when you want it.
nox.options.sessions = ["lint", "typecheck", "test"]

#: Sessions below declare this rather than repeating the incantation. It means
#: "run in the environment that is already here", which for this project is
#: `.venv`.
IN_PROJECT_VENV = {"venv_backend": "none"}


def venv_python(venv: Path) -> Path:
    """The interpreter inside `venv`, spelled the way this platform spells it."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def python(session: nox.Session) -> str:
    """The project interpreter: `.venv` if there is one, else the running one.

    Locally there is a `.venv` and it is what every session should use. On a CI
    runner the environment IS the interpreter, built by the setup step from the
    same pinned requirements, and insisting on a `.venv` there would mean
    building a second one inside the first for no reason. The chosen path is
    logged, because a session that ran against something other than what the
    reader expected is worth being able to see.
    """
    interpreter = venv_python(VENV)
    if interpreter.exists():
        return str(interpreter)
    session.log(f"no {interpreter}; running against {sys.executable}")
    return sys.executable


def run(session: nox.Session, *arguments: str) -> None:
    """Run the project interpreter with `arguments`, from the repository root."""
    session.run(python(session), *arguments, external=True)


@nox.session(**IN_PROJECT_VENV)
def env(session: nox.Session) -> None:
    """Create `.venv` and install the locked toolchain plus this package.

    uv is bootstrapped with pip because pip is what a fresh interpreter has, and
    then uv installs `pylock.toml`. The lock is the PEP 751 standard format and
    pip 25.1 and later can install it too, so nothing here is uv only; uv is
    used because it is what resolves the lock in the first place and because it
    is several times faster on an environment this size.
    """
    base = sys.executable
    if not VENV.exists():
        session.run(base, "-m", "venv", str(VENV), external=True)
    interpreter = str(venv_python(VENV))
    session.run(interpreter, "-m", "pip", "install", "--upgrade", "pip", "uv", external=True)
    session.run(
        interpreter, "-m", "uv", "pip", "install", "--python", interpreter,
        "-r", "pylock.toml", external=True,
    )  # fmt: skip
    # `--no-deps`, because the lock IS the resolution and an installer would
    # otherwise be free to move something to satisfy the declared ranges as well.
    session.run(
        interpreter, "-m", "uv", "pip", "install", "--python", interpreter,
        "-e", ".", "--no-deps", external=True,
    )  # fmt: skip
    session.run(
        interpreter,
        "-c",
        "import sys; print('environment ready on Python', sys.version.split()[0])",
        external=True,
    )


@nox.session(**IN_PROJECT_VENV)
def lock(session: nox.Session) -> None:
    """Recompile `pylock.toml` from `pyproject.toml`.

    Idempotent by design: uv reads the existing lock as a preference source, so
    running this after adding a dependency changes that dependency and leaves
    the rest of the resolution where it is. That matters here more than usual,
    because the versions in the lock are the ones every published number was
    measured under, and a lock that drifted on every recompile would quietly
    change what those numbers mean.

    `nox -s lock -- --upgrade` is how to move them ON PURPOSE, and that is a
    change that owes the calibration suite a rerun.
    """
    run(
        session, "-m", "uv", "pip", "compile", "pyproject.toml",
        "--extra", "dev", "--universal", "--python-version", "3.12",
        "--format", "pylock.toml", "--output-file", "pylock.toml",
        *session.posargs,
    )  # fmt: skip


@nox.session(**IN_PROJECT_VENV)
def lint(session: nox.Session) -> None:
    """Formatting, lint, the dash guard, and the committed fixtures."""
    run(session, "-m", "ruff", "format", "--check", ".")
    run(session, "-m", "ruff", "check", ".")
    run(session, "scripts/check_no_dashes.py", ".")
    run(session, "scripts/make_fixtures.py", "--check")


@nox.session(**IN_PROJECT_VENV)
def fmt(session: nox.Session) -> None:
    """Apply the formatter and the safe lint fixes."""
    run(session, "-m", "ruff", "format", ".")
    run(session, "-m", "ruff", "check", "--fix", ".")


@nox.session(**IN_PROJECT_VENV)
def typecheck(session: nox.Session) -> None:
    """`mypy --strict` over the package. No per module exceptions for our code."""
    run(session, "-m", "mypy", "--strict", "triage")


@nox.session(**IN_PROJECT_VENV)
def test(session: nox.Session) -> None:
    """The full suite. `nox -s test -- -m "not slow"` is the inner loop."""
    run(session, "-m", "pytest", "tests", *session.posargs)


@nox.session(**IN_PROJECT_VENV)
def test_stats(session: nox.Session) -> None:
    """Only the calibration gates, with their printed measurements visible."""
    run(session, "-m", "pytest", "tests/statistics", "-v", "-s", *session.posargs)


@nox.session(**IN_PROJECT_VENV)
def test_property(session: nox.Session) -> None:
    """The property suite at the thorough Hypothesis profile.

    The suite runs in every ordinary test pass at the `fast` profile, which is
    enough examples to catch the shape of a mistake and cheap enough that nobody
    is tempted to skip it. This is the deep run, and it is worth doing after any
    change to `triage/analysis/` or to the store.

    The profile is chosen by the environment rather than by
    `--hypothesis-profile`, for the reason in `tests/property/conftest.py`.
    """
    session.run(
        python(session),
        "-m",
        "pytest",
        "tests/property",
        "-v",
        *session.posargs,
        external=True,
        env={"HYPOTHESIS_PROFILE": "thorough"},
    )


@nox.session(**IN_PROJECT_VENV)
def demo(session: nox.Session) -> None:
    """Regenerate the synthetic sweep, the demo database and the HTML report."""
    run(session, "-m", "examples.demo_workflow", *session.posargs)


@nox.session(**IN_PROJECT_VENV)
def docs(session: nox.Session) -> None:
    """Render the measured calibration numbers into the Markdown documents.

    `nox -s docs -- --check` is the CI form: it renders to memory and fails if
    what is committed differs. Compiling the two PDFs needs a TeX installation
    and stays in `make pdfs` and the `reports` CI job.
    """
    run(session, "scripts/render_calibration_docs.py", *session.posargs)


@nox.session(**IN_PROJECT_VENV)
def assets(session: nox.Session) -> None:
    """Regenerate the report figures and tables from the committed database."""
    run(session, "scripts/gen_report_assets.py", "--database", str(DEMO_DATABASE))


@nox.session(**IN_PROJECT_VENV)
def pdfs(session: nox.Session) -> None:
    """Compile the main and debug reports. Needs a TeX installation."""
    run(session, "scripts/build_pdf.py", "report/main.tex", "--bibtex")
    run(session, "scripts/build_pdf.py", "report_debug/debug_report.tex")


@nox.session(**IN_PROJECT_VENV)
def images(session: nox.Session) -> None:
    """Regenerate the landing page charts. Installs kaleido on demand.

    Static image export is a documentation tool rather than a runtime
    dependency, and the images themselves are committed, so this is not part of
    the default run.
    """
    run(session, "-m", "pip", "install", "--quiet", "kaleido")
    run(session, "scripts/make_result_images.py")


@nox.session(**IN_PROJECT_VENV)
def clean(session: nox.Session) -> None:
    """Remove every generated artifact. `nox -s clean -- --venv` for a distclean."""
    run(session, "scripts/clean.py", *session.posargs)


@nox.session(**IN_PROJECT_VENV)
def package(session: nox.Session) -> None:
    """Build the wheel, then prove it installs and runs somewhere new (D31).

    Nothing had ever built, installed or run the wheel. Every integration test
    calls `cli.main` in this process from this source tree, so the one thing an
    installed copy does differently, resolving the report template out of
    site-packages instead of out of the checkout, was untested.

    Two installs, because there are two products: `[cli]`, which has to ingest
    logs and write a report; and the bare core, which has to import and do
    statistics with numpy and scipy alone (E1).
    """
    interpreter = python(session)
    for stale in ("dist", "build"):
        shutil.rmtree(ROOT / stale, ignore_errors=True)

    session.run(interpreter, "-m", "build", external=True)
    session.run(interpreter, "-m", "twine", "check", "--strict", *_distributions(), external=True)

    wheel = next(path for path in (ROOT / "dist").iterdir() if path.suffix == ".whl")
    # Outside the repository on purpose: an environment inside the tree can
    # import `triage` from the checkout and prove nothing about the wheel.
    workspace = Path(tempfile.mkdtemp(prefix="triage-package-"))
    try:
        _smoke_full_install(session, interpreter, wheel, workspace / "cli")
        _smoke_core_install(session, interpreter, wheel, workspace / "core")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    session.log(f"wheel verified: {wheel.name}")


def _distributions() -> list[str]:
    return sorted(str(path) for path in (ROOT / "dist").iterdir() if path.is_file())


def _fresh_venv(session: nox.Session, interpreter: str, target: Path, install: str) -> Path:
    session.run(interpreter, "-m", "venv", str(target), external=True)
    fresh = str(venv_python(target))
    session.run(fresh, "-m", "pip", "install", "--quiet", "--upgrade", "pip", external=True)
    session.run(fresh, "-m", "pip", "install", "--quiet", install, external=True)
    return Path(fresh)


def _smoke_full_install(session: nox.Session, interpreter: str, wheel: Path, target: Path) -> None:
    """`pip install ml-experiment-triage[cli]`, then use it as a user would."""
    fresh = _fresh_venv(session, interpreter, target, f"{wheel}[cli]")
    triage = fresh.parent / ("triage.exe" if sys.platform == "win32" else "triage")

    version = subprocess.run(
        [str(triage), "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    session.log(f"installed console script reports: {version}")

    database = target / "smoke.db"
    session.run(
        str(triage),
        "ingest",
        str(ROOT / "tests" / "fixtures" / "jsonl"),
        "--database",
        str(database),
        "--quiet",
        external=True,
    )
    if not database.exists():
        session.error("the installed `triage ingest` wrote no database")

    report = target / "report.html"
    session.run(str(triage), "demo", "--output", str(report), "--quiet", external=True)
    if not report.exists():
        session.error("the installed `triage demo` wrote no report")
    written = report.stat().st_size
    # The Plotly runtime alone is megabytes, so anything small means the
    # template rendered but the figures did not, which is exactly the failure
    # mode of a template resolved from the wrong place.
    if written < 1_000_000:
        session.error(f"the report from the installed wheel is only {written} bytes")
    session.log(f"installed wheel wrote a {written // 1024} KiB report")


def _smoke_core_install(session: nox.Session, interpreter: str, wheel: Path, target: Path) -> None:
    """`pip install ml-experiment-triage` with no extras: the consumer's install."""
    fresh = _fresh_venv(session, interpreter, target, str(wheel))
    installed = json.loads(
        subprocess.run(
            [str(fresh), "-m", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    names = {package["name"].lower() for package in installed}
    unwanted = names & {"pandas", "plotly", "jinja2", "tensorboard", "tqdm"}
    if unwanted:
        session.error(f"a core install pulled in {sorted(unwanted)}")

    session.run(
        str(fresh),
        "-c",
        (
            "import numpy as np\n"
            "import triage\n"
            "from triage import paired_permutation\n"
            "assert triage.__version__\n"
            "baseline = np.array([0.80, 0.82, 0.79, 0.81, 0.83, 0.80])\n"
            "candidate = np.array([0.88, 0.90, 0.87, 0.89, 0.91, 0.88])\n"
            "result = paired_permutation(baseline, candidate, statistic=np.mean)\n"
            "assert 0.0 < result.p_value <= 1.0, result.p_value\n"
            "print('core only statistics API:', round(result.p_value, 4))\n"
        ),
        external=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
