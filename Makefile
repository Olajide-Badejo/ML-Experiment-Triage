# ML Experiment Triage build entry points.
#
# **The recipes live in `noxfile.py`.** This file forwards to it and does
# nothing else. GNU Make is not on stock Windows, and where it is present it
# only runs these recipes because Make execs a single command recipe directly,
# without a shell: the moment a recipe gains a pipe or an `&&` it routes through
# cmd.exe, where `.venv/Scripts/nox.exe` with forward slashes does not resolve.
# So every recipe below is exactly one command, `make all` is a sequence of
# `$(MAKE)` calls rather than a chain, and anybody without Make runs
# `nox -s <session>` directly for the same result.
#
# Everything in this project runs on CPU. `make all` from a clean tree builds
# the environment, lints, runs the full test suite including the statistical
# calibration gates, regenerates the demo sweep and database, and compiles both
# PDFs. No step needs manual intervention.

.PHONY: all help env lock lint fmt check-style check-fixtures typecheck test \
        test-unit test-stats test-property test-integration demo verify-demo assets \
        calibration-docs check-calibration-docs html report report-debug \
        images pdfs package clean distclean

ifeq ($(OS),Windows_NT)
BASE_PY := py -3.13
PY := .venv/Scripts/python.exe
NOX := .venv/Scripts/nox.exe
else
BASE_PY := python3
PY := .venv/bin/python
NOX := .venv/bin/nox
endif

DEMO_DB := experiments/demo/triage.db
BASELINE := lr0.0010_bs32

help:
	@echo "Targets (each forwards to a nox session; run 'nox -l' for the full list):"
	@echo "  env             create .venv and install the locked toolchain"
	@echo "  lock            recompile pylock.toml from pyproject.toml"
	@echo "  lint            ruff format check, ruff lint, dash guard, fixture check"
	@echo "  typecheck       mypy --strict over the package"
	@echo "  test            full suite including calibration (slowest step)"
	@echo "  test-unit       the inner loop: everything but the calibration gates"
	@echo "  test-property   the property suite at the thorough Hypothesis profile"
	@echo "  demo            regenerate the synthetic sweep, database and HTML report"
	@echo "  assets          regenerate report figures and tables from the database"
	@echo "  calibration-docs render the measured calibration numbers into the Markdown"
	@echo "  pdfs            compile the main and debug reports"
	@echo "  images          regenerate the landing page charts (needs kaleido)"
	@echo "  package         build the wheel and prove it installs and runs"
	@echo "  all             everything above, in order"
	@echo "  clean           remove generated artifacts"

all:
	$(MAKE) env
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) demo
	$(MAKE) verify-demo
	$(MAKE) assets
	$(MAKE) check-calibration-docs
	$(MAKE) pdfs
	$(MAKE) package
	$(MAKE) lint
	@echo ""
	@echo "make all complete."

# The one recipe that cannot forward to nox, because nox lives in the
# environment this creates. Bootstrapping nox first and asking it to build its
# own interpreter would be one more moving part for no gain.
env:
	$(BASE_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip uv
	$(PY) -m uv pip install --python $(PY) -r pylock.toml
	$(PY) -m uv pip install --python $(PY) -e . --no-deps
	$(PY) -c "import sys; print('environment ready on Python', sys.version.split()[0])"

lock:
	$(NOX) -s lock

lint:
	$(NOX) -s lint

fmt:
	$(NOX) -s fmt

# Kept as their own targets because they name what they check, which is what a
# failing CI step should be called.
check-style:
	$(PY) scripts/check_no_dashes.py .

check-fixtures:
	$(PY) scripts/make_fixtures.py --check

typecheck:
	$(NOX) -s typecheck

test:
	$(NOX) -s test -- -v

test-unit:
	$(NOX) -s test -- -m "not slow" -q

test-stats:
	$(NOX) -s test_stats

test-property:
	$(NOX) -s test_property

test-integration:
	$(NOX) -s test -- tests/integration -v

demo:
	$(NOX) -s demo

verify-demo:
	$(NOX) -s demo -- --verify-committed

assets:
	$(NOX) -s assets

# The Markdown documents quote the same measured numbers as the report tables.
# Both sides render from triage/calibration.py, so neither can be edited alone.
calibration-docs:
	$(NOX) -s docs

check-calibration-docs:
	$(NOX) -s docs -- --check

html:
	$(PY) -m triage.cli report --database $(DEMO_DB) --baseline $(BASELINE) --output experiments/results/triage_report.html

report:
	$(PY) scripts/build_pdf.py report/main.tex --bibtex

report-debug:
	$(PY) scripts/build_pdf.py report_debug/debug_report.tex

pdfs:
	$(NOX) -s pdfs

images:
	$(NOX) -s images

package:
	$(NOX) -s package

clean:
	$(NOX) -s clean

distclean:
	$(NOX) -s clean -- --venv
