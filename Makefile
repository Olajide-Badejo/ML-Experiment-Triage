# ML Experiment Triage build entry points.
#
# Everything in this project runs on CPU. `make all` from a clean tree builds
# the environment, lints, runs the full test suite including the statistical
# calibration gates, regenerates the demo sweep and database, and compiles both
# PDFs. No step needs manual intervention.

.PHONY: all help env lint fmt check-style test test-unit test-stats \
        test-integration demo assets html report report-debug images \
        pdfs verify-demo clean distclean

ifeq ($(OS),Windows_NT)
BASE_PY := py -3.13
PY := .venv/Scripts/python.exe
else
BASE_PY := python3
PY := .venv/bin/python
endif

DEMO_RUNS := experiments/results/demo_sweep
DEMO_DB := experiments/demo/triage.db
BASELINE := lr0.0010_bs32

help:
	@echo "Targets:"
	@echo "  env             create .venv and install pinned requirements"
	@echo "  lint            ruff format check and ruff lint"
	@echo "  check-style     dash guard over sources and compiled PDFs"
	@echo "  test            full suite including calibration (slowest step)"
	@echo "  demo            regenerate the synthetic sweep, database and HTML report"
	@echo "  assets          regenerate report figures and tables from the database"
	@echo "  calibration-docs render the measured calibration numbers into the Markdown"
	@echo "  pdfs            compile the main and debug reports"
	@echo "  images          regenerate the landing page charts (needs kaleido)"
	@echo "  all             everything above, in order"
	@echo "  clean           remove generated artifacts"

all:
	$(MAKE) env
	$(MAKE) lint
	$(MAKE) check-style
	$(MAKE) test
	$(MAKE) demo
	$(MAKE) verify-demo
	$(MAKE) assets
	$(MAKE) check-calibration-docs
	$(MAKE) pdfs
	$(MAKE) check-style
	@echo ""
	@echo "make all complete."

env:
	$(BASE_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps
	$(PY) -c "import sys; print('environment ready on Python', sys.version.split()[0])"

lint:
	$(PY) -m ruff format --check .
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

check-style:
	$(PY) scripts/check_no_dashes.py .

test:
	$(PY) -m pytest tests -v

test-unit:
	$(PY) -m pytest tests/unit -v

test-stats:
	$(PY) -m pytest tests/statistics -v

test-integration:
	$(PY) -m pytest tests/integration -v

demo:
	$(PY) -m examples.demo_workflow

verify-demo:
	$(PY) -m examples.demo_workflow --verify-committed

assets:
	$(PY) scripts/gen_report_assets.py --database $(DEMO_DB)

# The Markdown documents quote the same measured numbers as the report tables.
# Both sides render from triage/calibration.py, so neither can be edited alone.
calibration-docs:
	$(PY) scripts/render_calibration_docs.py

check-calibration-docs:
	$(PY) scripts/render_calibration_docs.py --check

html:
	$(PY) -m triage.cli report --database $(DEMO_DB) --baseline $(BASELINE) \
	    --output experiments/results/triage_report.html

report:
	$(PY) scripts/build_pdf.py report/main.tex --bibtex

report-debug:
	$(PY) scripts/build_pdf.py report_debug/debug_report.tex

pdfs:
	$(MAKE) report
	$(MAKE) report-debug

# Regenerating the landing page charts needs kaleido for static image export.
# That is a development extra rather than a runtime dependency, so it is
# installed here on demand and `images` is not part of `make all`. The images
# themselves are committed.
images:
	$(PY) -m pip install --quiet kaleido
	$(PY) scripts/make_result_images.py

clean:
	$(PY) scripts/clean.py

distclean:
	$(PY) scripts/clean.py --venv
