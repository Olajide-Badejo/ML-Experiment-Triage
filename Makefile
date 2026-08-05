# ML Experiment Triage build entry points.
#
# Everything in this project runs on CPU. `make all` from a clean tree builds
# the environment, lints, runs the full test suite including the statistical
# calibration gates, regenerates the demo sweep and database, and compiles all
# three PDFs. No step needs manual intervention.

.PHONY: all help env lint fmt check-style test test-unit test-stats \
        test-integration demo assets html report report-debug report-for-me \
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
BASELINE := baseline_lr0.001_seed0

help:
	@echo "Targets:"
	@echo "  env             create .venv and install pinned requirements"
	@echo "  lint            ruff format check and ruff lint"
	@echo "  check-style     dash guard over sources and compiled PDFs"
	@echo "  test            full suite including calibration (slowest step)"
	@echo "  demo            regenerate the synthetic sweep, database and HTML report"
	@echo "  assets          regenerate report figures and tables from the database"
	@echo "  pdfs            compile main, debug and personal reports"
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
	$(PY) examples/demo_workflow.py

verify-demo:
	$(PY) examples/demo_workflow.py --verify-committed

assets:
	$(PY) scripts/gen_report_assets.py --database $(DEMO_DB)

html:
	$(PY) -m triage.cli report --database $(DEMO_DB) --baseline $(BASELINE) \
	    --output experiments/results/triage_report.html

report:
	$(PY) scripts/build_pdf.py report/main.tex --bibtex

report-debug:
	$(PY) scripts/build_pdf.py report_debug/debug_report.tex

report-for-me:
	$(PY) scripts/build_pdf.py report_for_me/report_for_me.tex

pdfs:
	$(MAKE) report
	$(MAKE) report-debug
	$(MAKE) report-for-me

clean:
	$(PY) scripts/clean.py

distclean:
	$(PY) scripts/clean.py --venv
