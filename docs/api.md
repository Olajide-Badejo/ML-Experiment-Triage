# API reference

Rendered by mkdocstrings from the docstrings in the package, so this page cannot
drift from the code: there is one copy of each explanation and it lives next to
what it explains.

Every name here is re exported from the top level `triage` package. The deep
path shown as the heading is the canonical one and hands back the same object.

## The model

::: triage.core.experiment.Experiment

::: triage.core.experiment.MetricSeries

::: triage.core.outcomes.Outcomes

## Storage

::: triage.core.store.Store

::: triage.core.store.StoreError

## Ingest and parsing

::: triage.ingest.ingest

::: triage.ingest.IngestResult

::: triage.parsers.discover_runs

::: triage.parsers.base.Parser

::: triage.parsers.base.ParseError

::: triage.parsers.outcomes_parser.OutcomesParser

## Comparison

::: triage.analysis.comparison.ComparisonConfig

::: triage.analysis.comparison.ComparisonResult

::: triage.analysis.comparison.ComparisonResults

::: triage.analysis.comparison.ComparisonRefusal

::: triage.analysis.comparison.ComparisonError

::: triage.analysis.comparison.compare_all

::: triage.analysis.comparison.paired_permutation

## Flagging and ranking

::: triage.analysis.regression.RegressionConfig

::: triage.analysis.regression.Finding

::: triage.analysis.regression.TriageReport

::: triage.analysis.regression.classify

::: triage.analysis.regression.rank

::: triage.analysis.regression.benjamini_hochberg

## Sensitivity

::: triage.analysis.sensitivity.analyse

::: triage.analysis.sensitivity.SensitivityResult

::: triage.analysis.sensitivity.SensitivityReport

## Reporting

::: triage.report.html_report.build_context

::: triage.report.html_report.render

## The measured calibration

The single source for every error rate this project publishes. The documents
that quote these numbers are rendered from this module, and CI fails if any of
them disagrees with it.

::: triage.calibration
    options:
      members: false
