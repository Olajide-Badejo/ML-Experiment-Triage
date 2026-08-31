"""Ranked, significance tested comparison of machine learning training runs.

    pip install ml-experiment-triage

    from triage import Store, compare_all, classify, rank, RegressionConfig

    with Store("triage.db") as store:
        results = compare_all(store.load_all(), baseline="lr0.0010_bs32")
    for finding in rank(classify(results, RegressionConfig())):
        print(finding.candidate, finding.tag, finding.verdict, finding.adjusted_p)

`dir(triage)` used to be empty. Every name below was reachable only through a
deep module path, which meant the package had a public API in practice and none
on paper: a consumer had nothing to pin to, and this package had no list it was
obliged not to break. `__all__` is that list, filed by the one downstream
project that depends on this one, and the deep import paths keep working
unchanged because those are what they already import.

**Most names resolve on first access, not on import.** `build_context` and
`render` live in a module that imports plotly and jinja2, which is 5 MB of
reporting stack that a consumer using only the statistics API never wants and,
once the packaging split lands, will not have installed. So the module hook
below imports each name's home module the first time somebody asks for it.

**`ingest` is the exception, and it is imported eagerly.** It is the one name in
`__all__` that is also the name of a submodule, and the import system binds
`triage.ingest` to the MODULE as a side effect of anybody anywhere importing it.
A lazily bound function would therefore be silently replaced by a module the
first time some other file wrote `from triage.ingest import ingest`, so
`from triage import ingest` would hand back a function or a module depending on
what had already been imported. A name that means two things depending on import
order is not a contract, so this one is resolved here, once, deterministically.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import TYPE_CHECKING, Any

from triage._version import VERSION
from triage.ingest import IngestResult, ingest

#: The distribution name, which is not the import name. `pip install` and
#: `importlib.metadata` both want this spelling.
DISTRIBUTION = "ml-experiment-triage"


def _resolve_version() -> str:
    """What is installed, or what is written in the source when nothing is.

    The installed metadata comes first because it is the version pip resolved
    and the version another package's requirement was checked against, and
    because a wheel built from this tree carries `VERSION` in that metadata:
    for a real install the two are the same string by construction. The
    fallback is for a plain checkout with nothing installed, where the source
    is the only version there is.
    """
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return VERSION


__version__ = _resolve_version()
__author__ = "Olajide Badejo"

#: `name -> the module that defines it`. The deep path is the contract and this
#: is a convenience over it, so the object handed out here IS the object at the
#: deep path: two names for two objects would be worse than one name, because a
#: consumer's isinstance check would then pass or fail by which import line they
#: happened to write.
_EXPORTS: dict[str, str] = {
    "Experiment": "triage.core.experiment",
    "MetricSeries": "triage.core.experiment",
    "Store": "triage.core.store",
    "StoreError": "triage.core.store",
    # `ingest` and `IngestResult` are bound eagerly at the top of this file; they
    # are listed here so the table stays the whole of `__all__` and the contract
    # test can hold the two in step.
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

#: The public surface, written out rather than derived from `_EXPORTS`, because
#: this is the list that was filed and a list is the kind of thing that should be
#: readable at a glance. `test_consumer_contract.py` holds the two in step.
__all__ = [
    "ComparisonConfig",
    "ComparisonResult",
    "Experiment",
    "IngestResult",
    "MetricSeries",
    "ParseError",
    "RegressionConfig",
    "Store",
    "StoreError",
    "TriageReport",
    "analyse",
    "build_context",
    "classify",
    "compare_all",
    "discover_runs",
    "ingest",
    "paired_permutation",
    "rank",
    "render",
]

if TYPE_CHECKING:
    # For type checkers and editors, which do not follow a module `__getattr__`.
    # At runtime these are the lazy lookups below; here they are the real thing,
    # so `triage.Store` resolves to the class under mypy exactly as it does under
    # the interpreter.
    from triage.analysis.comparison import (
        ComparisonConfig,
        ComparisonResult,
        compare_all,
        paired_permutation,
    )
    from triage.analysis.regression import RegressionConfig, TriageReport, classify, rank
    from triage.analysis.sensitivity import analyse
    from triage.core.experiment import Experiment, MetricSeries
    from triage.core.store import Store, StoreError
    from triage.parsers import ParseError, discover_runs
    from triage.report.html_report import build_context, render


def __getattr__(name: str) -> Any:
    """Resolve a public name to the object at its deep path, on first access."""
    module_path = _EXPORTS.get(name)
    if module_path is None:
        # Not a swallowed miss: a typo must still be an AttributeError, or the
        # lazy hook would turn every misspelling into something that looks
        # importable right up until it is called.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    # Cached on the module, so the import cost is paid once and later lookups
    # never reach this function at all.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
