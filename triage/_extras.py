"""Import the heavy dependencies late, and say which extra is missing.

A core install of this package is numpy and scipy. That is the shape the one
downstream consumer asked for (E1): they use the statistics layer, they read
JSONL, and they have no use for pandas, tensorboard, plotly or jinja2, which
between them are most of the install and none of the value for that use.

So the modules that need those four import them at the moment of use rather
than at import time, through the two helpers here. The difference that matters
is not the speed of the import; it is the message. Without this, a core install
that reaches the report writer fails with

    ModuleNotFoundError: No module named 'plotly'

which tells the reader that something is missing and nothing about whose fault
it is or what to do. With it, the same call says which feature wanted the
package, which extra ships it, and what to type.

`LazyModule` exists for the modules that use their dependency in a dozen
places. Binding it once at module scope, with the real import under
`TYPE_CHECKING`, keeps every call site reading as `pd.read_csv(...)` while the
import still happens on the first attribute access rather than at module load.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

#: The name to type after `pip install`. Not derived from `__package__`: the
#: distribution name and the import name differ, and the error message has to
#: carry the one that works in a shell.
DISTRIBUTION = "ml-experiment-triage"


class MissingExtraError(ImportError):
    """A dependency is absent because the extra that ships it was not installed.

    An `ImportError` on purpose, so that code which already guards an optional
    feature with `except ImportError` keeps working, and so that the traceback
    reads as what it is.
    """


def _message(module_name: str, extra: str, purpose: str) -> str:
    top_level = module_name.split(".")[0]
    return (
        f"{purpose} needs {top_level}, which is not installed. It ships in the "
        f'"{extra}" extra:\n\n'
        f'    pip install "{DISTRIBUTION}[{extra}]"\n\n'
        f"The core install is numpy and scipy only, so the statistics API and "
        f"the JSONL parser keep working without it."
    )


def require(module_name: str, extra: str, purpose: str) -> ModuleType:
    """Import `module_name`, or raise naming the extra that would supply it."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise MissingExtraError(_message(module_name, extra, purpose)) from error


class LazyModule:
    """A stand in for a module, imported on the first attribute access.

    Attribute access is the trigger rather than construction, so binding one of
    these at module scope costs nothing and a module that is never used is
    never imported. The failure surfaces at the line that first touches the
    dependency, which is the line that wanted it.
    """

    __slots__ = ("_extra", "_module", "_module_name", "_purpose")

    def __init__(self, module_name: str, extra: str, purpose: str) -> None:
        self._module_name = module_name
        self._extra = extra
        self._purpose = purpose
        self._module: ModuleType | None = None

    def __getattr__(self, name: str) -> Any:
        # Reached only for names that are not slots, so the four fields above
        # resolve normally and this never recurses on them.
        module = self._module
        if module is None:
            module = require(self._module_name, self._extra, self._purpose)
            self._module = module
        return getattr(module, name)

    def __repr__(self) -> str:
        state = "imported" if self._module is not None else "not imported yet"
        return f"<lazy module {self._module_name!r}, {state}>"
