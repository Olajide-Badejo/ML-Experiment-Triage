"""Write the calibration run's gate record out at the end of the session."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.statistics.gates import write_artifact


@pytest.fixture(scope="session", autouse=True)
def emit_gates_artifact() -> Iterator[None]:
    """Serialise what the calibration tests measured, once, after they finish.

    Teardown of a session fixture rather than `pytest_sessionfinish`, because a
    fixture is ordinary code that the suite can also call directly, and because
    a run that measured nothing writes nothing rather than an empty table.
    """
    yield
    write_artifact()
