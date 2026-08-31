"""Shared fixtures. Paths only: nothing here computes anything a test asserts on.

This directory is also put on the import path, so that `llm_fakes` is importable
by name from any test module. pytest happens to do the same thing while loading
this file, but a suite whose imports depend on an implementation detail of the
collector breaks the first time somebody runs one module directly, and the fix
is one explicit line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

FIXTURE_ROOT = HERE / "fixtures"


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture(scope="session")
def reference_series() -> dict[str, np.ndarray]:
    """The float32 series every committed parser fixture encodes."""
    stacked = np.load(FIXTURE_ROOT / "reference_series.npy")
    return {"train/loss": stacked[0], "val/accuracy": stacked[1]}


@pytest.fixture
def temp_database(tmp_path: Path) -> Path:
    return tmp_path / "triage.db"
