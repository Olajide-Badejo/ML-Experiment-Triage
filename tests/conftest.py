"""Shared fixtures. Paths only: nothing here computes anything a test asserts on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


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
