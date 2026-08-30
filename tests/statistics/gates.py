"""What the calibration run measured, collected so it can be written down.

`triage/calibration.py` publishes a table of gates: seven named error rates,
the sample each was measured over, and the interval each had to clear. Every
document this project ships renders that table, and until now nothing connected
it to the suite. The numbers were transcribed by hand from a run somebody did
once, so the published table was a claim about the tests rather than a product
of them, and a gate could be deleted from the suite while its row went on being
printed in the report.

Each calibration test now records its measured rate here under the published
gate's name. The run writes the collection out as JSON (see `conftest.py`), and
`test_gates_artifact.py` asserts that the seven gates this project publishes are
exactly the seven the suite ran.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from triage import calibration

ROOT = Path(__file__).resolve().parents[2]

#: Where the run writes its record. Overridable so that CI can put it somewhere
#: it collects artifacts from without the suite knowing anything about CI.
ARTIFACT_ENVIRONMENT_KEY = "TRIAGE_GATES_ARTIFACT"
DEFAULT_ARTIFACT = ROOT / "experiments" / "results" / "calibration_gates.json"

#: `published gate name -> the rate this run measured for it`. Module level
#: because the tests that fill it are spread over one module and the test that
#: reads it lives in another, and a fixture would give each its own copy.
MEASURED: dict[str, float] = {}


def record_measurement(gate_name: str, value: float) -> None:
    """Record `value` as this run's measurement of the published gate `gate_name`.

    Raises immediately on a name that is not in the published table, because a
    typo here would otherwise look exactly like a gate that was never run.
    """
    published = {gate.name for gate in calibration.GATES}
    if gate_name not in published:
        raise KeyError(
            f"{gate_name!r} is not a published gate. The names come from "
            f"triage.calibration.GATES: {sorted(published)}"
        )
    MEASURED[gate_name] = float(value)


def artifact_path() -> Path:
    override = os.environ.get(ARTIFACT_ENVIRONMENT_KEY)
    return Path(override) if override else DEFAULT_ARTIFACT


def payload() -> dict[str, Any]:
    """The published table joined to what this run measured, ready to serialise.

    A gate the run did not reach carries `null`, so a partial run (the fast
    inner loop deselects everything here) is legible as partial rather than as
    a table of zeroes.
    """
    return {
        "nominal_alpha": calibration.NOMINAL_ALPHA,
        "type_one_gate": list(calibration.TYPE_ONE_GATE),
        "power_gate": calibration.POWER_GATE,
        "gates": [
            {
                "name": gate.name,
                "result": gate.result,
                "sample": gate.sample,
                "gate": gate.gate,
                "measured": MEASURED.get(gate.name),
            }
            for gate in calibration.GATES
        ],
    }


def write_artifact() -> Path | None:
    """Write the record, or nothing at all when this run measured nothing."""
    if not MEASURED:
        return None
    destination = artifact_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload(), indent=2) + "\n", encoding="utf-8")
    return destination
