"""The published gate table and the gates the suite runs are the same table.

`triage/calibration.py` is where every document, the report footer and the
command line read their calibration numbers from. Its seven `GATES` rows were
transcribed by hand from a run somebody did once, so the table was a claim about
this suite rather than a product of it: a gate could have been renamed, weakened
or deleted here and the report would have gone on printing the old row.

This file closes that loop. Each calibration test records its measured rate
under the published gate's name, the session writes the collection out as JSON,
and the tests below assert that the record and the published table describe the
same seven gates.

**Named to sort after `test_calibration.py`,** because pytest collects files in
sorted order and these read what that file wrote. When the calibration tests are
deselected (`pytest -m "not slow"`, the inner loop) nothing is recorded and
these skip rather than fail, which is the honest outcome: a run that did not
measure the gates has nothing to say about them.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.statistics import gates
from triage.calibration import GATES


def measured_or_skip() -> dict[str, float]:
    if not gates.MEASURED:
        pytest.skip("the calibration gates were not run, so there is nothing to compare")
    return gates.MEASURED


@pytest.mark.slow
def test_every_published_gate_was_measured_by_this_run() -> None:
    """A row in the published table that no test produces is a row nobody checks."""
    measured = measured_or_skip()
    assert sorted(measured) == sorted(gate.name for gate in GATES)


@pytest.mark.slow
def test_the_artifact_carries_the_published_table_and_the_measurements() -> None:
    measured_or_skip()
    payload = gates.payload()

    assert payload["nominal_alpha"] == 0.05
    assert [row["name"] for row in payload["gates"]] == [gate.name for gate in GATES]
    for row, gate in zip(payload["gates"], GATES, strict=True):
        assert (row["result"], row["sample"], row["gate"]) == (gate.result, gate.sample, gate.gate)
        assert row["measured"] is not None, f"{gate.name} was published but never measured"


#: The one published row whose `result` is not a rate this suite records. Its
#: result column lists the four rejection rates at the four thresholds, while
#: the number recorded for it is the largest departure from uniformity in
#: standard errors, which is what the gate ("within 3 standard errors") is about.
NOT_A_PERCENTAGE = "Null p value uniformity"


@pytest.mark.slow
def test_the_published_result_strings_are_the_numbers_this_run_measured() -> None:
    """The strongest form of the claim: the printed table IS the measurement.

    Every document this project ships prints `result` verbatim. If the suite
    now measures 5.9 percent where the table says 4.53, the table is wrong
    everywhere it appears and this is where that is found.
    """
    measured = measured_or_skip()
    for gate in GATES:
        if gate.name == NOT_A_PERCENTAGE:
            continue
        # The confidence half width after the value is not one of the numbers
        # being claimed, so it is cut before the comparison.
        published = [
            float(found) for found in re.findall(r"\d+(?:\.\d+)?", gate.result.split("(")[0])
        ]
        assert round(measured[gate.name] * 100, 2) in published, (
            f"{gate.name}: this run measured {measured[gate.name] * 100:.2f} percent, "
            f"but triage.calibration publishes {gate.result!r}"
        )


@pytest.mark.slow
def test_the_artifact_is_written_and_is_readable_json() -> None:
    """CI collects this file, so it has to exist and to parse."""
    measured_or_skip()
    written = gates.write_artifact()

    assert written is not None
    reloaded = json.loads(written.read_text(encoding="utf-8"))
    assert reloaded == gates.payload()
