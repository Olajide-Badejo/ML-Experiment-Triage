"""The measured calibration results, in one place.

These numbers are produced by `tests/statistics/test_calibration.py` and are
quoted in five places: the README, the HTML report footer, the main report, the
figures, and the command line output. Holding them as data here means those five
cannot drift apart, and that updating them after a change to the statistics is a
single edit rather than a search.

Every value is a measurement from a deterministic, seeded run. Reproduce them
with `make test-stats`, which takes about a minute.
"""

from __future__ import annotations

from dataclasses import dataclass

NOMINAL_ALPHA = 0.05
TYPE_ONE_GATE = (0.02, 0.08)
POWER_GATE = 0.90


@dataclass(frozen=True)
class Gate:
    """One measured error rate against the gate it had to clear."""

    name: str
    result: str
    sample: str
    gate: str


GATES: tuple[Gate, ...] = (
    Gate(
        "Type I error, seed replicated mode",
        "4.53 percent (+/- 0.74)",
        "3000 null cases",
        "inside [2, 8] at a nominal 5",
    ),
    Gate(
        "Type I error, holding across seed variance 0.005 to 0.05",
        "4.40, 4.70, 5.00 percent",
        "1000 cases each",
        "inside [2, 8]",
    ),
    Gate(
        "Power, seed replicated mode, large effect",
        "96.25 percent",
        "800 cases",
        "above 90 percent",
    ),
    Gate(
        "Type I error, single run window block mode",
        "5.81 percent (+/- 1.04)",
        "1928 cases, 72 refused",
        "inside [2, 8]",
    ),
    Gate(
        "Power, single run mode, large effect",
        "100 percent",
        "291 cases",
        "above 90 percent",
    ),
    Gate(
        "Null p values uniform at 0.05, 0.10, 0.25, 0.50",
        "0.058, 0.099, 0.241, 0.485",
        "1500 cases",
        "within 3 standard errors",
    ),
)

# False positive rates when the true effect is exactly zero and the runs carry
# seed to seed variance. This is the measurement that justifies labelling the
# single run mode as a weaker claim everywhere it appears.
# (seed standard deviation, single run mode, seed replicated mode), as fractions.
WEAK_MODE_COST: tuple[tuple[float, float, float], ...] = (
    (0.01, 0.532, 0.044),
    (0.02, 0.770, 0.047),
    (0.04, 0.874, 0.050),
)

# Rejection rate under the null at each nominal threshold, as fractions.
UNIFORMITY: tuple[tuple[float, float], ...] = (
    (0.05, 0.058),
    (0.10, 0.099),
    (0.25, 0.241),
    (0.50, 0.485),
)

#: Short lines for the HTML report footer and the command line.
SUMMARY: dict[str, str] = {
    "Measured type I error, strong mode": "4.53 percent at a nominal 5, over 3000 null cases",
    "Measured power, strong mode": "96.25 percent on a large effect, over 800 cases",
    "Measured type I error, weak mode": "5.81 percent at a nominal 5, over 1928 null cases",
    "Cost of the weak mode": (
        "on runs with realistic seed variance and a true effect of zero, the single run mode "
        "fires on 53 to 87 percent of comparisons; the seed replicated mode stays at 4.4 to 5.0"
    ),
}
