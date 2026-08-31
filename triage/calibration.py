"""The measured calibration results, in one place.

These numbers are produced by `tests/statistics/test_calibration.py` and are
quoted in seven places: the README, the HTML report footer, the main report, the
debug report, the figures, the methodology and the command line output. Holding
them as data here means those seven cannot drift apart, and that updating them
after a change to the statistics is a single edit rather than a search. Nothing
downstream may hold a calibration number of its own: the LaTeX tables come
through `scripts/gen_report_assets.py` and the Markdown ones through
`scripts/render_calibration_docs.py`, both of which read this module.

`WALL_CLOCK` at the bottom is here for the same reason rather than for the same
subject: it is a measurement this project publishes, so it is owned here rather
than typed into a table that nobody re-measures.

Every value is a measurement from a deterministic, seeded run. Reproduce them
with `make test-stats`, which takes about two minutes.
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


# Names are kept short because this table is typeset into a fixed width page as
# well as rendered as Markdown. The detail that would have gone into a longer
# name lives in `sample` instead.
GATES: tuple[Gate, ...] = (
    Gate(
        "Type I error, seed replicated",
        "4.53 percent (+/- 0.74)",
        "3000 null cases",
        "[2, 8] percent",
    ),
    Gate(
        "Type I error across seed variance",
        "4.40, 4.70, 5.00 percent",
        "sigma 0.005 to 0.05",
        "[2, 8] percent",
    ),
    Gate(
        "Power, seed replicated",
        "96.25 percent",
        "800 cases",
        "above 90 percent",
    ),
    Gate(
        "Type I error, window block",
        "6.28 percent (+/- 1.07)",
        "1989 cases, 11 refused",
        "[2, 8] percent",
    ),
    Gate(
        "Power, window block",
        "100 percent",
        "300 cases",
        "above 90 percent",
    ),
    Gate(
        "Type I error, paired clustered",
        "4.90 percent (+/- 1.34)",
        "1000 cases, 10 clusters of 4 pairs",
        "[2, 8] percent",
    ),
    Gate(
        "Null p value uniformity",
        "0.0580, 0.0993, 0.2407, 0.4847",
        "1500 cases",
        "within 3 standard errors",
    ),
)

# The designs that break plain exchangeability, measured cell by cell rather
# than assumed away. Three gates rather than one, because the three regimes
# differ: equal spreads and heavy tails are exact and keep [2, 8]; unequal
# spreads with five or more runs a side are asymptotically right and get
# [2, 10]; unequal spreads with fewer than five runs on one side get [1, 15],
# set to catch the defect rather than to assert an exactness the design cannot
# deliver, since permuting the raw mean difference measured 17.92 percent in
# that cell. The tool warns on exactly that design.
DESIGN_ARMS: tuple[Gate, ...] = (
    Gate(
        "Unequal spread, 5 narrow against 5 wide",
        "7.83 percent (+/- 1.52)",
        "1200 null cases, sigma 0.01 against 0.05",
        "[2, 10] percent",
    ),
    Gate(
        "Unequal spread, 3 narrow against 7 wide",
        "2.08 percent (+/- 0.81)",
        "1200 null cases, sigma 0.01 against 0.05",
        "[1, 15] percent",
    ),
    Gate(
        "Unequal spread, 7 narrow against 3 wide",
        "12.92 percent (+/- 1.90)",
        "1200 null cases, sigma 0.01 against 0.05",
        "[1, 15] percent",
    ),
    Gate(
        "Unequal counts only, 7 against 3",
        "4.25 percent (+/- 1.14)",
        "1200 null cases, one spread",
        "[2, 8] percent",
    ),
    Gate(
        "Heavy tailed noise, Student t at 3 df",
        "5.00 percent (+/- 1.23)",
        "1200 null cases, 5 runs a side",
        "[2, 8] percent",
    ),
    Gate(
        "Paired clustered, macro F1",
        "2.00 percent (+/- 1.23)",
        "500 null cases, a discrete statistic",
        "at most 8 percent",
    ),
    Gate(
        "Paired, clustering ignored",
        "12.10 percent",
        "1000 null cases, the same clustered data",
        "measured, not gated",
    ),
)

# The rate at which the raw mean difference, permuted, rejected true nulls in
# the same three heteroscedastic cells before the statistic was studentized.
# Kept beside the measurements above because a fix is only worth what it
# changed, and because the middle cell shows how little it changed there.
UNSTUDENTIZED_DESIGN_ARMS: tuple[tuple[str, float, float], ...] = (
    ("5 narrow against 5 wide", 0.0825, 0.0783),
    ("3 narrow against 7 wide", 0.0108, 0.0208),
    ("7 narrow against 3 wide", 0.1792, 0.1292),
)

# False positive rates when the true effect is exactly zero and the runs carry
# seed to seed variance. This is the measurement that justifies labelling the
# single run mode as a weaker claim everywhere it appears. Both columns come out
# of one arm at the same three seed deviations, so the two modes are compared on
# the axis the table is indexed by.
# (seed standard deviation, single run mode, seed replicated mode), as fractions.
WEAK_MODE_COST: tuple[tuple[float, float, float], ...] = (
    (0.01, 0.5427, 0.0413),
    (0.02, 0.7663, 0.0375),
    (0.04, 0.8769, 0.0413),
)

# Rejection rate under the null at each nominal threshold, as fractions.
UNIFORMITY: tuple[tuple[float, float], ...] = (
    (0.05, 0.0580),
    (0.10, 0.0993),
    (0.25, 0.2407),
    (0.50, 0.4847),
)

#: What one full `make test-stats` costs and covers. The count is the sum of the
#: case counts named in the tables above; the wall clock is measured on the
#: machine recorded in the README.
SUITE: dict[str, int] = {
    "comparisons": 23900,
    "seconds": 110,
}

#: What every other build entry point costs, on the same machine, as
#: `(command, what it does, seconds)`. A wall clock is not an error rate, and it
#: is here for the reason the error rates are: it is a measurement this project
#: publishes, so it has exactly one source and the documents that quote it are
#: rendered from that source rather than typed. These rows were typed into the
#: README by hand until 1.1.0, and two of them still said "measured at 1.0.0" a
#: release later, which is the drift this module exists to prevent.
#:
#: Re-measure at a release with `make all` from a clean tree, `make all` in a
#: fresh clone, and each session timed on its own.
WALL_CLOCK: tuple[tuple[str, str, int], ...] = (
    ("nox -s test", "the full suite at 1.1.0, 889 tests", 283),
    ("nox -s demo", "synthesise 31 runs, ingest, compare, report", 22),
    (
        "nox -s demo-autofill",
        "generate, sweep, evaluate, ingest, compare, report, at the quick sizes "
        "with the LLM step skipped",
        6,
    ),
    ("nox -s gif", "re-record the README animation in headless Chrome", 8),
    ("make all", "from a clean tree", 472),
    ("make all", "from a fresh clone, including creating the environment", 502),
)


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.0f}"


def weak_mode_range() -> str:
    """`54 to 88`: the span of the single run mode's false positive rate."""
    rates = [weak for _, weak, _ in WEAK_MODE_COST]
    return f"{_percent(min(rates))} to {_percent(max(rates))}"


def strong_mode_range() -> str:
    """The same span for the seed replicated mode, on the same data."""
    rates = [strong for _, _, strong in WEAK_MODE_COST]
    low, high = min(rates), max(rates)
    if f"{low * 100:.1f}" == f"{high * 100:.1f}":
        return f"{low * 100:.1f}"
    return f"{low * 100:.1f} to {high * 100:.1f}"


#: Short lines for the HTML report footer and the command line.
SUMMARY: dict[str, str] = {
    "Measured type I error, strong mode": "4.53 percent at a nominal 5, over 3000 null cases",
    "Measured power, strong mode": "96.25 percent on a large effect, over 800 cases",
    "Measured type I error, weak mode": "6.28 percent at a nominal 5, over 1989 null cases",
    "Cost of the weak mode": (
        f"on runs with realistic seed variance and a true effect of zero, the single run mode "
        f"fires on {weak_mode_range()} percent of comparisons; the seed replicated mode stays at "
        f"{strong_mode_range()} on the same seed variance"
    ),
    "Measured type I error, unequal spreads": (
        "7.83 percent at five runs a side, 12.92 percent at three runs of the wider condition "
        "against seven of the narrower, where the tool warns"
    ),
}
