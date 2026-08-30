"""Render the measured calibration numbers into the Markdown that quotes them.

    python scripts/render_calibration_docs.py            rewrite in place
    python scripts/render_calibration_docs.py --check    fail if anything is stale

`triage/calibration.py` is the single source for every published error rate, and
`scripts/gen_report_assets.py` already renders the LaTeX side of that from it.
The Markdown side was typed by hand, which is exactly the arrangement rule 6 of
the build specification exists to forbid: the README, the methodology and the
design decisions each held their own copy of the same measurements, and after a
change to the statistics they were three chances to update two of them.

Regions are marked in the documents with HTML comments, which render as nothing:

    <!-- calibration:weakrange -->54 to 88<!-- /calibration:weakrange -->

Everything between the two markers is replaced, so a region can be a whole table
or three words inside a sentence, and a document reads normally either way. A
missing region is an error rather than a silent no-op, because a region that has
been edited away is precisely the drift this script exists to prevent.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from triage.calibration import (
    DESIGN_ARMS,
    GATES,
    SUITE,
    UNIFORMITY,
    WEAK_MODE_COST,
    strong_mode_range,
    weak_mode_range,
)

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
METHODOLOGY = ROOT / "docs" / "methodology.md"
DECISIONS = ROOT / "docs" / "DESIGN_DECISIONS.md"

BADGE_COLOUR = "1baf7a"


def headline_type_one() -> str:
    """The number the badge and the abstract lead with, from the first gate."""
    match = re.match(r"([\d.]+) percent", GATES[0].result)
    if match is None:  # pragma: no cover, the table is ours
        raise SystemExit(f"cannot read a rate out of {GATES[0].result!r}")
    return match.group(1)


def markdown_table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def headline_table() -> str:
    rows = [(gate.name, f"{gate.result}, {gate.sample}", gate.gate) for gate in GATES]
    return markdown_table(rows, ("Measurement", "Result", "Gate"))


def design_table() -> str:
    rows = [(arm.name, f"{arm.result}, {arm.sample}", arm.gate) for arm in DESIGN_ARMS]
    return markdown_table(rows, ("Design", "Type I error at a nominal 5", "Gate"))


def weak_mode_table() -> str:
    rows = [
        (f"{sigma:.2f}", f"{weak * 100:.1f} percent", f"{strong * 100:.1f} percent")
        for sigma, weak, strong in WEAK_MODE_COST
    ]
    return markdown_table(
        rows, ("Seed standard deviation", "Single run mode", "Seed replicated mode")
    )


def uniformity_table() -> str:
    rows = [(f"{threshold:.2f}", f"{measured:.4f}") for threshold, measured in UNIFORMITY]
    return markdown_table(rows, ("Nominal threshold", "Measured rejection rate"))


def badge() -> str:
    rate = headline_type_one()
    return (
        f'  <img alt="Type I error {rate} percent" '
        f'src="https://img.shields.io/badge/measured%20type%20I-{rate}%25%20vs%205%25%20nominal'
        f'-{BADGE_COLOUR}">'
    )


def weak_mode_figure() -> str:
    """The hero figure, whose alt text is itself a published claim.

    Rendered as the whole tag rather than as the text inside it: an HTML comment
    cannot mark a region inside an attribute value, and putting one there would
    have read the markers out loud to anyone using a screen reader.
    """
    return (
        f'  <img alt="False positive rate against seed variance. Comparing single runs fires on '
        f"{weak_mode_range()} percent of comparisons where the true effect is zero, while the "
        f"seed replicated comparison stays at {strong_mode_range()} percent, on the nominal 5 "
        f'percent line." src="assets/images/weak-mode-cost-light.png">'
    )


def regions() -> dict[Path, dict[str, str]]:
    """Every rendered region, by file and by name."""
    block = {
        "badge": f"\n{badge()}\n",
        "headline": f"\n{headline_table()}\n",
        "designs": f"\n{design_table()}\n",
        "weakmodetable": f"\n{weak_mode_table()}\n",
        "uniformitytable": f"\n{uniformity_table()}\n",
        "weakmodefigure": f"\n{weak_mode_figure()}\n",
    }
    inline = {
        "weakrange": weak_mode_range(),
        "strongrange": strong_mode_range(),
        "comparisons": f"{SUITE['comparisons']:,}",
        "statsclock": f"{SUITE['seconds']} s",
        "blocktypeone": (
            f"{GATES[3].result.split(' (')[0]} at a nominal 5, over {GATES[3].sample.split(',')[0]}"
        ),
    }
    everything = {**block, **inline}
    return {README: everything, METHODOLOGY: everything, DECISIONS: everything}


#: One number lives where an HTML comment cannot: inside a mermaid diagram,
#: where a comment would be drawn as part of the node label. It is matched by
#: shape instead, which is the same guarantee by a less pretty route.
MERMAID_NODE = re.compile(r"(Calibration suite<br/>)[\d,]+( synthetic comparisons)")


def render_mermaid(text: str) -> tuple[str, int]:
    return MERMAID_NODE.subn(rf"\g<1>{SUITE['comparisons']:,}\g<2>", text)


def render(text: str, name: str, body: str) -> tuple[str, int]:
    """Replace every marked instance of one region. Returns the text and a count."""
    pattern = re.compile(
        rf"(<!-- calibration:{name} -->).*?(<!-- /calibration:{name} -->)", re.DOTALL
    )
    rendered, count = pattern.subn(lambda m: m.group(1) + body + m.group(2), text)
    return rendered, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any document is out of date with triage/calibration.py",
    )
    args = parser.parse_args()

    stale: list[str] = []
    rendered_regions = 0
    for path, bodies in regions().items():
        original = path.read_text(encoding="utf-8")
        text = original
        for name, body in bodies.items():
            text, count = render(text, name, body)
            rendered_regions += count
        text, count = render_mermaid(text)
        rendered_regions += count
        if text == original:
            continue
        if args.check:
            stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(text, encoding="utf-8")
            print(f"rendered {path.relative_to(ROOT)}")

    if not rendered_regions:
        print(
            "error: no calibration regions found; the markers have been edited away",
            file=sys.stderr,
        )
        return 2
    if stale:
        print(
            "error: these files no longer match triage/calibration.py: " + ", ".join(stale),
            file=sys.stderr,
        )
        print("run scripts/render_calibration_docs.py to regenerate them", file=sys.stderr)
        return 1
    print(f"{rendered_regions} calibration regions {'checked' if args.check else 'rendered'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
