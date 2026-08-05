"""Remove every generated artifact so `make clean && make all` starts honest.

Nothing here touches tracked sources. The demo database under `experiments/demo`
is tracked on purpose (the report compiles from it in CI), so it is rebuilt by
`make demo` rather than deleted here.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    "experiments/results",
    "report/figures",
    "report/tables",
    "report/build",
    "report_debug/build",
    ".pytest_cache",
    ".ruff_cache",
]

GLOBS = [
    "**/__pycache__",
    "**/*.pyc",
    "**/*.egg-info",
    "report/*.aux",
    "report/*.log",
    "report/*.out",
    "report/*.toc",
    "report/*.bbl",
    "report/*.blg",
    "report_debug/*.aux",
    "report_debug/*.log",
    "report_debug/*.out",
    "report_debug/*.toc",
]


def remove(path: Path, removed: list[str]) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    removed.append(str(path.relative_to(ROOT)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        action="store_true",
        help="also remove the virtual environment (a full distclean)",
    )
    args = parser.parse_args()

    removed: list[str] = []
    for name in DIRECTORIES:
        remove(ROOT / name, removed)
    for pattern in GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if ".venv" in path.parts:
                continue
            remove(path, removed)
    if args.venv:
        remove(ROOT / ".venv", removed)

    for name in removed:
        print(f"removed {name}")
    print(f"clean: {len(removed)} paths removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
