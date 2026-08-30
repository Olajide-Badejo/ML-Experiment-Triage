"""Remove every generated artifact so `make clean && make all` starts honest.

Nothing here touches tracked sources. The demo database under `experiments/demo`
is tracked on purpose (the report compiles from it in CI), so it is rebuilt by
`make demo` rather than deleted here.

**A removal is reported only after it is checked.** `--venv` used to ask the
virtual environment's own python to delete the directory that python.exe lives
in. Windows holds a running executable open, so the tree survived with pieces
missing, `ignore_errors=True` discarded the error, and the script printed the
removal as done. The next `make env` then built on top of the wreckage. So this
script now refuses to delete the environment it is running inside, and every
other removal is verified on disk before it is claimed.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
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


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def remove(path: Path, root: Path, removed: list[str], failed: list[str]) -> None:
    """Delete `path`, then look, then say which of the two lists it belongs in."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
    # The check is the point. rmtree with ignore_errors reports nothing at all,
    # so without this the caller cannot tell a deletion from a locked file.
    if path.exists():
        failed.append(relative(path, root))
    else:
        removed.append(relative(path, root))


def venv_refusal(venv: Path) -> str | None:
    """Why `venv` cannot be deleted right now, or None when it can be.

    A python process cannot remove the environment it was launched from on
    Windows, and on Linux it leaves a running interpreter with its files gone.
    The honest answer is to say so and stop, not to try and report success.
    """
    try:
        prefix = Path(sys.prefix).resolve()
        target = venv.resolve()
    except OSError:  # pragma: no cover - a path that cannot be resolved at all
        return None
    if target != prefix and target not in prefix.parents:
        return None
    return (
        f"refusing to delete {target}: it is the environment running this script. "
        f"Deactivate it and remove the directory with your shell, or run this "
        f"script with a python from outside it."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        action="store_true",
        help="also remove the virtual environment (a full distclean)",
    )
    args = parser.parse_args(argv)

    root = ROOT
    removed: list[str] = []
    failed: list[str] = []
    for name in DIRECTORIES:
        remove(root / name, root, removed, failed)
    for pattern in GLOBS:
        for path in sorted(root.glob(pattern)):
            if ".venv" in path.parts:
                continue
            remove(path, root, removed, failed)

    refused: str | None = None
    if args.venv:
        refused = venv_refusal(root / ".venv")
        if refused is None:
            remove(root / ".venv", root, removed, failed)

    for name in removed:
        print(f"removed {name}")
    for name in failed:
        print(f"could not be removed: {name}")
    if refused is not None:
        print(refused)
    print(f"clean: {len(removed)} paths removed, {len(failed)} left behind")
    return 1 if failed or refused is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
