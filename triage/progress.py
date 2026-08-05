"""Progress reporting that is a bar on a terminal and plain lines in a log.

A tqdm bar redraws itself with carriage returns. On a terminal that is a live
bar; in a CI log or a piped file it is hundreds of lines of partial redraws that
bury whatever you were actually looking for. So the bar is drawn only when
stderr is a terminal, and otherwise the same information arrives as occasional
plain lines.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator

from tqdm import tqdm


def is_terminal() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def track[T](
    items: Iterable[T],
    description: str,
    total: int | None = None,
    enabled: bool = True,
    label: Callable[[T], str] | None = None,
    every: int = 10,
) -> Iterator[T]:
    """Iterate `items`, showing a bar on a terminal or plain lines otherwise."""
    if not enabled:
        yield from items
        return

    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None

    if is_terminal():
        bar = tqdm(items, desc=description, unit="run", total=total, leave=False)
        for item in bar:
            if label is not None:
                bar.set_postfix_str(label(item)[:38], refresh=False)
            yield item
        bar.close()
        return

    counted = total if total is not None else 0
    for index, item in enumerate(items, start=1):
        if index == 1 or index % every == 0 or index == counted:
            suffix = f" {label(item)}" if label is not None else ""
            print(f"[{description} {index}/{counted or '?'}]{suffix}", flush=True)
        yield item
