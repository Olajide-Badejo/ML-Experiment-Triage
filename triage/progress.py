"""Progress reporting that is a bar on a terminal and plain lines in a log.

A tqdm bar redraws itself with carriage returns. On a terminal that is a live
bar; in a CI log or a piped file it is hundreds of lines of partial redraws that
bury whatever you were actually looking for. So the bar is drawn only when
stderr is a terminal, and otherwise the same information arrives as occasional
plain lines.

**Everything here writes to stderr, and that is not a detail.** The plain line
branch used to `print` to stdout, which is the same stream the report path and
the verdict table write their results to; piping the output of a run to a file
interleaved progress chatter with the data, verified. Progress is diagnostics,
results are data, and the two belong on different streams.

**tqdm is optional, and its absence is not an error.** This module is on the
import path of `triage.ingest`, which a core install has to be able to use
(E1), and the bar is a convenience rather than a result. So a missing tqdm
downgrades to the plain line branch instead of raising: refusing to ingest
because the progress bar is unavailable would be a strange thing to do to
somebody who asked for numbers.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any


def is_terminal() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def tqdm_class() -> type[Any] | None:
    """The tqdm class, or None when tqdm is not installed."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm


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

    bar_class = tqdm_class() if is_terminal() else None
    if bar_class is not None:
        # try/finally, because this is a generator: a consumer that raises, or
        # simply stops iterating, abandons it part way and the bar's close()
        # would never run. What that leaves behind is a half drawn bar with no
        # trailing newline, sitting on top of whatever the traceback printed
        # next. The finally runs on garbage collection of an abandoned
        # generator as well as on a normal exit, so the terminal is restored
        # either way.
        progress_bar = bar_class(
            items, desc=description, unit="run", total=total, leave=False, file=sys.stderr
        )
        try:
            for item in progress_bar:
                if label is not None:
                    progress_bar.set_postfix_str(label(item)[:38], refresh=False)
                yield item
        finally:
            progress_bar.close()
        return

    counted = total if total is not None else 0
    for index, item in enumerate(items, start=1):
        if index == 1 or index % every == 0 or index == counted:
            suffix = f" {label(item)}" if label is not None else ""
            print(f"[{description} {index}/{counted or '?'}]{suffix}", file=sys.stderr, flush=True)
        yield item
