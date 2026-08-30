"""The one place the version number is written.

`pyproject.toml` declares `dynamic = ["version"]` and reads `VERSION` from here,
so the distribution metadata and the package agree by construction rather than
by somebody remembering to edit two files. Releasing is therefore one edit.

This module holds a literal and imports nothing, deliberately. setuptools reads
an `attr:` version out of the source with the AST when it can and imports the
module when it cannot, and the build runs in an isolated environment with no
numpy and no scipy in it, so anything importable from `triage/__init__.py`
would have to be installed to build a wheel.
"""

from __future__ import annotations

#: Bumped by the release, and by nothing else.
VERSION = "1.1.0"
