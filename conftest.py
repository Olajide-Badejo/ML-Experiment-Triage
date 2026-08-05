"""Put the repository root on the import path for the test run.

The integration tests import `examples.make_synthetic_runs`, which is a runnable
example rather than part of the installed package. An editable install happens
to make that importable, but relying on the details of how setuptools lays out
an editable install is fragile, and it would fail on a plain `pytest` in a
checkout with nothing installed. Putting the root on the path here makes the
dependency explicit and identical locally and in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
