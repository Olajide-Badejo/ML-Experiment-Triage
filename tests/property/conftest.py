"""Hypothesis profiles for the property suite.

Two, because the property tests answer two different questions. **`fast`** is
what runs by default and in the pull request job: enough examples to catch the
shape of a mistake, cheap enough that nobody is tempted to skip it. **`thorough`**
is what `nox -s test_property` runs, and it is the one that goes looking.

The profile is chosen by an environment variable rather than by
`--hypothesis-profile`, because the pytest plugin loads that flag's profile in
`pytest_configure`, which runs BEFORE this file is imported during collection:
loading a profile here would then silently overrule the flag the caller passed.

`deadline` is off in both. A per example timeout measures the machine rather
than the code, and this suite runs on shared CI runners across three operating
systems where a 200 ms budget is a coin toss and not a finding.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

#: `function_scoped_fixture` is suppressed for the store round trip, which takes
#: `tmp_path`. The health check exists because a fixture is set up once per test
#: rather than once per example, so a fixture carrying DATA would leak state
#: between examples. `tmp_path` carries a directory, and the test writes a fresh
#: database into it per example, which is the documented reason to suppress it.
COMMON = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.function_scoped_fixture],
}

settings.register_profile("fast", max_examples=25, **COMMON)
settings.register_profile("thorough", max_examples=400, **COMMON)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))
