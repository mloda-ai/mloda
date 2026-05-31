"""Shared helper: run a Python subprocess with pyarrow blocked via sys.meta_path."""

from __future__ import annotations

import subprocess  # nosec
import sys

# The preamble installs a meta_path finder that raises ModuleNotFoundError for any
# pyarrow import, then confirms pyarrow is truly blocked.
BLOCKER_PREAMBLE: str = """
import sys

class _BlockPyarrow:
    def find_spec(self, name, path=None, target=None):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError("pyarrow blocked for test")
        return None

sys.meta_path.insert(0, _BlockPyarrow())

# Defensively confirm pyarrow is truly blocked.
_blocked = False
try:
    import pyarrow  # noqa: F401
except ModuleNotFoundError:
    _blocked = True
assert _blocked, "pyarrow should be blocked but was importable"

"""


def run_blocked(body: str, timeout: int = 30) -> "subprocess.CompletedProcess[str]":
    """Run *body* in a subprocess where pyarrow is blocked by a sys.meta_path finder.

    The returned CompletedProcess always has text-mode stdout/stderr.
    """
    return subprocess.run(  # nosec
        [sys.executable, "-c", BLOCKER_PREAMBLE + body],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
