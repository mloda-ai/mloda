"""Shared helper: run a Python subprocess with a top-level module blocked via sys.meta_path.

The blocked module defaults to pyarrow, which is what every caller in this directory needs.
``run_blocked(body, module="polars")`` blocks any other top-level library instead, so the same
subprocess machinery serves the backend import policy tests (issue #736).
"""

from __future__ import annotations

import subprocess  # nosec
import sys

# The preamble installs a meta_path finder that raises ModuleNotFoundError for any
# import of the blocked module, then confirms the module is truly blocked.
_PREAMBLE_TEMPLATE: str = """
import sys

class _BlockModule:
    def find_spec(self, name, path=None, target=None):
        if name == "{module}" or name.startswith("{module}."):
            raise ModuleNotFoundError("{module} blocked for test", name=name)
        return None

sys.meta_path.insert(0, _BlockModule())

# Defensively confirm {module} is truly blocked.
_blocked = False
try:
    import {module}  # noqa: F401
except ModuleNotFoundError:
    _blocked = True
assert _blocked, "{module} should be blocked but was importable"

"""


def blocker_preamble(module: str = "pyarrow") -> str:
    """Preamble source that makes every import of *module* raise ModuleNotFoundError."""
    return _PREAMBLE_TEMPLATE.format(module=module)


def run_blocked(
    body: str, module: str = "pyarrow", timeout: int = 30, *, also_block: tuple[str, ...] = ()
) -> "subprocess.CompletedProcess[str]":
    """Run *body* in a subprocess where *module* (and each of *also_block*) is blocked by a sys.meta_path finder.

    The returned CompletedProcess always has text-mode stdout/stderr.
    """
    preamble = "".join(blocker_preamble(name) for name in (module, *also_block))
    return subprocess.run(  # nosec
        [sys.executable, "-c", preamble + body],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
