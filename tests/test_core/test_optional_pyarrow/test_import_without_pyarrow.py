"""Verify mloda core works without pyarrow installed.

This test launches a subprocess (in the same venv, where pyarrow IS installed)
that installs a meta_path import blocker so any ``import pyarrow`` raises
ModuleNotFoundError. Inside that subprocess we ``import mloda.user`` and run a
self-contained, single-compute-framework, in-memory, single-process end-to-end
computation using ``PythonDictFramework``.

The test always runs (no skipif): the subprocess blocks pyarrow internally, so
the result is independent of whether pyarrow is present in the environment.
"""

import os
import subprocess  # nosec
import sys

import pytest

# Set MLODA_TEST_BLOCK_PYARROW=0 in the environment to run the inline
# FeatureGroup end-to-end WITHOUT the blocker (sanity check that the inline
# FeatureGroup itself is correct in this venv where pyarrow is available).
_SCRIPT = '''
import os
import sys

if os.environ.get("MLODA_TEST_BLOCK_PYARROW", "1") == "1":
    class _BlockPyarrow:
        def find_spec(self, name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError("pyarrow blocked for test")
            return None

    sys.meta_path.insert(0, _BlockPyarrow())

    # Defensively confirm pyarrow truly cannot be imported, so a regression
    # that quietly re-imports pyarrow is caught here rather than silently
    # passing.
    _pyarrow_blocked = False
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        _pyarrow_blocked = True
    assert _pyarrow_blocked, "pyarrow should be blocked but was importable"

# The eager import path of mloda.user must not require pyarrow.
import mloda.user
from mloda.user import mloda, Feature, PluginCollector
from mloda.provider import FeatureGroup, FeatureSet, DataCreator, BaseInputData
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from typing import Any, Optional


class InlinePythonDictCreator(FeatureGroup):
    """Self-contained primary-source FeatureGroup producing PythonDict data."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"inline_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # PythonDict native format: list[dict[str, Any]] (one dict per row).
        return [{"inline_value": 1}, {"inline_value": 2}, {"inline_value": 3}]

    @classmethod
    def compute_framework_rule(cls) -> set:
        return {PythonDictFramework}


result = mloda.run_all(
    [Feature("inline_value")],
    compute_frameworks={PythonDictFramework},
    plugin_collector=PluginCollector.enabled_feature_groups({InlinePythonDictCreator}),
)

assert len(result) >= 1, f"expected at least one result, got {result!r}"
assert result[0], f"expected non-empty data, got {result[0]!r}"

print("OK")
sys.exit(0)
'''


@pytest.mark.timeout(30)
def test_import_and_run_without_pyarrow() -> None:
    """mloda.user import and a PythonDict end-to-end run must work without pyarrow."""
    result = subprocess.run(  # nosec
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env={**os.environ, "MLODA_TEST_BLOCK_PYARROW": "1"},
        timeout=30,
    )

    assert result.returncode == 0, (
        "Subprocess failed: mloda core could not import/run without pyarrow.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
