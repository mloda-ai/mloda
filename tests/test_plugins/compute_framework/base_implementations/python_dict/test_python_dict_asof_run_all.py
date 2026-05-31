"""End-to-end ASOF (point-in-time) join integration test for PythonDict.

Consumes the shared AsofRunAllTestBase. PythonDictFramework is a guaranteed
dependency and supports SYNC + THREADING, so the subclass is hooks-only and
inherits the base SYNC+THREADING parametrization.
"""

from tests.test_plugins.compute_framework.test_tooling.asof.asof_run_all_test_base import AsofRunAllTestBase


class TestPythonDictAsofRunAll(AsofRunAllTestBase):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on PythonDict."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PythonDictFramework"
