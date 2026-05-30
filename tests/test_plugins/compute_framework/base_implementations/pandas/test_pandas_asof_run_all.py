"""End-to-end ASOF (point-in-time) join integration test for pandas.

Consumes the shared AsofRunAllTestBase. Pandas is a guaranteed dependency,
so no import guard is needed; the subclass only selects the framework by name.
"""

from tests.test_plugins.compute_framework.test_tooling.asof.asof_run_all_test_base import AsofRunAllTestBase


class TestPandasAsofRunAll(AsofRunAllTestBase):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on pandas."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PandasDataFrame"
