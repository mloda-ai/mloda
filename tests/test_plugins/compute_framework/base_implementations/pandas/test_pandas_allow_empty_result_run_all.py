"""End-to-end ``allow_empty_result`` policy test for pandas.

Consumes the shared EmptyResultRunAllTestBase. Pandas is a guaranteed dependency,
so no import guard is needed; the subclass only selects the framework by name.
"""

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)


class TestPandasAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on pandas."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PandasDataFrame"
