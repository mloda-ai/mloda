"""End-to-end ``allow_empty_result`` policy test for PythonDict.

Consumes the shared EmptyResultRunAllTestBase. PythonDict is selected by name; no
connection is required. The shared empty-producing FeatureGroups emit a columnar dict
with one empty column, which transforms to a zero-row list on the PythonDict framework.
"""

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)


class TestPythonDictAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on PythonDict."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PythonDictFramework"
