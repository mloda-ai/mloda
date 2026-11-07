"""End-to-end ASOF (point-in-time) join integration test for PythonDict, on the shipped AsofRunAllSuite."""

from mloda._testing.compute_framework.run_all_suite import AsofRunAllSuite
from mloda._testing.framework_adapter import PythonDictAdapter


class TestPythonDictAsofRunAll(AsofRunAllSuite):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on PythonDict."""

    adapter_class = PythonDictAdapter
