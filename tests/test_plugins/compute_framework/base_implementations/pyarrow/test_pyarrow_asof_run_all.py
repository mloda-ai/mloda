"""End-to-end ASOF (point-in-time) join integration test for PyArrow, on the shipped AsofRunAllSuite."""

from mloda._testing.compute_framework.run_all_suite import AsofRunAllSuite
from mloda._testing.framework_adapter import PyArrowAdapter


class TestPyArrowAsofRunAll(AsofRunAllSuite):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on PyArrow."""

    adapter_class = PyArrowAdapter
