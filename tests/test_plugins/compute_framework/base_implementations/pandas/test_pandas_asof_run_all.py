"""End-to-end ASOF (point-in-time) join integration test for pandas, on the shipped AsofRunAllSuite."""

from mloda._testing.compute_framework.run_all_suite import AsofRunAllSuite
from mloda._testing.framework_adapter import PandasAdapter


class TestPandasAsofRunAll(AsofRunAllSuite):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on pandas."""

    adapter_class = PandasAdapter
