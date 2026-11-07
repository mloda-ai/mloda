"""End-to-end ASOF (point-in-time) join integration tests for eager and lazy polars, on AsofRunAllSuite."""

from mloda._testing.compute_framework.run_all_suite import AsofRunAllSuite
from mloda._testing.framework_adapter import PolarsAdapter, PolarsLazyAdapter


class TestPolarsAsofRunAll(AsofRunAllSuite):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on eager polars."""

    adapter_class = PolarsAdapter


class TestPolarsLazyAsofRunAll(AsofRunAllSuite):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on lazy polars."""

    adapter_class = PolarsLazyAdapter
