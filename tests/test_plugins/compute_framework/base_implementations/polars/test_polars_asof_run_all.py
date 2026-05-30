"""End-to-end ASOF (point-in-time) join integration tests for polars.

Two classes: one for the eager PolarsDataFrame, one for the lazy
PolarsLazyDataFrame. Both consume the shared AsofRunAllTestBase.
"""

import pytest

from tests.test_plugins.compute_framework.test_tooling.asof.asof_run_all_test_base import AsofRunAllTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore[assignment]


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsAsofRunAll(AsofRunAllTestBase):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on eager polars."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsDataFrame"


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyAsofRunAll(AsofRunAllTestBase):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all on lazy polars."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsLazyDataFrame"
