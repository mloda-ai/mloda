"""End-to-end ``allow_empty_result`` policy tests for polars.

Two classes: one for the eager PolarsDataFrame, one for the lazy PolarsLazyDataFrame.
Both consume the shared EmptyResultRunAllTestBase.

Note for the implementer: the eager framework derives emptiness from ``height == 0``, but
a ``pl.LazyFrame`` has no height without a ``collect()``, so the lazy framework needs its
own ``_is_empty`` predicate. The lazy subclass below exercises only the end-to-end behavior
through ``run_all`` (it does not assert a native representation).
"""

import pytest

from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore[assignment]


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on eager polars."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsDataFrame"


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on lazy polars."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsLazyDataFrame"
