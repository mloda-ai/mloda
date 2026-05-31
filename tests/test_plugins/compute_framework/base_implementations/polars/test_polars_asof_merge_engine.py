"""
Tests for PolarsMergeEngine / PolarsLazyMergeEngine merge_asof (as-of join).

Two classes: one for the eager PolarsMergeEngine (pl.DataFrame), one for the
lazy PolarsLazyMergeEngine (pl.LazyFrame).
"""

from typing import Any, Optional

import pytest

from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for the eager PolarsMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PolarsMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pl is None:
            raise ImportError("Polars is not installed")
        return pl.DataFrame

    def get_connection(self) -> Optional[Any]:
        return None


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for the lazy PolarsLazyMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PolarsLazyMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pl is None:
            raise ImportError("Polars is not installed")
        return pl.LazyFrame

    def get_connection(self) -> Optional[Any]:
        return None
