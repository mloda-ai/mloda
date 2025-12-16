from typing import Any, Optional, Type
import pytest

from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyMergeEngine(MultiIndexMergeEngineTestBase):
    """Test PolarsLazyMergeEngine using shared multi-index test scenarios."""

    @classmethod
    def merge_engine_class(cls) -> Type[BaseMergeEngine]:
        """Return the PolarsLazyMergeEngine class."""
        return PolarsLazyMergeEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return polars LazyFrame type."""
        if pl is None:
            raise ImportError("Polars is not installed")
        return pl.LazyFrame

    def get_connection(self) -> Optional[Any]:
        """Polars does not require a connection object."""
        return None
