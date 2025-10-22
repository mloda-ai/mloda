from typing import Any, Optional, Type
import pytest

from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine
from tests.test_plugins.compute_framework.test_tooling.filter import FilterEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsFilterEngine(FilterEngineTestBase):
    """Test PolarsFilterEngine using shared filter test scenarios."""

    @classmethod
    def filter_engine_class(cls) -> Type[BaseFilterEngine]:
        """Return the PolarsFilterEngine class."""
        return PolarsFilterEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return Polars DataFrame type."""
        if pl is None:
            raise ImportError("Polars is not installed")
        dataframe_type: Type[Any] = pl.DataFrame
        return dataframe_type

    def get_connection(self) -> Optional[Any]:
        """Polars does not require a connection object."""
        return None
