from typing import Any, Optional, Type
import pytest

from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine
from tests.test_plugins.compute_framework.test_tooling.filter import FilterEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasFilterEngine(FilterEngineTestBase):
    """Test PandasFilterEngine using shared filter test scenarios."""

    @classmethod
    def filter_engine_class(cls) -> Type[BaseFilterEngine]:
        """Return the PandasFilterEngine class."""
        return PandasFilterEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return pandas DataFrame type."""
        if pd is None:
            raise ImportError("Pandas is not installed")
        dataframe_type: Type[Any] = pd.DataFrame
        return dataframe_type

    def get_connection(self) -> Optional[Any]:
        """Pandas does not require a connection object."""
        return None
