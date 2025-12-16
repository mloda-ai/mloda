import pytest
from typing import Any, Optional, Type

from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasMergeEngine(MultiIndexMergeEngineTestBase):
    """Test PandasMergeEngine using shared multi-index test scenarios."""

    @classmethod
    def merge_engine_class(cls) -> Type[BaseMergeEngine]:
        """Return the PandasMergeEngine class."""
        return PandasMergeEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return pandas DataFrame type."""
        if pd is None:
            raise ImportError("Pandas is not installed")
        # mypy can't infer pd.DataFrame type correctly
        dataframe_type: Type[Any] = pd.DataFrame
        return dataframe_type

    def get_connection(self) -> Optional[Any]:
        """Pandas does not require a connection object."""
        return None
