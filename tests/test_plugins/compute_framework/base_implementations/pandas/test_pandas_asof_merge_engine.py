"""
Tests for PandasMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase.
"""

from typing import Any, Optional

import pytest

from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for PandasMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PandasMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pd is None:
            raise ImportError("Pandas is not installed")
        # mypy can't infer pd.DataFrame type correctly
        dataframe_type: type[Any] = pd.DataFrame
        return dataframe_type

    def get_connection(self) -> Optional[Any]:
        return None
