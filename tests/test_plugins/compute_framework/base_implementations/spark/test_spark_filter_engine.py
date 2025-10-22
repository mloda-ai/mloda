from typing import Any, Optional, Type
import pytest

from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_engine import SparkFilterEngine
from tests.test_plugins.compute_framework.test_tooling.filter import FilterEngineTestBase

# Import shared fixtures and availability flags from conftest.py
try:
    from tests.test_plugins.compute_framework.base_implementations.spark.conftest import PYSPARK_AVAILABLE, SKIP_REASON
except ImportError:
    # Fallback for when running tests directly
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import PYSPARK_AVAILABLE, SKIP_REASON  # type: ignore

import logging

logger = logging.getLogger(__name__)

# Import PySpark types for schema creation (only if available)
if PYSPARK_AVAILABLE:
    try:
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
    except ImportError:
        PYSPARK_AVAILABLE = False
        SparkSession = None


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkFilterEngine(FilterEngineTestBase):
    """Test SparkFilterEngine using shared filter test scenarios."""

    @classmethod
    def filter_engine_class(cls) -> Type[BaseFilterEngine]:
        """Return the SparkFilterEngine class."""
        return SparkFilterEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return Spark DataFrame type."""
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is not available")
        from pyspark.sql import DataFrame

        dataframe_type: Type[Any] = DataFrame
        return dataframe_type

    def get_connection(self) -> Optional[Any]:
        """Spark requires a SparkSession."""
        if not hasattr(self, "_spark_session"):
            if not PYSPARK_AVAILABLE:
                return None
            from pyspark.sql import SparkSession

            self._spark_session = (
                SparkSession.builder.appName("FilterEngineTest")
                .master("local[1]")
                .config("spark.driver.host", "localhost")
                .getOrCreate()
            )
        return self._spark_session

    @pytest.mark.skip(reason="Spark cannot handle empty DataFrames without schema in this context")
    def test_filter_with_empty_data(self) -> None:
        """Skip empty data test for Spark."""
        pass
