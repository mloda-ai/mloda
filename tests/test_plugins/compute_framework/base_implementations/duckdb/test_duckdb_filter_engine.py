from typing import Any, Optional, Type
import pytest

from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine
from tests.test_plugins.compute_framework.test_tooling.filter import FilterEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore
    pa = None


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBFilterEngine(FilterEngineTestBase):
    """Test DuckDBFilterEngine using shared filter test scenarios."""

    @pytest.mark.skip(reason="DuckDB cannot handle empty tables without schema")
    def test_filter_with_empty_data(self) -> None:
        """Skip empty data test for DuckDB."""
        pass

    @pytest.fixture
    def connection(self) -> Any:
        """Create a DuckDB connection for testing."""
        if duckdb is None:
            pytest.skip("DuckDB is not installed")
        return duckdb.connect()

    @classmethod
    def filter_engine_class(cls) -> Type[BaseFilterEngine]:
        """Return the DuckDBFilterEngine class."""
        return DuckDBFilterEngine

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return DuckDB Relation type."""
        if duckdb is None:
            raise ImportError("DuckDB is not installed")
        return duckdb.DuckDBPyRelation

    def get_connection(self) -> Optional[Any]:
        """DuckDB requires a connection object."""
        if not hasattr(self, "_connection"):
            if duckdb is None:
                return None
            self._connection = duckdb.connect()
        return self._connection
