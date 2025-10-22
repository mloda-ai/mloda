from typing import Any, Optional, Type
import pytest
import logging

from tests.test_plugins.compute_framework.test_tooling.transformer.transformer_test_base import TransformerTestBase
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_pyarrow_transformer import (
    DuckDBPyarrowTransformer,
)

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa

    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None
    DUCKDB_AVAILABLE = False


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBPyarrowTransformer(TransformerTestBase):
    """Test DuckDB PyArrow transformer using base test class."""

    def setup_method(self) -> None:
        """Create a DuckDB connection before each test."""
        self._connection = duckdb.connect()

    def teardown_method(self) -> None:
        """Close the DuckDB connection after each test."""
        if hasattr(self, "_connection") and self._connection is not None:
            try:
                self._connection.close()
            except Exception as e:
                logger.debug(f"Error closing DuckDB connection: {e}")

    @classmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the DuckDB transformer class."""
        return DuckDBPyarrowTransformer

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return DuckDB's framework type."""
        return duckdb.DuckDBPyRelation

    def get_connection(self) -> Optional[Any]:
        """Return the DuckDB connection object."""
        return self._connection

    def test_check_imports(self) -> None:
        """Test that import checks work correctly."""
        assert DuckDBPyarrowTransformer.check_imports() is True

    def test_connection_independence(self) -> None:
        """Test that transformations work with different connections."""
        arrow_table = pa.Table.from_pydict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        relation1 = self._connection.from_arrow(arrow_table)

        arrow_result = DuckDBPyarrowTransformer.transform_fw_to_other_fw(relation1)

        conn2 = duckdb.connect()
        relation2 = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_result, conn2)

        df1 = relation1.df()
        df2 = relation2.df()
        assert df1.equals(df2)

        conn2.close()
