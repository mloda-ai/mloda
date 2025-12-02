import os
from typing import Any, Optional, Type
import pytest
from unittest.mock import patch
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None


class TestDuckDBFrameworkAvailability:
    @patch("builtins.__import__")
    def test_is_available_when_duckdb_not_installed(self, mock_import: Any) -> None:
        """Test that is_available() returns False when duckdb import fails."""

        def side_effect(name: Any, *args: Any, **kwargs: Any) -> Any:
            if name == "duckdb":
                raise ImportError("No module named 'duckdb'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect
        assert DuckDBFramework.is_available() is False


class TestDuckDBInstallation:
    @pytest.mark.skipif(
        os.getenv("SKIP_DUCKDB_INSTALLATION_TEST", "false").lower() == "true",
        reason="DuckDB installation test is disabled by environment variable",
    )
    def test_duckdb_is_installed(self) -> None:
        """Test that DuckDB is properly installed and can be imported."""
        try:
            import duckdb
            import pyarrow as pa

            # Test basic functionality
            conn = duckdb.connect()
            data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            arrow_table = pa.Table.from_pydict(data)
            relation = conn.from_arrow(arrow_table)
            result = relation.df()
            assert len(result) == 3
            assert list(result.columns) == ["a", "b"]
        except ImportError:
            pytest.fail("DuckDB is not installed but is required for this test environment")


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckDBFrameworkComputeFramework:
    if duckdb:
        duckdb_framework = DuckDBFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}

        # Create expected data using DuckDB
        import pyarrow as pa

        expected_arrow = pa.Table.from_pydict(dict_data)
        conn = duckdb.connect()
        expected_data = conn.from_arrow(expected_arrow)

        # Test data for merges
        left_arrow = pa.Table.from_pydict({"idx": [1, 3], "col1": ["a", "b"]})
        right_arrow = pa.Table.from_pydict({"idx": [1, 2], "col2": ["x", "z"]})
        left_data = conn.from_arrow(left_arrow)
        right_data = conn.from_arrow(right_arrow)
        idx = Index(("idx",))

    def test_expected_data_framework(self) -> None:
        assert self.duckdb_framework.expected_data_framework() == duckdb.DuckDBPyRelation

    def test_transform_dict_to_relation(self) -> None:
        self.duckdb_framework.set_framework_connection_object(self.conn)
        result = self.duckdb_framework.transform(self.dict_data, set())
        result_df = result.df()
        expected_df = self.expected_data.df()

        # Compare the dataframes
        assert result_df.equals(expected_df)

    def test_transform_invalid_data(self) -> None:
        with pytest.raises(ValueError):
            self.duckdb_framework.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self) -> None:
        # Note: This test might need adjustment based on actual DuckDB relation API
        # For now, we'll test the basic functionality
        data = self.duckdb_framework.select_data_by_column_names(self.expected_data, {FeatureName("column1")})
        assert "column1" in data.columns

    def test_set_column_names(self) -> None:
        self.duckdb_framework.data = self.expected_data
        self.duckdb_framework.set_column_names()
        assert "column1" in self.duckdb_framework.column_names
        assert "column2" in self.duckdb_framework.column_names


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckDBFrameworkMerge(DataFrameTestBase):
    """Test DuckDBFramework merge operations using the base test class."""

    @classmethod
    def framework_class(cls) -> Type[Any]:
        """Return the DuckDBFramework class."""
        return DuckDBFramework

    def setup_method(self) -> None:
        """Set up DuckDB connection and test data."""
        self.conn = duckdb.connect()
        super().setup_method()

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        """Create a DuckDB relation from a dictionary."""
        arrow_table = pa.Table.from_pydict(data)
        return self.conn.from_arrow(arrow_table)

    def get_connection(self) -> Optional[Any]:
        """Return DuckDB connection object."""
        return self.conn

    def _create_test_framework(self) -> Any:
        """Create a framework instance with sync mode and DuckDB connection."""
        framework = super()._create_test_framework()
        framework.set_framework_connection_object(self.conn)
        return framework

    def _get_merge_engine(self, framework: Any) -> Any:
        """Get merge engine factory that returns an instance with connection for DuckDB."""
        merge_engine_class = framework.merge_engine()
        framework_connection = framework.get_framework_connection_object()

        class MergeEngineFactory:
            def __call__(self) -> Any:
                return merge_engine_class(framework_connection)

        return MergeEngineFactory()
