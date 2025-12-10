import pytest
from unittest.mock import Mock, patch
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)

import logging

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.catalog import Catalog
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None
    IcebergTable = None  # type: ignore
    Catalog = None  # type: ignore


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergFrameworkComputeFramework:
    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.iceberg_framework = IcebergFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        self.dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        self.expected_arrow_data = pa.Table.from_pydict(self.dict_data)

    def test_is_available(self) -> None:
        """Test that is_available returns True when dependencies are installed."""
        assert IcebergFramework.is_available() is True

    def test_expected_data_framework(self) -> None:
        """Test that expected_data_framework returns IcebergTable type."""
        assert self.iceberg_framework.expected_data_framework() == IcebergTable

    def test_transform_dict_to_arrow(self) -> None:
        """Test transforming dict to PyArrow table (intermediate step for Iceberg)."""
        result = self.iceberg_framework.transform(self.dict_data, set())
        assert isinstance(result, pa.Table)
        assert result.column_names == ["column1", "column2"]
        assert result.to_pydict() == self.dict_data

    def test_transform_iceberg_table_passthrough(self) -> None:
        """Test that Iceberg tables pass through unchanged."""
        mock_iceberg_table = Mock(spec=IcebergTable)
        result = self.iceberg_framework.transform(mock_iceberg_table, set())
        assert result is mock_iceberg_table

    def test_transform_invalid_data(self) -> None:
        """Test that invalid data types raise ValueError."""
        with pytest.raises(ValueError, match="Data type .* is not supported"):
            self.iceberg_framework.transform(data=["invalid"], feature_names=set())

    def test_set_framework_connection_object_catalog(self) -> None:
        """Test setting a catalog as framework connection object."""
        mock_catalog = Mock(spec=Catalog)
        mock_catalog.load_table = Mock()

        self.iceberg_framework.set_framework_connection_object(mock_catalog)
        assert self.iceberg_framework.framework_connection_object is mock_catalog

    def test_set_framework_connection_object_table(self) -> None:
        """Test setting an Iceberg table as framework connection object."""
        mock_table = Mock(spec=IcebergTable)

        self.iceberg_framework.set_framework_connection_object(mock_table)
        assert self.iceberg_framework.framework_connection_object is mock_table

    def test_set_framework_connection_object_invalid(self) -> None:
        """Test that invalid connection objects raise ValueError."""
        with pytest.raises(ValueError, match="Expected an Iceberg catalog or table"):
            self.iceberg_framework.set_framework_connection_object("invalid")

    def test_select_data_by_column_names_non_iceberg(self) -> None:
        """Test that non-Iceberg data passes through unchanged."""
        data = "not_iceberg_table"
        feature_names = {FeatureName("column1")}
        result = self.iceberg_framework.select_data_by_column_names(data, feature_names)
        assert result is data

    def test_set_column_names_iceberg_table(self) -> None:
        """Test setting column names from Iceberg table."""
        mock_table = Mock(spec=IcebergTable)
        mock_schema = Mock()
        mock_schema.column_names = ["col1", "col2", "col3"]
        mock_table.schema.return_value = mock_schema

        self.iceberg_framework.data = mock_table
        self.iceberg_framework.set_column_names()

        assert self.iceberg_framework.column_names == {"col1", "col2", "col3"}

    def test_set_column_names_no_data(self) -> None:
        """Test setting column names when no data is present."""
        self.iceberg_framework.data = None
        self.iceberg_framework.set_column_names()
        # Should not raise an error, column_names should remain empty
        assert self.iceberg_framework.column_names == set()

    def test_merge_engine_not_implemented(self) -> None:
        """Test that merge engine raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Merge functionality is not implemented"):
            self.iceberg_framework.merge_engine()

    def test_filter_engine_returns_iceberg_filter_engine(self) -> None:
        """Test that filter_engine returns IcebergFilterEngine."""
        from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_filter_engine import (
            IcebergFilterEngine,
        )

        assert self.iceberg_framework.filter_engine() == IcebergFilterEngine


@pytest.mark.skipif(
    pyiceberg is not None and pa is not None, reason="PyIceberg and PyArrow are installed. Skipping unavailable test."
)
class TestIcebergFrameworkUnavailable:
    """Test behavior when PyIceberg is not available."""

    def test_is_available_false_when_not_installed(self) -> None:
        """Test that is_available returns False when dependencies are not installed."""
        assert_unavailable_when_import_blocked(IcebergFramework, ["pyiceberg"])

    def test_expected_data_framework_raises_when_not_installed(self) -> None:
        """Test that expected_data_framework raises ImportError when PyIceberg is not installed."""
        with patch("mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework.IcebergTable", None):
            with pytest.raises(ImportError, match="PyIceberg is not installed"):
                IcebergFramework.expected_data_framework()

    def test_set_framework_connection_object_raises_when_not_installed(self) -> None:
        """Test that set_framework_connection_object raises ImportError when PyIceberg is not installed."""
        framework = IcebergFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

        with patch("mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework.Catalog", None):
            with pytest.raises(ImportError, match="PyIceberg is not installed"):
                framework.set_framework_connection_object(Mock())
