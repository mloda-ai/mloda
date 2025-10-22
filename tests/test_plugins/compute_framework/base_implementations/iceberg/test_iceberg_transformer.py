from typing import Any, Optional, Type
from unittest.mock import Mock
import pytest
import logging

from tests.test_plugins.compute_framework.test_tooling.transformer.transformer_test_base import TransformerTestBase
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer import (
    IcebergPyarrowTransformer,
)

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None
    IcebergTable = None  # type: ignore


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergPyarrowTransformer(TransformerTestBase):
    """Test Iceberg PyArrow transformer using base test class."""

    @classmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the Iceberg transformer class."""
        return IcebergPyarrowTransformer

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return Iceberg's framework type."""
        return IcebergTable

    def get_connection(self) -> Optional[Any]:
        """Return connection object - Iceberg doesn't need one for basic operations."""
        return None

    def test_check_imports(self) -> None:
        """Test that import checks work correctly."""
        assert IcebergPyarrowTransformer.check_imports() is True

    def test_framework_returns_iceberg_table(self) -> None:
        """Test that framework() returns IcebergTable type."""
        assert IcebergPyarrowTransformer.framework() == IcebergTable

    def test_other_framework_returns_pyarrow_table(self) -> None:
        """Test that other_framework() returns PyArrow Table type."""
        assert IcebergPyarrowTransformer.other_framework() == pa.Table

    def test_import_fw_success(self) -> None:
        """Test that import_fw() succeeds when PyIceberg is available."""
        IcebergPyarrowTransformer.import_fw()

    def test_import_other_fw_success(self) -> None:
        """Test that import_other_fw() succeeds when PyArrow is available."""
        IcebergPyarrowTransformer.import_other_fw()

    def test_transform_fw_to_other_fw_with_mock(self) -> None:
        """Test successful transformation from Iceberg table to PyArrow table using mock."""
        mock_iceberg_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_arrow_table = Mock(spec=pa.Table)
        mock_scan.to_arrow.return_value = mock_arrow_table
        mock_iceberg_table.scan.return_value = mock_scan

        result = IcebergPyarrowTransformer.transform_fw_to_other_fw(mock_iceberg_table)

        mock_iceberg_table.scan.assert_called_once()
        mock_scan.to_arrow.assert_called_once()
        assert result is mock_arrow_table

    def test_transform_fw_to_other_fw_invalid_input(self) -> None:
        """Test that transform_fw_to_other_fw raises ValueError for invalid input."""
        with pytest.raises(ValueError, match="Expected Iceberg table"):
            IcebergPyarrowTransformer.transform_fw_to_other_fw("not_iceberg_table")

    def test_identify_orientation_left(self) -> None:
        """Test orientation identification for Iceberg -> PyArrow transformation."""
        orientation = IcebergPyarrowTransformer.identify_orientation(IcebergTable, pa.Table)
        assert orientation == "left"

    def test_identify_orientation_right(self) -> None:
        """Test orientation identification for PyArrow -> Iceberg transformation."""
        orientation = IcebergPyarrowTransformer.identify_orientation(pa.Table, IcebergTable)
        assert orientation == "right"

    def test_identify_orientation_unsupported(self) -> None:
        """Test orientation identification for unsupported framework combinations."""
        orientation = IcebergPyarrowTransformer.identify_orientation(str, int)
        assert orientation is None

    def test_transform_left_direction(self) -> None:
        """Test transform method with left direction (Iceberg -> PyArrow)."""
        mock_iceberg_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_arrow_table = Mock(spec=pa.Table)
        mock_scan.to_arrow.return_value = mock_arrow_table
        mock_iceberg_table.scan.return_value = mock_scan

        result = IcebergPyarrowTransformer.transform(IcebergTable, pa.Table, mock_iceberg_table, None)

        assert result is mock_arrow_table
