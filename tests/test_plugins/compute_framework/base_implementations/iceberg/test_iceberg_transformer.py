import pytest
from unittest.mock import Mock, patch
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer import (
    IcebergPyArrowTransformer,
)

import logging

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
class TestIcebergPyArrowTransformer:
    def test_framework_returns_iceberg_table(self) -> None:
        """Test that framework() returns IcebergTable type."""
        assert IcebergPyArrowTransformer.framework() == IcebergTable

    def test_other_framework_returns_pyarrow_table(self) -> None:
        """Test that other_framework() returns PyArrow Table type."""
        assert IcebergPyArrowTransformer.other_framework() == pa.Table

    def test_check_imports_success(self) -> None:
        """Test that check_imports returns True when dependencies are available."""
        assert IcebergPyArrowTransformer.check_imports() is True

    def test_import_fw_success(self) -> None:
        """Test that import_fw() succeeds when PyIceberg is available."""
        # Should not raise an exception
        IcebergPyArrowTransformer.import_fw()

    def test_import_other_fw_success(self) -> None:
        """Test that import_other_fw() succeeds when PyArrow is available."""
        # Should not raise an exception
        IcebergPyArrowTransformer.import_other_fw()

    def test_transform_fw_to_other_fw_success(self) -> None:
        """Test successful transformation from Iceberg table to PyArrow table."""
        # Create mock Iceberg table
        mock_iceberg_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_arrow_table = Mock(spec=pa.Table)
        mock_scan.to_arrow.return_value = mock_arrow_table
        mock_iceberg_table.scan.return_value = mock_scan

        result = IcebergPyArrowTransformer.transform_fw_to_other_fw(mock_iceberg_table)

        mock_iceberg_table.scan.assert_called_once()
        mock_scan.to_arrow.assert_called_once()
        assert result is mock_arrow_table

    def test_transform_fw_to_other_fw_invalid_input(self) -> None:
        """Test that transform_fw_to_other_fw raises ValueError for invalid input."""
        with pytest.raises(ValueError, match="Expected Iceberg table"):
            IcebergPyArrowTransformer.transform_fw_to_other_fw("not_iceberg_table")

    def test_identify_orientation_left(self) -> None:
        """Test orientation identification for Iceberg -> PyArrow transformation."""
        orientation = IcebergPyArrowTransformer.identify_orientation(IcebergTable, pa.Table)
        assert orientation == "left"

    def test_identify_orientation_right(self) -> None:
        """Test orientation identification for PyArrow -> Iceberg transformation."""
        orientation = IcebergPyArrowTransformer.identify_orientation(pa.Table, IcebergTable)
        assert orientation == "right"

    def test_identify_orientation_unsupported(self) -> None:
        """Test orientation identification for unsupported framework combinations."""
        orientation = IcebergPyArrowTransformer.identify_orientation(str, int)
        assert orientation is None

    def test_transform_left_direction(self) -> None:
        """Test transform method with left direction (Iceberg -> PyArrow)."""
        mock_iceberg_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_arrow_table = Mock(spec=pa.Table)
        mock_scan.to_arrow.return_value = mock_arrow_table
        mock_iceberg_table.scan.return_value = mock_scan

        result = IcebergPyArrowTransformer.transform(IcebergTable, pa.Table, mock_iceberg_table, None)

        assert result is mock_arrow_table


@pytest.mark.skipif(
    pyiceberg is not None and pa is not None, reason="PyIceberg and PyArrow are installed. Skipping unavailable test."
)
class TestIcebergPyArrowTransformerUnavailable:
    """Test behavior when dependencies are not available."""

    def test_framework_raises_when_not_installed(self) -> None:
        """Test that framework() raises ImportError when PyIceberg is not installed."""
        with patch(
            "mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer.IcebergTable",
            None,
        ):
            assert IcebergPyArrowTransformer.framework() == NotImplementedError

    def test_other_framework_raises_when_not_installed(self) -> None:
        """Test that other_framework() raises ImportError when PyArrow is not installed."""
        with patch("mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer.pa", None):
            assert IcebergPyArrowTransformer.other_framework() == NotImplementedError

    def test_check_imports_false_when_not_installed(self) -> None:
        """Test that check_imports returns False when dependencies are not available."""
        with patch(
            "mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer.IcebergTable",
            None,
        ):
            assert IcebergPyArrowTransformer.check_imports() is False

    def test_import_fw_raises_when_not_installed(self) -> None:
        """Test that import_fw raises ImportError when PyIceberg is not installed."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'pyiceberg'")):
            with pytest.raises(ImportError):
                IcebergPyArrowTransformer.import_fw()

    def test_import_other_fw_raises_when_not_installed(self) -> None:
        """Test that import_other_fw raises ImportError when PyArrow is not installed."""
        with patch("builtins.__import__", side_effect=ImportError("No module named 'pyarrow'")):
            with pytest.raises(ImportError):
                IcebergPyArrowTransformer.import_other_fw()
