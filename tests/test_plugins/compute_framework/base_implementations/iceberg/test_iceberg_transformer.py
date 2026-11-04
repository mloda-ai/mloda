import pytest
from typing import Any
from unittest.mock import patch
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
    pa = None  # type: ignore[assignment, unused-ignore]
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

    def test_transform_fw_to_other_fw_success(self, iceberg_adapter: Any) -> None:
        """Test successful transformation from Iceberg table to PyArrow table."""
        expected = pa.table({"a": [1, 2, 3], "s": ["x", "y", "z"]})
        table = iceberg_adapter.from_arrow(expected)

        result = IcebergPyArrowTransformer.transform_fw_to_other_fw(table)

        assert isinstance(result, pa.Table)
        assert result.schema == expected.schema
        assert result.to_pydict() == expected.to_pydict()

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

    def test_transform_left_direction(self, iceberg_adapter: Any) -> None:
        """Test transform method with left direction (Iceberg -> PyArrow)."""
        expected = pa.table({"a": [1, 2, 3], "s": ["x", "y", "z"]})
        table = iceberg_adapter.from_arrow(expected)

        result = IcebergPyArrowTransformer.transform(IcebergTable, pa.Table, table, None)

        assert isinstance(result, pa.Table)
        assert result.schema == expected.schema
        assert result.to_pydict() == expected.to_pydict()


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
            assert IcebergPyArrowTransformer.framework() is NotImplementedError

    def test_other_framework_raises_when_not_installed(self) -> None:
        """Test that other_framework() raises ImportError when PyArrow is not installed."""
        with patch("mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer.pa", None):
            assert IcebergPyArrowTransformer.other_framework() is NotImplementedError

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
