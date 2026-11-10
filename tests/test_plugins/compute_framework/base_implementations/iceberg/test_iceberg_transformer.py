import pytest
from typing import Any, Callable
from unittest.mock import patch
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_pyarrow_transformer import (
    normalize_arrow_types,
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


_DRIFT_TYPES: dict[str, Callable[[], tuple[Any, Any]]] = {
    "string": lambda: (pa.string(), "a"),
    "binary": lambda: (pa.binary(), b"x"),
    "list_int": lambda: (pa.list_(pa.int64()), [1, 2]),
    "list_string": lambda: (pa.list_(pa.string()), ["a"]),
    "struct_string": lambda: (pa.struct([("x", pa.string())]), {"x": "a"}),
    "map_string": lambda: (pa.map_(pa.string(), pa.string()), [("k", "v")]),
}


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergZeroRowTypeDrift:
    @pytest.mark.parametrize("factory", list(_DRIFT_TYPES.values()), ids=list(_DRIFT_TYPES))
    def test_zero_rows_and_non_empty_share_schema(
        self, iceberg_adapter: Any, factory: Callable[[], tuple[Any, Any]]
    ) -> None:
        arrow_type, value = factory()
        schema = pa.schema([("c", arrow_type)])
        full = iceberg_adapter.from_arrow(pa.table({"c": pa.array([value], type=arrow_type)}, schema=schema))
        empty = iceberg_adapter.from_arrow(schema.empty_table())

        full_arrow = IcebergPyArrowTransformer.transform_fw_to_other_fw(full)
        empty_arrow = IcebergPyArrowTransformer.transform_fw_to_other_fw(empty)

        assert empty_arrow.schema == full_arrow.schema
        assert empty_arrow.schema == schema


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestNormalizeArrowTypes:
    def test_keeps_child_field_names_and_nullability(self) -> None:
        large = pa.large_list(pa.field("element", pa.large_string(), nullable=False))
        table = pa.table({"c": pa.array([["a"]], type=large)})
        result = normalize_arrow_types(table)
        assert result.schema.field("c").type == pa.list_(pa.field("element", pa.string(), nullable=False))

    def test_keeps_keys_sorted_on_map(self) -> None:
        map_type = pa.map_(pa.large_string(), pa.large_string(), keys_sorted=True)
        result = normalize_arrow_types(pa.table({"c": pa.array([[("k", "v")]], type=map_type)}))
        assert result.schema.field("c").type == pa.map_(pa.string(), pa.string(), keys_sorted=True)

    def test_keeps_fixed_size_list_size(self) -> None:
        fixed = pa.list_(pa.large_string(), 2)
        result = normalize_arrow_types(pa.table({"c": pa.array([["a", "b"]], type=fixed)}))
        assert result.schema.field("c").type == pa.list_(pa.string(), 2)

    def test_table_without_large_types_is_returned_unchanged(self) -> None:
        table = pa.table({"c": pa.array([["a"]], type=pa.list_(pa.string()))})
        assert normalize_arrow_types(table) is table
