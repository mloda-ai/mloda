from typing import Any
import pytest

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer import (
    PythonDictPyArrowTransformer,
)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestPythonDictPyArrowTransformer:
    """Unit tests for the PythonDictPyArrowTransformer class (columnar dict <-> pa.Table)."""

    @pytest.fixture
    def sample_python_dict_data(self) -> Any:
        """Create sample columnar PythonDict data."""
        return {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }

    @pytest.fixture
    def sample_pyarrow_data(self) -> Any:
        """Create sample PyArrow data."""
        return pa.table({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

    def test_framework(self) -> None:
        """Test that framework returns correct type."""
        assert PythonDictPyArrowTransformer.framework() is dict

    def test_other_framework(self) -> None:
        """Test that other_framework returns correct type."""
        assert PythonDictPyArrowTransformer.other_framework() == pa.Table

    def test_import_fw(self) -> None:
        """Test that import_fw doesn't raise errors."""
        PythonDictPyArrowTransformer.import_fw()  # Should not raise

    def test_import_other_fw(self) -> None:
        """Test that import_other_fw doesn't raise errors."""
        PythonDictPyArrowTransformer.import_other_fw()  # Should not raise

    def test_transform_fw_to_other_fw(self, sample_python_dict_data: Any) -> None:
        """Test transformation from columnar PythonDict to PyArrow."""
        result = PythonDictPyArrowTransformer.transform_fw_to_other_fw(sample_python_dict_data)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 3
        assert result.num_columns == 3
        assert set(result.column_names) == {"id", "name", "age"}

        # Check data integrity
        assert result.to_pydict() == sample_python_dict_data

    def test_transform_other_fw_to_fw(self, sample_pyarrow_data: Any) -> None:
        """Test transformation from PyArrow to columnar PythonDict."""
        result = PythonDictPyArrowTransformer.transform_other_fw_to_fw(sample_pyarrow_data)

        assert isinstance(result, dict)
        assert result == {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}

    def test_roundtrip_transformation(self, sample_python_dict_data: Any) -> None:
        """Test that PythonDict -> PyArrow -> PythonDict preserves data."""
        pyarrow_data = PythonDictPyArrowTransformer.transform_fw_to_other_fw(sample_python_dict_data)
        result = PythonDictPyArrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        assert result == sample_python_dict_data

    def test_transform_empty_dict(self) -> None:
        """Test transformation of the schema-less empty dict."""
        result = PythonDictPyArrowTransformer.transform_fw_to_other_fw({})

        assert isinstance(result, pa.Table)
        assert result.num_rows == 0
        assert result.num_columns == 0

    def test_transform_zero_row_column_keeps_schema(self) -> None:
        """A zero-row column ``{"col": []}`` yields a 0-row table that keeps the column."""
        result = PythonDictPyArrowTransformer.transform_fw_to_other_fw({"col": []})

        assert isinstance(result, pa.Table)
        assert result.num_rows == 0
        assert result.column_names == ["col"]

    def test_transform_empty_table(self) -> None:
        """Test transformation of empty PyArrow table."""
        empty_table = pa.table({})
        result = PythonDictPyArrowTransformer.transform_other_fw_to_fw(empty_table)

        assert isinstance(result, dict)
        assert result == {}

    def test_transform_fw_to_other_fw_invalid_type(self) -> None:
        """Test that invalid input type raises error."""
        with pytest.raises(ValueError, match="Expected dict, got"):
            PythonDictPyArrowTransformer.transform_fw_to_other_fw("invalid")

    def test_transform_other_fw_to_fw_invalid_type(self) -> None:
        """Test that invalid PyArrow input type raises error."""
        with pytest.raises(ValueError, match="Expected pa.Table, got"):
            PythonDictPyArrowTransformer.transform_other_fw_to_fw("invalid")

    def test_transform_with_mixed_types(self) -> None:
        """Test transformation with mixed data types."""
        mixed_data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.2, None],
            "active": [True, False, True],
        }

        pyarrow_data = PythonDictPyArrowTransformer.transform_fw_to_other_fw(mixed_data)
        result = PythonDictPyArrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        assert result == mixed_data

    def test_transform_with_nested_structures(self) -> None:
        """Test transformation with nested data structures."""
        nested_data = {
            "id": [1, 2],
            "metadata": [
                {"tags": ["python", "data"], "count": 5},
                {"tags": ["arrow", "table"], "count": 3},
            ],
        }

        pyarrow_data = PythonDictPyArrowTransformer.transform_fw_to_other_fw(nested_data)
        result = PythonDictPyArrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        assert result == nested_data
