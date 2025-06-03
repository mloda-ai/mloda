from typing import Any, Dict, List
import pytest

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer import (
    PythonDictPyarrowTransformer,
)


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestPythonDictPyarrowTransformer:
    """Unit tests for the PythonDictPyarrowTransformer class."""

    @pytest.fixture
    def sample_python_dict_data(self) -> Any:
        """Create sample PythonDict data."""
        return [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30},
            {"id": 3, "name": "Charlie", "age": 35},
        ]

    @pytest.fixture
    def sample_pyarrow_data(self) -> Any:
        """Create sample PyArrow data."""
        return pa.table({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

    def test_framework(self) -> None:
        """Test that framework returns correct type."""
        assert PythonDictPyarrowTransformer.framework() == list

    def test_other_framework(self) -> None:
        """Test that other_framework returns correct type."""
        assert PythonDictPyarrowTransformer.other_framework() == pa.Table

    def test_import_fw(self) -> None:
        """Test that import_fw doesn't raise errors."""
        PythonDictPyarrowTransformer.import_fw()  # Should not raise

    def test_import_other_fw(self) -> None:
        """Test that import_other_fw doesn't raise errors."""
        PythonDictPyarrowTransformer.import_other_fw()  # Should not raise

    def test_transform_fw_to_other_fw(self, sample_python_dict_data: Any) -> None:
        """Test transformation from PythonDict to PyArrow."""
        result = PythonDictPyarrowTransformer.transform_fw_to_other_fw(sample_python_dict_data)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 3
        assert result.num_columns == 3
        assert set(result.column_names) == {"id", "name", "age"}

        # Check data integrity
        result_dict = result.to_pylist()
        assert result_dict == sample_python_dict_data

    def test_transform_other_fw_to_fw(self, sample_pyarrow_data: Any) -> None:
        """Test transformation from PyArrow to PythonDict."""
        result = PythonDictPyarrowTransformer.transform_other_fw_to_fw(sample_pyarrow_data)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(row, dict) for row in result)

        expected = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30},
            {"id": 3, "name": "Charlie", "age": 35},
        ]
        assert result == expected

    def test_roundtrip_transformation(self, sample_python_dict_data: Any) -> None:
        """Test that PythonDict -> PyArrow -> PythonDict preserves data."""
        # Transform to PyArrow
        pyarrow_data = PythonDictPyarrowTransformer.transform_fw_to_other_fw(sample_python_dict_data)

        # Transform back to PythonDict
        result = PythonDictPyarrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        # Should be identical to original
        assert result == sample_python_dict_data

    def test_transform_empty_list(self) -> None:
        """Test transformation of empty list."""
        empty_list: List[Dict[str, Any]] = []
        result = PythonDictPyarrowTransformer.transform_fw_to_other_fw(empty_list)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 0
        assert result.num_columns == 0

    def test_transform_empty_table(self) -> None:
        """Test transformation of empty PyArrow table."""
        empty_table = pa.table({})
        result = PythonDictPyarrowTransformer.transform_other_fw_to_fw(empty_table)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_transform_fw_to_other_fw_invalid_type(self) -> None:
        """Test that invalid input type raises error."""
        with pytest.raises(ValueError, match="Expected list, got"):
            PythonDictPyarrowTransformer.transform_fw_to_other_fw("invalid")

    def test_transform_fw_to_other_fw_invalid_list_content(self) -> None:
        """Test that list with non-dict items raises error."""
        invalid_data = [{"id": 1}, "invalid", {"id": 3}]

        with pytest.raises(ValueError, match="Expected dict at index 1, got"):
            PythonDictPyarrowTransformer.transform_fw_to_other_fw(invalid_data)

    def test_transform_other_fw_to_fw_invalid_type(self) -> None:
        """Test that invalid PyArrow input type raises error."""
        with pytest.raises(ValueError, match="Expected pa.Table, got"):
            PythonDictPyarrowTransformer.transform_other_fw_to_fw("invalid")

    def test_transform_with_mixed_types(self) -> None:
        """Test transformation with mixed data types."""
        mixed_data = [
            {"id": 1, "name": "Alice", "score": 95.5, "active": True},
            {"id": 2, "name": "Bob", "score": 87.2, "active": False},
            {"id": 3, "name": "Charlie", "score": None, "active": True},
        ]

        # Transform to PyArrow and back
        pyarrow_data = PythonDictPyarrowTransformer.transform_fw_to_other_fw(mixed_data)
        result = PythonDictPyarrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        assert result == mixed_data

    def test_transform_with_nested_structures(self) -> None:
        """Test transformation with nested data structures."""
        nested_data = [
            {"id": 1, "metadata": {"tags": ["python", "data"], "count": 5}},
            {"id": 2, "metadata": {"tags": ["arrow", "table"], "count": 3}},
        ]

        # Transform to PyArrow and back
        pyarrow_data = PythonDictPyarrowTransformer.transform_fw_to_other_fw(nested_data)
        result = PythonDictPyarrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        assert result == nested_data

    def test_transform_with_inconsistent_schemas(self) -> None:
        """Test transformation with inconsistent column schemas should raise an error."""
        inconsistent_data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob"},  # Missing age
            {"id": 3, "name": "Charlie", "age": 35, "city": "New York"},  # Extra column
        ]

        # Should raise an error due to inconsistent schema
        with pytest.raises(ValueError, match="Inconsistent schema at index 1"):
            PythonDictPyarrowTransformer.transform_fw_to_other_fw(inconsistent_data)
