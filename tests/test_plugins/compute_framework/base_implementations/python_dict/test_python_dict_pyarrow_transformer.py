from typing import Any, Optional, Type
import pytest

from tests.test_plugins.compute_framework.test_tooling.transformer.transformer_test_base import TransformerTestBase
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer import (
    PythonDictPyarrowTransformer,
)

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not available")
class TestPythonDictPyarrowTransformer(TransformerTestBase):
    """Test Python Dict PyArrow transformer using base test class."""

    @classmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the Python Dict transformer class."""
        return PythonDictPyarrowTransformer

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return Python Dict's framework type."""
        return list

    def get_connection(self) -> Optional[Any]:
        """Return None as Python Dict doesn't need a connection."""
        return None

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

        pyarrow_data = PythonDictPyarrowTransformer.transform_fw_to_other_fw(mixed_data)
        result = PythonDictPyarrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        assert result == mixed_data

    def test_transform_with_nested_structures(self) -> None:
        """Test transformation with nested data structures."""
        nested_data = [
            {"id": 1, "metadata": {"tags": ["python", "data"], "count": 5}},
            {"id": 2, "metadata": {"tags": ["arrow", "table"], "count": 3}},
        ]

        pyarrow_data = PythonDictPyarrowTransformer.transform_fw_to_other_fw(nested_data)
        result = PythonDictPyarrowTransformer.transform_other_fw_to_fw(pyarrow_data)

        assert result == nested_data

    def test_transform_with_inconsistent_schemas(self) -> None:
        """Test transformation with inconsistent column schemas should raise an error."""
        inconsistent_data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "New York"},
        ]

        with pytest.raises(ValueError, match="Inconsistent schema at index 1"):
            PythonDictPyarrowTransformer.transform_fw_to_other_fw(inconsistent_data)

    def test_empty_table(self) -> None:
        """Test handling empty tables - Python Dict cannot preserve schema."""
        empty_list: list[dict[str, Any]] = []
        result = PythonDictPyarrowTransformer.transform_fw_to_other_fw(empty_list)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 0
        assert result.num_columns == 0
