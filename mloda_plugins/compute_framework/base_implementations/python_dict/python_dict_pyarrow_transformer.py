from typing import Any, Optional

from mloda_core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer

try:
    import pyarrow as pa
except ImportError:
    pa = None


class PythonDictPyarrowTransformer(BaseTransformer):
    """
    Transformer for converting between PythonDict (List[Dict]) and PyArrow Table.

    This transformer handles bidirectional conversion between List[Dict[str, Any]]
    and PyArrow Table data structures, using PyArrow's built-in methods for
    efficient conversion.
    """

    @classmethod
    def framework(cls) -> Any:
        return list

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        pass

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform a List[Dict] to a PyArrow Table.

        Args:
            data: List[Dict[str, Any]] representing tabular data

        Returns:
            pa.Table: PyArrow Table representation of the data
        """
        if pa is None:
            raise ImportError("PyArrow is not installed. To be able to use this transformer, please install pyarrow.")

        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")

        if not data:
            # Handle empty list case
            return pa.table({})

        # Verify all items are dictionaries and have consistent schema
        if data:
            first_keys = set(data[0].keys())
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Expected dict at index {i}, got {type(item)}")

                item_keys = set(item.keys())
                if item_keys != first_keys:
                    missing_keys = first_keys - item_keys
                    extra_keys = item_keys - first_keys
                    error_msg = f"Inconsistent schema at index {i}."
                    if missing_keys:
                        error_msg += f" Missing keys: {missing_keys}."
                    if extra_keys:
                        error_msg += f" Extra keys: {extra_keys}."
                    raise ValueError(error_msg)

        # Use PyArrow's from_pylist method for efficient conversion
        return pa.Table.from_pylist(data)

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """
        Transform a PyArrow Table to a List[Dict].

        Args:
            data: pa.Table representing tabular data

        Returns:
            List[Dict[str, Any]]: List of dictionaries representation of the data
        """
        if pa is None:
            raise ImportError("PyArrow is not installed. To be able to use this transformer, please install pyarrow.")

        if not isinstance(data, pa.Table):
            raise ValueError(f"Expected pa.Table, got {type(data)}")

        # Use PyArrow's to_pylist method for efficient conversion
        return data.to_pylist()
