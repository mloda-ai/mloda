from typing import Any, Optional

from mloda.provider import BaseTransformer

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


class PythonDictPyArrowTransformer(BaseTransformer):
    """
    Transformer for converting between PythonDict (columnar dict) and PyArrow Table.

    This transformer handles bidirectional conversion between ``dict[str, list[Any]]``
    and PyArrow Table data structures, using PyArrow's built-in methods for
    efficient conversion.
    """

    @classmethod
    def framework(cls) -> Any:
        return dict

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
        import pyarrow as pa  # noqa: F401

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform a columnar dict to a PyArrow Table.

        Args:
            data: ``dict[str, list[Any]]`` representing tabular data

        Returns:
            pa.Table: PyArrow Table representation of the data
        """
        if pa is None:
            raise ImportError("PyArrow is not installed. To be able to use this transformer, please install pyarrow.")

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        # A columnar dict of columns maps directly onto a PyArrow table. An empty column
        # ``{"col": []}`` yields a 0-row table that keeps the column name; ``{}`` -> empty table.
        return pa.table(data)

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """
        Transform a PyArrow Table to a columnar dict.

        Args:
            data: pa.Table representing tabular data

        Returns:
            dict[str, list[Any]]: columnar dict representation of the data
        """
        if pa is None:
            raise ImportError("PyArrow is not installed. To be able to use this transformer, please install pyarrow.")

        if not isinstance(data, pa.Table):
            raise ValueError(f"Expected pa.Table, got {type(data)}")

        # Use PyArrow's to_pydict method: columnar dict, schema preserved even at zero rows.
        return data.to_pydict()
