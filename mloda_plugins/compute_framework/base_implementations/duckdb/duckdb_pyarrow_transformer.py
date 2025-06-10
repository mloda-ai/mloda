from typing import Any, Optional

from mloda_core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

try:
    import pyarrow as pa
except ImportError:
    pa = None


class DuckDBPyarrowTransformer(BaseTransformer):
    """
    Transformer for converting between DuckDB relations and PyArrow Table.

    This transformer handles bidirectional conversion between DuckDB DuckDBPyRelation
    and PyArrow Table data structures, leveraging DuckDB's native PyArrow integration
    for efficient zero-copy operations where possible.
    """

    @classmethod
    def framework(cls) -> Any:
        if duckdb is None:
            return NotImplementedError
        return duckdb.DuckDBPyRelation

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        import duckdb

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform a DuckDB relation to a PyArrow Table.

        This method uses DuckDB's native to_arrow_table() method for efficient conversion.
        """
        return data.to_arrow_table()

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """
        Transform a PyArrow Table to a DuckDB relation.

        This method uses DuckDB's native from_arrow() method for efficient conversion.
        """
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")

        # Create a connection and convert PyArrow table to DuckDB relation
        if framework_connection_object is None:
            raise ValueError("A DuckDB connection object is required for this transformation.")

        if not isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            raise ValueError(f"Expected a DuckDB connection object, got {type(framework_connection_object)}")

        return framework_connection_object.from_arrow(data)
