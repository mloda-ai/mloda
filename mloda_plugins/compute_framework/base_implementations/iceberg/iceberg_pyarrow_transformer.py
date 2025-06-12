from typing import Any, Optional
from mloda_core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer

try:
    from pyiceberg.table import Table as IcebergTable
    import pyarrow as pa
except ImportError:
    IcebergTable = None  # type: ignore
    pa = None


class IcebergPyarrowTransformer(BaseTransformer):
    """
    Transformer for converting between Iceberg tables and PyArrow tables.

    This transformer enables data interchange between Iceberg and other mloda
    compute frameworks through PyArrow as the common format.

    Note: Currently only supports Iceberg -> PyArrow conversion.
    PyArrow -> Iceberg conversion requires catalog context and is not yet supported.
    """

    @classmethod
    def framework(cls) -> Any:
        """Return the Iceberg table type."""
        if IcebergTable is None:
            return NotImplementedError
        return IcebergTable

    @classmethod
    def other_framework(cls) -> Any:
        """Return the PyArrow table type."""
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        """Import PyIceberg."""
        import pyiceberg.table

    @classmethod
    def import_other_fw(cls) -> None:
        """Import PyArrow."""
        import pyarrow

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """
        Transform Iceberg table to PyArrow table.

        Args:
            data: Iceberg table

        Returns:
            PyArrow table
        """

        if isinstance(data, pa.Table):
            # If data is already a PyArrow table, return it directly
            return data

        if not isinstance(data, IcebergTable):
            raise ValueError(f"Expected Iceberg table, got {type(data)}")

        # Use Iceberg's scan to convert to PyArrow
        return data.scan().to_arrow()

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """
        Transform PyArrow table to Iceberg table.

        For testing and development purposes, we pass through PyArrow tables as-is
        since the Iceberg framework can work with PyArrow tables directly.

        In a production environment, this would create a proper Iceberg table
        using the catalog context and schema management.

        Args:
            data: PyArrow table
            framework_connection_object: Iceberg catalog (optional)

        Returns:
            PyArrow table (passed through for compatibility)
        """
        if not isinstance(data, pa.Table):
            raise ValueError(f"Expected PyArrow table, got {type(data)}")

        # For now, pass through PyArrow tables as the Iceberg framework
        # can handle them directly. In a real implementation, you would:
        # 1. Use the framework_connection_object (catalog) to create an Iceberg table
        # 2. Define the table schema and location
        # 3. Write the PyArrow data to the Iceberg table
        # 4. Return the Iceberg table reference

        return data
