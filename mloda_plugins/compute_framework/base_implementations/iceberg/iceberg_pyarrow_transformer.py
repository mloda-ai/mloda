from typing import Any
from mloda.provider import BaseTransformer

try:
    from pyiceberg.table import Table as IcebergTable
    import pyarrow as pa
except ImportError:
    IcebergTable = None  # type: ignore[assignment,misc]
    pa = None  # type: ignore[assignment, unused-ignore]


def normalize_arrow_types(table: Any) -> Any:
    """Recast large_* types (also nested) to plain Arrow types; Iceberg types have no width.

    Empty scans and files written from large Arrow types return large_*; a single chunk over 2 GiB fails the cast.
    """
    target = pa.schema([field.with_type(_plain_type(field.type)) for field in table.schema], table.schema.metadata)
    if target.equals(table.schema, check_metadata=True):
        return table
    return table.cast(target)


def _plain_type(arrow_type: Any) -> Any:
    if pa.types.is_large_string(arrow_type):
        return pa.string()
    if pa.types.is_large_binary(arrow_type):
        return pa.binary()
    if pa.types.is_large_list(arrow_type):
        return pa.list_(arrow_type.value_field.with_type(_plain_type(arrow_type.value_type)))
    if pa.types.is_fixed_size_list(arrow_type):
        return pa.list_(arrow_type.value_field.with_type(_plain_type(arrow_type.value_type)), arrow_type.list_size)
    if pa.types.is_list(arrow_type):
        return pa.list_(arrow_type.value_field.with_type(_plain_type(arrow_type.value_type)))
    if pa.types.is_struct(arrow_type):
        return pa.struct([f.with_type(_plain_type(f.type)) for f in arrow_type])
    if pa.types.is_map(arrow_type):
        return pa.map_(
            arrow_type.key_field.with_type(_plain_type(arrow_type.key_type)),
            arrow_type.item_field.with_type(_plain_type(arrow_type.item_type)),
            keys_sorted=arrow_type.keys_sorted,
        )
    return arrow_type


class IcebergPyArrowTransformer(BaseTransformer):
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
        import pyiceberg.table  # noqa: F401

    @classmethod
    def import_other_fw(cls) -> None:
        """Import PyArrow."""
        import pyarrow  # noqa: F401

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
        return normalize_arrow_types(data.scan().to_arrow())

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
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
