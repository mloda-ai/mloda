from collections.abc import Sequence
from typing import Any, Optional
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseMergeEngine
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_filter_engine import IcebergFilterEngine

try:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.types import (
        BinaryType,
        BooleanType,
        DateType,
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        TimestampType,
        TimestamptzType,
    )
    import pyarrow as pa
except ImportError:
    Catalog = None  # type: ignore[assignment,misc]
    IcebergTable = None  # type: ignore[assignment,misc]
    BinaryType = None  # type: ignore[assignment,misc]
    BooleanType = None  # type: ignore[assignment,misc]
    DateType = None  # type: ignore[assignment,misc]
    DecimalType = None  # type: ignore[assignment,misc]
    DoubleType = None  # type: ignore[assignment,misc]
    FloatType = None  # type: ignore[assignment,misc]
    IntegerType = None  # type: ignore[assignment,misc]
    LongType = None  # type: ignore[assignment,misc]
    StringType = None  # type: ignore[assignment,misc]
    TimestampType = None  # type: ignore[assignment,misc]
    TimestamptzType = None  # type: ignore[assignment,misc]
    pa = None  # type: ignore[assignment]


class IcebergFramework(ComputeFramework):
    """
    Iceberg compute framework implementation.

    This framework provides integration with Apache Iceberg tables, supporting
    schema evolution, time travel, and efficient data management. It uses PyArrow
    as the interchange format for compatibility with other mloda frameworks.

    Note: This implementation focuses on read operations. The catalog must be
    provided via set_framework_connection_object() before use.
    """

    def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
        """
        Set the Iceberg catalog for table operations.

        Args:
            framework_connection_object: Iceberg catalog instance
        """
        if Catalog is None:
            raise ImportError("PyIceberg is not installed. To use this framework, please install pyiceberg.")

        if self.framework_connection_object is None:
            if framework_connection_object is not None:
                # Accept either a catalog instance or a table instance
                if hasattr(framework_connection_object, "load_table"):
                    # It's a catalog
                    self.framework_connection_object = framework_connection_object
                elif isinstance(framework_connection_object, IcebergTable):
                    # It's already a table - store it directly
                    self.framework_connection_object = framework_connection_object
                else:
                    raise ValueError(f"Expected an Iceberg catalog or table, got {type(framework_connection_object)}")

    @classmethod
    def _connection_matches(cls, conn: Any) -> bool:
        if Catalog is None:
            return False
        return hasattr(conn, "load_table") or (IcebergTable is not None and isinstance(conn, IcebergTable))

    @staticmethod
    def is_available() -> bool:
        """Check if PyIceberg is installed and available."""
        try:
            import pyiceberg  # noqa: F401
            import pyarrow  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def expected_data_framework(cls) -> Any:
        """Return the expected Iceberg table type."""
        if IcebergTable is None:
            raise ImportError("PyIceberg is not installed. To use this framework, please install pyiceberg.")
        return IcebergTable

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        """Iceberg tables don't support direct merging in this framework context."""
        raise NotImplementedError(
            f"Merge functionality is not implemented for {cls.__name__}. "
            "Iceberg tables are typically used for data lake scenarios where merging "
            "is handled at the catalog/table/engine level, not at the compute framework level."
        )

    def select_data_by_column_names(
        self,
        data: Any,
        selected_feature_names: Sequence[FeatureName],
        column_ordering: Optional[str] = None,
        request_feature_order: Optional[list[str]] = None,
    ) -> Any:
        """
        Select specific columns from Iceberg table.

        Args:
            data: Iceberg table
            selected_feature_names: Sequence of feature names to select
            column_ordering: Optional column ordering strategy

        Returns:
            Iceberg table scan with selected columns
        """
        if not isinstance(data, IcebergTable):
            return data

        column_names = set(data.schema().column_names)
        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, column_names, ordering=column_ordering, request_feature_order=request_feature_order
        )

        # Use Iceberg's scan with column selection
        return data.scan(selected_fields=tuple(_selected_feature_names))

    def _extract_column_names(self, data: Any) -> set[str]:
        if IcebergTable is not None and isinstance(data, IcebergTable):
            return set(data.schema().column_names)
        # After transform, data may be a PyArrow table
        return set(data.schema.names)

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        if IcebergTable is None or not isinstance(data, IcebergTable):
            return None
        schema = data.schema()
        if column_name not in set(schema.column_names):
            return None
        field = schema.find_field(column_name)
        if field is None:
            return None
        return str(field.field_type)

    def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
        if IcebergTable is None or not isinstance(data, IcebergTable):
            return None
        schema = data.schema()
        if column_name not in set(schema.column_names):
            return None
        field = schema.find_field(column_name)
        if field is None:
            return None
        field_type = field.field_type
        if isinstance(field_type, IntegerType):
            return DataType.INT32
        if isinstance(field_type, LongType):
            return DataType.INT64
        if isinstance(field_type, FloatType):
            return DataType.FLOAT
        if isinstance(field_type, DoubleType):
            return DataType.DOUBLE
        if isinstance(field_type, BooleanType):
            return DataType.BOOLEAN
        if isinstance(field_type, StringType):
            return DataType.STRING
        if isinstance(field_type, BinaryType):
            return DataType.BINARY
        if isinstance(field_type, DateType):
            return DataType.DATE
        if isinstance(field_type, (TimestampType, TimestamptzType)):
            return DataType.TIMESTAMP_MICROS
        if isinstance(field_type, DecimalType):
            return DataType.DECIMAL
        return None

    def transform(self, data: Any, feature_names: Sequence[str]) -> Any:
        """
        Transform data to Iceberg table format.

        Args:
            data: Input data (dict, PyArrow table, etc.)
            feature_names: Sequence of feature names

        Returns:
            Transformed data in Iceberg table format
        """
        # First try the standard transformer approach
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to PyArrow table (Iceberg table creation requires catalog context)"""
            # Convert dict to PyArrow table first
            # The transformer will handle conversion to Iceberg table when needed
            if pa is None:
                raise ImportError("PyArrow is not installed. To use this framework, please install pyarrow.")
            return pa.Table.from_pydict(data)

        if isinstance(data, IcebergTable):
            """Data is already an Iceberg table"""
            return data

        if pa is not None and isinstance(data, pa.Table):
            """PyArrow table: Pass through as-is since Iceberg can work with PyArrow"""
            # For now, we'll pass PyArrow tables through as-is
            # In a real implementation, you might want to convert to Iceberg table
            # but that requires catalog context and table naming
            return data

        raise ValueError(f"Data type {type(data)} is not supported by {self.__class__.__name__}")

    def validate_expected_framework(self, location: Optional[str] = None) -> None:
        """
        Override to accept both Iceberg tables and PyArrow tables.

        Since Iceberg framework can work with PyArrow tables as an interchange format,
        we accept both types.
        """
        if self.expected_data_framework() is None:
            return

        if self.data is None:
            return

        # If location is a string, it means it is a uuid of the object in arrow flight.
        if isinstance(location, str) and self.data is not None:
            return

        # Accept both Iceberg tables and PyArrow tables
        if isinstance(self.data, self.expected_data_framework()):
            return

        if pa is not None and isinstance(self.data, pa.Table):
            return

        raise ValueError(f"Data type {type(self.data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        """Return the Iceberg filter engine."""
        return IcebergFilterEngine
