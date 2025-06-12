from typing import Any, Set, Type, Optional
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_filter_engine import IcebergFilterEngine

try:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table as IcebergTable
    import pyarrow as pa
except ImportError:
    Catalog = None  # type: ignore
    IcebergTable = None  # type: ignore
    pa = None


class IcebergFramework(ComputeFrameWork):
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

    @staticmethod
    def is_available() -> bool:
        """Check if PyIceberg is installed and available."""
        try:
            import pyiceberg
            import pyarrow

            return True
        except ImportError:
            return False

    @staticmethod
    def expected_data_framework() -> Any:
        """Return the expected Iceberg table type."""
        if IcebergTable is None:
            raise ImportError("PyIceberg is not installed. To use this framework, please install pyiceberg.")
        return IcebergTable

    def merge_engine(self) -> Type[BaseMergeEngine]:
        """Iceberg tables don't support direct merging in this framework context."""
        raise NotImplementedError(
            f"Merge functionality is not implemented for {self.__class__.__name__}. "
            "Iceberg tables are typically used for data lake scenarios where merging "
            "is handled at the catalog/table/engine level, not at the compute framework level."
        )

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        """
        Select specific columns from Iceberg table.

        Args:
            data: Iceberg table
            selected_feature_names: Set of feature names to select

        Returns:
            Iceberg table scan with selected columns
        """
        if not isinstance(data, IcebergTable):
            return data

        column_names = set(data.schema().column_names)
        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)

        # Use Iceberg's scan with column selection
        return data.scan(selected_fields=tuple(_selected_feature_names))

    def set_column_names(self) -> None:
        """Set column names from the current data."""
        if self.data is not None and isinstance(self.data, IcebergTable):
            self.column_names = set(self.data.schema().column_names)

    def transform(self, data: Any, feature_names: Set[str]) -> Any:
        """
        Transform data to Iceberg table format.

        Args:
            data: Input data (dict, PyArrow table, etc.)
            feature_names: Set of feature names

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

    def filter_engine(self) -> Type[BaseFilterEngine]:
        """Return the Iceberg filter engine."""
        return IcebergFilterEngine
