import logging
from typing import Any, Set, Type, Optional
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

logger = logging.getLogger(__name__)


class DuckDBFramework(ComputeFrameWork):
    """DuckDB framework implementation for ComputeFrameWork.

    This framework does not support multiprocessing, so it should not be used with multiprocessing.
    """

    def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
        """Use given DuckDB connection."""
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")

        if self.framework_connection_object is None:
            if framework_connection_object is not None:
                if not isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
                    raise ValueError(f"Expected a DuckDB connection object, got {type(framework_connection_object)}")
                self.framework_connection_object = framework_connection_object

    @staticmethod
    def is_available() -> bool:
        """Check if DuckDB is installed and available."""

        try:
            import duckdb

            return True
        except ImportError:
            return False

    @staticmethod
    def expected_data_framework() -> Any:
        return DuckDBFramework.duckdb_relation()

    def merge_engine(self) -> Type[BaseMergeEngine]:
        return DuckDBMergeEngine

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)

        # Use DuckDB relation's select method directly
        selected_columns = list(_selected_feature_names)
        return data.select(*selected_columns).df()

    def set_column_names(self) -> None:
        self.column_names = set(self.data.columns)

    @staticmethod
    def duckdb_relation() -> Any:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")
        return duckdb.DuckDBPyRelation

    def transform(
        self,
        data: Any,
        feature_names: Set[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to DuckDB relation"""
            # Convert dict to PyArrow first, then to DuckDB relation
            import pyarrow as pa

            arrow_table = pa.Table.from_pydict(data)

            if self.framework_connection_object is None:
                raise ValueError(
                    "Framework connection object is not set. Please call set_framework_connection_object() first."
                )
            return self.framework_connection_object.from_arrow(arrow_table)

        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
            """Added data: Add column to relation"""
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                if hasattr(self.data, "columns") and feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the relation")

                # Convert the new data to a temporary relation and join it
                # This is a simplified approach - in practice, we'd need more sophisticated column addition
                temp_data = {feature_name: list(data) if hasattr(data, "__iter__") else [data]}
                import pyarrow as pa

                temp_arrow = pa.Table.from_pydict(temp_data)

                if self.framework_connection_object is None:
                    raise ValueError(
                        "Framework connection object is not set. Please call set_framework_connection_object() first."
                    )
                temp_relation = self.framework_connection_object.from_arrow(temp_arrow)

                # For now, we'll assume the data has the same number of rows and can be combined
                # In a real implementation, we'd need proper indexing logic
                return temp_relation
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    def filter_engine(self) -> Type[BaseFilterEngine]:
        return DuckDBFilterEngine
