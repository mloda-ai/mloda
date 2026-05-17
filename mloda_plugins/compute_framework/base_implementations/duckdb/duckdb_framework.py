import logging
from typing import Any, Optional
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda.user import FeatureName, ParallelizationMode
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine, BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_mask_engine import DuckDBMaskEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


_DUCKDB_TYPE_MAP: dict[str, DataType] = {
    "INTEGER": DataType.INT32,
    "BIGINT": DataType.INT64,
    "SMALLINT": DataType.INT32,
    "TINYINT": DataType.INT32,
    "HUGEINT": DataType.INT64,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.DOUBLE,
    "REAL": DataType.FLOAT,
    "BOOLEAN": DataType.BOOLEAN,
    "VARCHAR": DataType.STRING,
    "BLOB": DataType.BINARY,
    "DATE": DataType.DATE,
    "TIMESTAMP": DataType.TIMESTAMP_MICROS,
    "TIMESTAMP_MS": DataType.TIMESTAMP_MILLIS,
    "TIMESTAMP_S": DataType.TIMESTAMP_MICROS,
    "TIMESTAMP_NS": DataType.TIMESTAMP_MICROS,
    "TIMESTAMP WITH TIME ZONE": DataType.TIMESTAMP_MICROS,
    "TIMESTAMPTZ": DataType.TIMESTAMP_MICROS,
}


class DuckDBFramework(ComputeFramework):
    """DuckDB framework implementation for ComputeFramework.

    This framework does not support multiprocessing, so it should not be used with multiprocessing.
    """

    def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
        """Use given DuckDB connection."""
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")
        if framework_connection_object is None:
            raise ValueError("A DuckDB connection object is required.")
        if not isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            raise ValueError(f"Expected a DuckDB connection object, got {type(framework_connection_object)}")
        if self.framework_connection_object is not None:
            if self.framework_connection_object is not framework_connection_object:
                raise ValueError("A different connection is already set. Cannot replace an existing connection.")
            return
        self.framework_connection_object = framework_connection_object

    @staticmethod
    def is_available() -> bool:
        """Check if DuckDB is installed and available."""

        try:
            import duckdb  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def expected_data_framework(cls) -> Any:
        return cls.duckdb_relation()

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return DuckDBMergeEngine

    def select_data_by_column_names(
        self, data: Any, selected_feature_names: set[FeatureName], column_ordering: Optional[str] = None
    ) -> Any:
        """Materialize the final result as a PyArrow Table.

        Override this method in a subclass to stay lazy (return a DuckdbRelation)
        or to use a different output format.
        """
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, column_names, ordering=column_ordering
        )

        selected_columns = list(_selected_feature_names)
        return data.select(*selected_columns).to_arrow_table()

    def _extract_column_names(self, data: Any) -> set[str]:
        return set(data.columns)

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        if column_name in data.columns:
            dtypes = data._relation.dtypes
            idx = data.columns.index(column_name)
            return str(dtypes[idx])
        return None

    def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
        if column_name not in data.columns:
            return None
        idx = data.columns.index(column_name)
        duckdb_type = data._relation.dtypes[idx]
        type_str = str(duckdb_type).upper()
        if type_str in _DUCKDB_TYPE_MAP:
            return _DUCKDB_TYPE_MAP[type_str]
        if type_str.startswith("DECIMAL"):
            return DataType.DECIMAL
        return None

    @classmethod
    def duckdb_relation(cls) -> Any:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")
        return DuckdbRelation

    def transform(
        self,
        data: Any,
        feature_names: set[str],
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
            return DuckdbRelation.from_arrow(self.framework_connection_object, arrow_table)

        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))
                if hasattr(self.data, "columns") and feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the relation")
                return self.data.append_column(feature_name, list(data))
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC}

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        return DuckDBFilterEngine

    @classmethod
    def mask_engine(cls) -> type[BaseMaskEngine]:
        return DuckDBMaskEngine
