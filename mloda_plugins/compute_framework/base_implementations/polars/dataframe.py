from typing import Any, Optional
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine, BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_mask_engine import PolarsMaskEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


class PolarsDataFrame(ComputeFramework):
    @staticmethod
    def is_available() -> bool:
        """Check if Polars is installed and available."""
        try:
            import polars  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def expected_data_framework(cls) -> Any:
        return cls.pl_dataframe()

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return PolarsMergeEngine

    def select_data_by_column_names(
        self, data: Any, selected_feature_names: set[FeatureName], column_ordering: Optional[str] = None
    ) -> Any:
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, column_names, ordering=column_ordering
        )
        return data.select(list(_selected_feature_names))

    def _extract_column_names(self, data: Any) -> set[str]:
        return set(data.columns)

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        if column_name in data.columns:
            return str(data[column_name].dtype)
        return None

    def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
        if column_name not in data.columns:
            return None
        return self._polars_type_to_data_type(data[column_name].dtype)

    @staticmethod
    def _polars_type_to_data_type(dtype: Any) -> Optional[DataType]:
        if pl is None:
            return None
        if dtype == pl.Int32:
            return DataType.INT32
        if dtype == pl.Int64:
            return DataType.INT64
        if dtype == pl.Float32:
            return DataType.FLOAT
        if dtype == pl.Float64:
            return DataType.DOUBLE
        if dtype == pl.Boolean:
            return DataType.BOOLEAN
        if dtype == pl.String:
            return DataType.STRING
        if dtype == pl.Binary:
            return DataType.BINARY
        if dtype == pl.Date:
            return DataType.DATE
        if isinstance(dtype, pl.Datetime):
            return DataType.TIMESTAMP_MILLIS if dtype.time_unit == "ms" else DataType.TIMESTAMP_MICROS
        if isinstance(dtype, pl.Decimal):
            return DataType.DECIMAL
        return None

    @classmethod
    def pl_dataframe(cls) -> Any:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.DataFrame

    @classmethod
    def pl_series(cls) -> Any:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.Series

    def _is_empty(self, data: Any) -> bool:
        return bool(data.height == 0)

    def transform(
        self,
        data: Any,
        feature_names: set[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to table"""
            return self.pl_dataframe()(data)

        if isinstance(data, self.pl_series()):
            """Added data: Add column to table"""
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                if feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the dataframe")

                # In Polars, we use with_columns to add new columns
                return self.data.with_columns(data.alias(feature_name))
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        return PolarsFilterEngine

    @classmethod
    def mask_engine(cls) -> type[BaseMaskEngine]:
        return PolarsMaskEngine
