from collections.abc import Mapping, Sequence
from datetime import UTC
from typing import Any
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.hook_context import OutputSchema
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine, BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.attribute_column_read import is_attribute_column_read
from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_mask_engine import PolarsMaskEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.arrow_build import (
    aware_zone,
    has_gap,
    is_gap,
    nested_leaves,
    null_gaps,
)
from mloda_plugins.compute_framework.base_implementations.polars.polars_pyarrow_transformer import (
    reject_sub_hour_zone,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


def _polars_zone(zone: str) -> str:
    """Map a whole-hour "+HH:MM" offset to its Etc/GMT zone (inverted sign); other names pass through."""
    if zone[:1] in "+-" and zone[3:] == ":00":
        hours = int(zone[1:3])
        return "UTC" if hours == 0 else f"Etc/GMT{'-' if zone[0] == '+' else '+'}{hours}"
    return zone


_POLARS_EXACT: dict[Any, DataType] = (
    {
        pl.Int8: DataType.INT8,
        pl.Int32: DataType.INT32,
        pl.Int64: DataType.INT64,
        pl.UInt8: DataType.UINT8,
        pl.UInt64: DataType.UINT64,
        pl.Float32: DataType.FLOAT,
        pl.Float64: DataType.DOUBLE,
        pl.Boolean: DataType.BOOLEAN,
        pl.String: DataType.STRING,
        pl.Binary: DataType.BINARY,
        pl.Date: DataType.DATE,
    }
    if pl is not None
    else {}
)
_POLARS_DATETIME_EXACT = {"ms": DataType.TIMESTAMP_MILLIS, "us": DataType.TIMESTAMP_MICROS}


class PolarsDataFrame(ComputeFramework):
    INSTALL_HINT = 'pip install "mloda[polars]"'

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
    def is_missing_column_error(cls, error: Exception) -> bool:
        if pl is None:
            return False
        if isinstance(error, pl.exceptions.ColumnNotFoundError):
            return True
        if not is_attribute_column_read(error, (pl.DataFrame, pl.LazyFrame)):
            return False
        obj = getattr(error, "obj", None)
        return not (isinstance(obj, pl.DataFrame) and getattr(error, "name", None) in obj.columns)

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return PolarsMergeEngine

    def select_data_by_column_names(
        self,
        data: Any,
        selected_feature_names: Sequence[FeatureName],
        column_ordering: str | None = None,
        request_feature_order: list[str] | None = None,
    ) -> Any:
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, column_names, ordering=column_ordering, request_feature_order=request_feature_order
        )
        return data.select(list(_selected_feature_names))

    @classmethod
    def extract_column_names(cls, data: Any) -> set[str]:
        return set(data.columns)

    def _append_columns(self, data: Any, columns: dict[str, Any]) -> Any | None:
        if not isinstance(data, pl.DataFrame):
            return None
        if any(len(values) != data.height for values in columns.values()):
            return None
        aware = self._aware_series(columns)
        return data.with_columns(
            [aware[name] if name in aware else pl.Series(name, values) for name, values in columns.items()]
        )

    @staticmethod
    def _aware_series(columns: Mapping[str, Any]) -> dict[str, Any]:
        """Series for aware and gapped list columns; raise on mixed, unnameable or sub-hour zones."""
        series = {}
        for name, values in columns.items():
            if isinstance(values, (list, tuple)):
                leaves = nested_leaves(values)
                if leaves is not None:
                    nested_zone = aware_zone(leaves, name)
                    if nested_zone is not None:
                        reject_sub_hour_zone(name, nested_zone)
                zone = aware_zone(values, name)
                if zone is not None:
                    reject_sub_hour_zone(name, zone)
                    utc = [None if is_gap(v) else v.astimezone(UTC) for v in values]
                    series[name] = pl.Series(name, utc, dtype=pl.Datetime("us", _polars_zone(zone)))
                elif has_gap(values):
                    series[name] = pl.Series(name, null_gaps(values))
        return series

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        if column_name in data.columns:
            return str(data[column_name].dtype)
        return None

    def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
        if column_name not in data.columns:
            return None
        return self._polars_type_to_data_type(data[column_name].dtype)

    def _exact_column_data_type(self, data: Any, column_name: str) -> DataType | None:
        if column_name not in data.columns:
            return None
        return self._polars_exact_data_type(data[column_name].dtype)

    @staticmethod
    def _polars_type_to_data_type(dtype: Any) -> DataType | None:
        if pl is None:
            return None
        if dtype in (pl.Int16, pl.UInt16):
            return DataType.INT32
        if dtype == pl.UInt32:
            return DataType.INT64
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None and dtype.time_unit == "ns":
            return DataType.TIMESTAMP_MICROS
        if isinstance(dtype, pl.Decimal):
            return DataType.DECIMAL
        return PolarsDataFrame._polars_exact_data_type(dtype)

    @staticmethod
    def _polars_exact_data_type(dtype: Any) -> DataType | None:
        if pl is None:
            return None
        if isinstance(dtype, (pl.List, pl.Array)):
            return DataType.LIST
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone is not None:
                return DataType.TIMESTAMP_MICROS_TZ
            return _POLARS_DATETIME_EXACT.get(dtype.time_unit)
        if isinstance(dtype, pl.Decimal):
            return DataType.DECIMAL if (dtype.precision, dtype.scale) == (38, 18) else None
        return _POLARS_EXACT.get(dtype)

    def _output_schema(self, data: Any) -> OutputSchema | None:
        """Read `.schema` once and build the sorted (name, dtype) pairs from it directly. The dict
        interchange shape must be handled first, since it reaches this method before transform()
        normalizes it.
        """
        if isinstance(data, dict):
            return super()._output_schema(data)
        schema = data.schema
        if not schema:
            return None
        return tuple((name, str(dtype)) for name, dtype in sorted(schema.items(), key=lambda pair: pair[0]))

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

    def transform(
        self,
        data: Any,
        feature_names: Sequence[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to table"""
            return self.pl_dataframe()({**data, **self._aware_series(data)})

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
