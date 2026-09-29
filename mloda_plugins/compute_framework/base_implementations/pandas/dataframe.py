from collections.abc import Sequence
from typing import Any

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseFilterEngine, BaseMaskEngine, BaseMergeEngine, ComputeFramework, OutputSchema
from mloda.user import FeatureName
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import PandasMaskEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine

try:
    import pandas as pd
except ImportError:
    pd = None


class PandasDataFrame(ComputeFramework):
    @staticmethod
    def is_available() -> bool:
        """Check if Pandas is installed and available."""
        try:
            import pandas  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def expected_data_framework(cls) -> Any:
        return cls.pd_dataframe()

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return PandasMergeEngine

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
        return data[[f for f in _selected_feature_names]]

    def _extract_column_names(self, data: Any) -> set[str]:
        return set(data.columns)

    def _first_column_dtype(self, data: Any, column_name: str) -> Any | None:
        """Return column_name's dtype from its first occurrence; indexing by a duplicated label
        returns a DataFrame, which has no .dtype.
        """
        for name, dtype in zip(data.columns, data.dtypes):
            if name == column_name:
                return dtype
        return None

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        dtype = self._first_column_dtype(data, column_name)
        if dtype is None:
            return None
        return str(dtype)

    def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
        dtype = self._first_column_dtype(data, column_name)
        if dtype is None:
            return None

        if isinstance(dtype, pd.ArrowDtype) and DataType.from_arrow_type_safe(dtype.pyarrow_dtype) == DataType.DECIMAL:
            return DataType.DECIMAL
        if isinstance(dtype, pd.StringDtype):
            return DataType.STRING
        if isinstance(dtype, pd.BooleanDtype):
            return DataType.BOOLEAN
        if isinstance(dtype, pd.api.types.DatetimeTZDtype):
            return DataType.TIMESTAMP_MICROS
        if pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOLEAN
        if pd.api.types.is_integer_dtype(dtype):
            return DataType.INT32 if dtype.itemsize <= 4 else DataType.INT64
        if pd.api.types.is_float_dtype(dtype):
            return DataType.FLOAT if dtype.itemsize <= 4 else DataType.DOUBLE
        if pd.api.types.is_datetime64_dtype(dtype):
            dtype_str = str(dtype)
            unit = dtype_str[len("datetime64[") : -1] if "[" in dtype_str else "ns"
            return DataType.TIMESTAMP_MILLIS if unit == "ms" else DataType.TIMESTAMP_MICROS

        return None

    def _output_schema(self, data: Any) -> OutputSchema | None:
        """Zip columns/dtypes positionally (indexing by name breaks on duplicates); first
        occurrence's dtype wins for duplicate names.
        """
        if isinstance(data, dict):
            return super()._output_schema(data)
        columns = data.columns
        if len(columns) == 0:
            return None
        seen: dict[str, str] = {}
        for name, dtype in zip(columns, data.dtypes):
            seen.setdefault(str(name), str(dtype))
        return tuple((name, seen[name]) for name in sorted(seen, key=str))

    @classmethod
    def pd_dataframe(cls) -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.DataFrame

    @classmethod
    def pd_series(cls) -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.Series

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
            return self.pd_dataframe().from_dict(data)

        if isinstance(data, self.pd_series()):
            """Added data: Add column to table"""
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                if feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the dataframe")

                self.data[feature_name] = data
                return self.data
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return self.pd_dataframe()(data)

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        return PandasFilterEngine

    @classmethod
    def mask_engine(cls) -> type[BaseMaskEngine]:
        return PandasMaskEngine
