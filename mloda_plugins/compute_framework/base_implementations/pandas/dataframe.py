from collections.abc import Sequence
from typing import Any

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseFilterEngine, BaseMaskEngine, BaseMergeEngine, ComputeFramework, OutputSchema
from mloda.user import FeatureName
from mloda_plugins.compute_framework.base_implementations.attribute_column_read import is_attribute_column_read
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import PandasMaskEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine

try:
    import pandas as pd
except ImportError:
    pd = None

_PANDAS_LOCAL_VARIABLE_PREFIX = "local variable"
# datetime64 s and ns count as exact TIMESTAMP_MICROS because pandas coerces them.
_PANDAS_EXACT: dict[tuple[str, int | None], DataType] = {
    ("i", 1): DataType.INT8,
    ("i", 4): DataType.INT32,
    ("i", 8): DataType.INT64,
    ("u", 1): DataType.UINT8,
    ("u", 8): DataType.UINT64,
    ("f", 4): DataType.FLOAT,
    ("f", 8): DataType.DOUBLE,
    ("b", 1): DataType.BOOLEAN,
}
_PANDAS_FOLDS: dict[tuple[str, int | None], DataType] = {
    ("i", 2): DataType.INT32,
    ("u", 2): DataType.INT32,
    ("u", 4): DataType.INT64,
    ("f", 2): DataType.FLOAT,
}


def _kind_and_size(dtype: Any) -> tuple[str, int | None]:
    numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
    return getattr(numpy_dtype, "kind", ""), getattr(numpy_dtype, "itemsize", None)


class PandasDataFrame(ComputeFramework):
    INSTALL_HINT = 'pip install "mloda[pandas]"'

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
    def is_missing_column_error(cls, error: Exception) -> bool:
        if pd is None:
            return False
        if isinstance(error, pd.errors.UndefinedVariableError):
            return not str(error).startswith(_PANDAS_LOCAL_VARIABLE_PREFIX)
        return is_attribute_column_read(error, (pd.DataFrame,))

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

    @classmethod
    def extract_column_names(cls, data: Any) -> set[str]:
        return set(data.columns)

    def _first_column_dtype(self, data: Any, column_name: str) -> Any | None:
        """Return column_name's dtype from its first occurrence; indexing by a duplicated label
        returns a DataFrame, which has no .dtype.
        """
        for name, dtype in zip(data.columns, data.dtypes):
            if name == column_name:
                return dtype
        return None

    def _append_columns(self, data: Any, columns: dict[str, Any]) -> Any | None:
        if not isinstance(data, pd.DataFrame):
            return None
        if any(len(values) != len(data) for values in columns.values()):
            return None
        return data.assign(**{name: list(values) for name, values in columns.items()})

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        dtype = self._first_column_dtype(data, column_name)
        if dtype is None:
            return None
        return str(dtype)

    def _extract_column_data_type(self, data: Any, column_name: str) -> DataType | None:
        dtype = self._first_column_dtype(data, column_name)
        if dtype is None:
            return None
        if isinstance(dtype, pd.ArrowDtype):
            return DataType.from_arrow_type_safe(dtype.pyarrow_dtype)
        fold = _PANDAS_FOLDS.get(_kind_and_size(dtype))
        if fold is not None:
            return fold
        exact = self._pandas_dtype_to_exact_data_type(dtype)
        if exact is not None:
            return exact
        if pd.api.types.is_bool_dtype(dtype):
            return DataType.BOOLEAN
        if pd.api.types.is_float_dtype(dtype) and dtype.itemsize > 8:
            return DataType.DOUBLE
        return None

    def _exact_column_data_type(self, data: Any, column_name: str) -> DataType | None:
        dtype = self._first_column_dtype(data, column_name)
        if dtype is None:
            return None
        return self._pandas_dtype_to_exact_data_type(dtype)

    @staticmethod
    def _pandas_dtype_to_exact_data_type(dtype: Any) -> DataType | None:
        if isinstance(dtype, pd.ArrowDtype):
            return DataType._from_arrow_type_exact(dtype.pyarrow_dtype)
        if isinstance(dtype, pd.StringDtype):
            return DataType.STRING
        if isinstance(dtype, pd.DatetimeTZDtype):
            return DataType.TIMESTAMP_MICROS_TZ
        kind, size = _kind_and_size(dtype)
        if kind == "M":
            return DataType.TIMESTAMP_MILLIS if str(dtype) == "datetime64[ms]" else DataType.TIMESTAMP_MICROS
        return _PANDAS_EXACT.get((kind, size))

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
