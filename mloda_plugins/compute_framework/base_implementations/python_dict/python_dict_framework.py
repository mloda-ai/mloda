import datetime
import decimal
from typing import Any, Optional
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine, BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)


class PythonDictFramework(ComputeFramework):
    """
    PythonDict Compute Framework

    Uses a COLUMNAR ``dict[str, list[Any]]`` as the data structure for tabular data.
    This framework provides a simple, dependency-free implementation using
    native Python data structures.

    Data Structure:
        dict[str, list[Any]] - Each key is a column name, each value is that column's
        list of cell values. All value-lists share one length (= the row count). The
        schema is the set of keys, present even at zero rows.

    Example:
        {
            "col1": [1, 2],
            "col2": ["a", "b"],
        }

    ``{"a": []}`` is a valid schema-bearing zero-row frame (one known column). ``{}`` is
    the only schema-less value (zero columns).
    """

    @classmethod
    def expected_data_framework(cls) -> Any:
        return dict

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return PythonDictMergeEngine

    def select_data_by_column_names(
        self,
        data: dict[str, list[Any]],
        selected_feature_names: set[FeatureName],
        column_ordering: Optional[str] = None,
    ) -> dict[str, list[Any]]:
        if not data:
            return {}

        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, set(data.keys()), ordering=column_ordering
        )

        return {k: data[k] for k in _selected_feature_names if k in data}

    def _extract_column_names(self, data: Any) -> set[str]:
        if isinstance(data, dict):
            return set(data.keys())
        return set()

    def _is_schemaless_empty(self, data: Any) -> bool:
        """Only the empty dict ``{}`` (zero columns) is schema-less. A zero-row but
        column-bearing frame such as ``{"a": []}`` carries a schema and is NOT schema-less.
        """
        return isinstance(data, dict) and not data

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        for value in data.get(column_name, []):
            if value is not None:
                return type(value).__name__
        return None

    def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
        for val in data.get(column_name, []):
            if val is None:
                continue
            if isinstance(val, bool):
                return DataType.BOOLEAN
            if isinstance(val, int):
                return DataType.INT64
            if isinstance(val, float):
                return DataType.DOUBLE
            if isinstance(val, str):
                return DataType.STRING
            if isinstance(val, bytes):
                return DataType.BINARY
            if isinstance(val, datetime.datetime):
                return DataType.TIMESTAMP_MICROS
            if isinstance(val, datetime.date):
                return DataType.DATE
            if isinstance(val, decimal.Decimal):
                return DataType.DECIMAL
            return None
        return None

    def transform(self, data: Any, feature_names: set[str]) -> dict[str, list[Any]]:
        """
        Transforms data to the COLUMNAR PythonDict framework format.

        Args:
            data: Input data to transform
            feature_names: Set of feature names being processed

        Returns:
            dict[str, list[Any]]: Data in columnar PythonDict format

        Raises:
            ValueError: If data type is not supported
        """

        # The representational empties [] and {} normalize to the schema-less {}. None is a
        # missing return value (state A), not an empty result, and falls through to the
        # unsupported-type rejection below.
        if isinstance(data, list) and not data:
            return {}
        if isinstance(data, dict) and not data:
            return {}

        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data  # type: ignore[no-any-return]

        if isinstance(data, dict):
            # Columnar format: every value must be a list of equal length.
            if not all(isinstance(v, list) for v in data.values()):
                raise ValueError(
                    f"Columnar dict values must all be lists (rows must arrive as a list of dicts). Got: {data}"
                )

            lengths = {len(v) for v in data.values()}
            if len(lengths) > 1:
                raise ValueError(f"All columns must have the same length. Got column lengths {lengths}.")

            return data

        if isinstance(data, list):
            # List of row dicts -> pivot to columnar. Keys must be homogeneous across rows.
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Expected list of dictionaries, but item at index {i} is {type(item)}")

            first_keys = list(data[0].keys())
            first_keys_set = set(first_keys)
            for i, item in enumerate(data):
                if set(item.keys()) != first_keys_set:
                    raise ValueError(
                        f"Inconsistent row keys at index {i}: expected {first_keys_set}, got {set(item.keys())}."
                    )

            return {key: [row[key] for row in data] for key in first_keys}

        raise ValueError(f"Data type {type(data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        """
        Returns the filter engine for PythonDict framework.

        Returns:
            Type[BaseFilterEngine]: PythonDictFilterEngine class
        """
        return PythonDictFilterEngine

    @classmethod
    def mask_engine(cls) -> type[BaseMaskEngine]:
        return PythonDictMaskEngine
