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

    Uses List[Dict[str, Any]] as the data structure for tabular data.
    This framework provides a simple, dependency-free implementation using
    native Python data structures.

    Data Structure:
        List[Dict[str, Any]] - Each dict represents a row, keys are column names

    Example:
        [
            {"col1": 1, "col2": "a"},
            {"col1": 2, "col2": "b"}
        ]
    """

    @classmethod
    def expected_data_framework(cls) -> Any:
        return list

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return PythonDictMergeEngine

    def select_data_by_column_names(
        self,
        data: list[dict[str, Any]],
        selected_feature_names: set[FeatureName],
        column_ordering: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not data:
            return []

        # Get all unique column names from all rows
        column_names: set[str] = set()
        for row in data:
            column_names.update(row.keys())

        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, column_names, ordering=column_ordering
        )

        return [{k: record.get(k) for k in _selected_feature_names if k in record} for record in data]

    def _extract_column_names(self, data: Any) -> set[str]:
        all_columns: set[str] = set()
        for row in data:
            if isinstance(row, dict):
                all_columns.update(row.keys())
        return all_columns

    def _is_schemaless_empty(self, data: Any) -> bool:
        """``[]`` and ``{}`` are PythonDict's representational empties; ``transform``
        collapses both to ``[]``. None is excluded: filter validation never sees a
        None result path.
        """
        return isinstance(data, (list, dict)) and not data

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        for row in data:
            if isinstance(row, dict) and column_name in row and row[column_name] is not None:
                return type(row[column_name]).__name__
        return None

    def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
        for row in data:
            if not isinstance(row, dict) or column_name not in row or row[column_name] is None:
                continue
            val = row[column_name]
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

    def transform(self, data: Any, feature_names: set[str]) -> list[dict[str, Any]]:
        """
        Transforms data to the PythonDict framework format.

        Args:
            data: Input data to transform
            feature_names: Set of feature names being processed

        Returns:
            List[Dict]: Data in PythonDict format

        Raises:
            ValueError: If data type is not supported
        """

        if data is None or (isinstance(data, (list, dict)) and not data):
            return []

        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data  # type: ignore[no-any-return]

        if isinstance(data, dict):
            """Initial data: Transform columnar dict to row-based list of dicts"""
            if all(isinstance(v, list) for v in data.values()):
                # Columnar format: {"col1": [1,2], "col2": [3,4]}
                # Convert to: [{"col1":1,"col2":3}, {"col1":2,"col2":4}]

                # Get the length from the first column
                first_key = next(iter(data.keys()))
                length = len(data[first_key])

                # Verify all columns have the same length
                for key, values in data.items():
                    if len(values) != length:
                        raise ValueError(
                            f"All columns must have the same length. Column '{key}' has length {len(values)}, expected {length}"
                        )

                return [{key: data[key][i] for key in data.keys()} for i in range(length)]
            else:
                # Single row dict: {"col1": 1, "col2": 2} -> [{"col1": 1, "col2": 2}]
                return [data]

        if isinstance(data, list):
            """Data is already in list format"""

            # Verify it's a list of dicts
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise ValueError(f"Expected list of dictionaries, but item at index {i} is {type(item)}")

            return data

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
