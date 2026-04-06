from typing import Any, Optional
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine, BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_mask_engine import (
    PythonDictFilterMaskEngine,
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
            raise ValueError(f"Data cannot be empty: {selected_feature_names}")

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

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        for row in data:
            if isinstance(row, dict) and column_name in row and row[column_name] is not None:
                return type(row[column_name]).__name__
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

        if not data:
            raise ValueError("Data cannot be empty")

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
    def filter_mask_engine(cls) -> type[BaseFilterMaskEngine]:
        return PythonDictFilterMaskEngine
