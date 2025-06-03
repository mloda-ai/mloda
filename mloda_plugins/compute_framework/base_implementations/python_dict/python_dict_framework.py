from typing import Any, Set, Type, List, Dict
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)


class PythonDictFramework(ComputeFrameWork):
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

    @staticmethod
    def expected_data_framework() -> Any:
        return list

    def merge_engine(self) -> Type[BaseMergeEngine]:
        return PythonDictMergeEngine

    def select_data_by_column_names(
        self, data: List[Dict[str, Any]], selected_feature_names: Set[FeatureName]
    ) -> List[Dict[str, Any]]:
        if not data:
            raise ValueError(f"Data cannot be empty: {selected_feature_names}")

        # Get all unique column names from all rows
        column_names: Set[str] = set()
        for row in data:
            column_names.update(row.keys())

        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)

        return [{k: record.get(k) for k in _selected_feature_names if k in record} for record in data]

    def set_column_names(self) -> None:
        if self.data and isinstance(self.data, list) and len(self.data) > 0:
            # Get all unique column names from all rows
            all_columns: Set[str] = set()
            for row in self.data:
                if isinstance(row, dict):
                    all_columns.update(row.keys())
            self.column_names = all_columns
        else:
            raise ValueError("Data is empty or not in expected format. Cannot set column names.")

    def transform(self, data: Any, feature_names: Set[str]) -> List[Dict[str, Any]]:
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
            return transformed_data  # type: ignore

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

    def filter_engine(self) -> Type[BaseFilterEngine]:
        """
        Returns the filter engine for PythonDict framework.

        Returns:
            Type[BaseFilterEngine]: PythonDictFilterEngine class
        """
        return PythonDictFilterEngine
