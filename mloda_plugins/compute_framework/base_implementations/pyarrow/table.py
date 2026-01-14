from typing import Any, Set, Type
from mloda.provider import BaseMergeEngine
from mloda.provider import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine
import pyarrow as pa

from mloda.user import FeatureName
from mloda.provider import ComputeFramework


try:
    import pandas as pd
except ImportError:
    pd = None


class PyArrowTable(ComputeFramework):
    @staticmethod
    def is_available() -> bool:
        """Check if PyArrow is installed and available."""
        try:
            import pyarrow

            return True
        except ImportError:
            return False

    @classmethod
    def expected_data_framework(cls) -> Any:
        return pa.Table

    @classmethod
    def merge_engine(cls) -> Type[BaseMergeEngine]:
        return PyArrowMergeEngine

    @classmethod
    def filter_engine(cls) -> Type[BaseFilterEngine]:
        return PyArrowFilterEngine

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        column_names = set(data.schema.names)
        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)
        return data.select([f for f in _selected_feature_names])

    def set_column_names(self) -> None:
        self.column_names = set(self.data.schema.names)

    def transform(
        self,
        data: Any,
        feature_names: Set[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to table"""
            return pa.table(data)

        if isinstance(data, pa.ChunkedArray) or isinstance(data, pa.Array):
            """Added data: Add column to table"""
            if len(feature_names) == 1:
                return self.data.append_column(next(iter(feature_names)), data)
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")
