from typing import Any, Set, Type
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
import pyarrow as pa

from mloda_core.abstract_plugins.components.cfw_transformer import ComputeFrameworkTransformMap
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork


try:
    import pandas as pd
except ImportError:
    pd = None


class PyarrowTable(ComputeFrameWork):
    @staticmethod
    def expected_data_framework() -> Any:
        return pa.Table

    def merge_engine(self) -> Type[BaseMergeEngine]:
        return PyArrowMergeEngine

    def transform(
        self,
        data: Any,
        feature_names: Set[str],
        transform_map: ComputeFrameworkTransformMap = ComputeFrameworkTransformMap(),
    ) -> Any:
        transformed_data = self.transform_refactored(data, transform_map)
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

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        return data.select([f.name for f in selected_feature_names])

    def set_column_names(self) -> None:
        self.column_names = set(self.data.schema.names)
