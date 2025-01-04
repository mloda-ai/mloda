from typing import Any, Set
import pyarrow as pa

from mloda_core.abstract_plugins.components.cfw_transformer import ComputeFrameworkTransformMap
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType


try:
    import pandas as pd
except ImportError:
    pd = None


class PyarrowTable(ComputeFrameWork):
    @staticmethod
    def expected_data_framework() -> Any:
        return pa.Table

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

    def merge_inner(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("inner", right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("left outer", right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("right outer", right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outter(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("full outer", right_data, left_index, right_index, JoinType.OUTER)

    def framework_merge_function(
        self, join_type: str, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        if left_index == right_index:
            self.data = self.data.join(right_data, keys=left_index.index[0], join_type=join_type)
            return self.data
        else:
            raise ValueError(
                f"JoinType {join_type} {left_index} {right_index} are not yet implemented {self.__class__.__name__}"
            )
