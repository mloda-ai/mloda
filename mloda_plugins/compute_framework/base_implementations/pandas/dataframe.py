from typing import Any, Dict, Set, Tuple, Type
import pyarrow as pa
from mloda_core.abstract_plugins.components.cfw_transformer import ComputeFrameworkTransformMap
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine

try:
    import pandas as pd
except ImportError:
    pd = None

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType


class PandasDataframe(ComputeFrameWork):
    @staticmethod
    def pd_dataframe() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.DataFrame

    @staticmethod
    def pd_series() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.Series

    @staticmethod
    def pd_merge() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.merge

    @staticmethod
    def expected_data_framework() -> Any:
        return PandasDataframe.pd_dataframe()

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

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        return data[[f.name for f in selected_feature_names]]

    def set_column_names(self) -> None:
        self.column_names = set(self.data.columns)

    def merge_inner(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("inner", right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("left", right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("right", right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outter(self, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.framework_merge_function("outer", right_data, left_index, right_index, JoinType.OUTER)

    def framework_merge_function(
        self, join_type: str, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        if left_index.is_multi_index() or right_index.is_multi_index():
            raise ValueError(f"MultiIndex is not yet implemented {self.__class__.__name__}")

        if left_index == right_index:
            left_idx = left_index.index[0]
            right_idx = right_index.index[0]
            self.data = self.pd_merge()(self.data, right_data, left_on=left_idx, right_on=right_idx, how=join_type)
            return self.data

        else:
            raise ValueError(
                f"JoinType {join_type} {left_index} {right_index} are not yet implemented {self.__class__.__name__}"
            )

    @staticmethod
    def get_framework_transform_functions(from_other: bool, other: Any) -> Tuple[Any, Dict[str, Any]]:
        def pyarrow_table_to_pandas_dataframe(data: pd.DataFrame, parameters: Dict[str, Any] = {}) -> pa.Table:
            return pa.Table.to_pandas(data, **parameters)

        def pandas_dataframe_to_pyarrow_table(data: pd.DataFrame, parameters: Dict[str, Any] = {}) -> pa.Table:
            return pa.Table.from_pandas(data, **parameters)

        if other == pa.Table:
            if from_other:
                return pyarrow_table_to_pandas_dataframe, {}
            return pandas_dataframe_to_pyarrow_table, {}

        return None, {"defensive": None}

    def filter_engine(self) -> Type[BaseFilterEngine]:
        return PandasFilterEngine
