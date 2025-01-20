from typing import Any

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine

try:
    import pandas as pd
except ImportError:
    pd = None


class PandasMergeEngine(BaseMergeEngine):
    def check_import(self) -> None:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("outer", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.pd_concat()([left_data, right_data], ignore_index=True)

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        combined = self.merge_append(left_data, right_data, left_index, right_index)
        return combined.drop_duplicates()

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        if left_index.is_multi_index() or right_index.is_multi_index():
            raise ValueError(f"MultiIndex is not yet implemented {self.__class__.__name__}")

        left_idx = left_index.index[0]
        right_idx = right_index.index[0]
        left_data = self.pd_merge()(left_data, right_data, left_on=left_idx, right_on=right_idx, how=join_type)
        return left_data

    @staticmethod
    def pd_merge() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.merge

    @staticmethod
    def pd_concat() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.concat
