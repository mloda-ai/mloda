from typing import Any, Union

from mloda.user import Index
from mloda.user import JoinType
from mloda.provider import BaseMergeEngine

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
        left_idx: Union[str, list[str]]
        right_idx: Union[str, list[str]]
        if left_index.is_multi_index() or right_index.is_multi_index():
            left_idx = list(left_index.index)
            right_idx = list(right_index.index)
        else:
            left_idx = left_index.index[0]
            right_idx = right_index.index[0]

        left_data = self.pd_merge()(left_data, right_data, left_on=left_idx, right_on=right_idx, how=join_type)
        return left_data

    @classmethod
    def pd_merge(cls) -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.merge

    @classmethod
    def pd_concat(cls) -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.concat
