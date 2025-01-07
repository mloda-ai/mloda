from typing import Any

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine


class PyArrowMergeEngine(BaseMergeEngine):
    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.merge_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.merge_logic("left outer", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.merge_logic("right outer", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.merge_logic("full outer", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        if left_index == right_index:
            left_data = left_data.join(right_data, keys=left_index.index[0], join_type=join_type)
            return left_data
        else:
            raise ValueError(
                f"JoinType {join_type} {left_index} {right_index} are not yet implemented {self.__class__.__name__}"
            )
