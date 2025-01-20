from typing import Any

import pyarrow as pa

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine


class PyArrowMergeEngine(BaseMergeEngine):
    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left outer", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right outer", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("full outer", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        # Ensure the schemas of both tables match before appending
        if left_data.schema != right_data.schema:
            raise ValueError("Schemas of the tables do not match for append operation.")
        return pa.concat_tables([left_data, right_data])

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        """
        https://github.com/apache/arrow/issues/30950 Currently, not existing in base pyarrow.
        If needed, one could add it.
        """
        raise ValueError(f"JoinType union are not yet implemented {self.__class__.__name__}")

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        if left_index.is_multi_index() or right_index.is_multi_index():
            raise ValueError(f"MultiIndex is not yet implemented {self.__class__.__name__}")

        if left_index.index[0] != right_index.index[0]:
            # PyArrow drops the index column in all cases.
            # Thus, we create a copy of the index column and append it to the right_data to avoid this in case of different index columns.
            _right_index = "mloda_right_index"
            if _right_index in right_data.column_names:
                raise ValueError(f"Column name {_right_index} already exists in right_data.")

            right_data = right_data.append_column(_right_index, right_data[right_index.index[0]])
        else:
            _right_index = right_index.index[0]

        left_data = left_data.join(
            right_data,
            keys=[left_index.index[0]],
            right_keys=[_right_index],
            join_type=join_type,
        )

        return left_data
