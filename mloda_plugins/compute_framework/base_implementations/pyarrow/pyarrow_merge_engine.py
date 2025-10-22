from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine


class PyArrowMergeEngine(BaseMergeEngine):
    @staticmethod
    def _normalize_string_types(table: pa.Table, key_columns: list[str]) -> pa.Table:
        """
        Normalize string types in key columns to ensure join compatibility.

        PyArrow has both string and large_string types which are incompatible
        for join operations. This method casts all string-like types in the
        specified key columns to the standard string type.

        Args:
            table: PyArrow table to normalize
            key_columns: List of column names that are join keys

        Returns:
            pa.Table: Table with normalized string types in key columns
        """
        schema = table.schema
        new_columns = []
        new_fields = []

        for field in schema:
            column = table[field.name]

            # If this is a join key column and has a string-like type, normalize it
            if field.name in key_columns and (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
                # Cast to standard string type
                normalized_column = pc.cast(column, pa.string())
                new_columns.append(normalized_column)
                new_fields.append(pa.field(field.name, pa.string()))
            else:
                new_columns.append(column)
                new_fields.append(field)

        return pa.table(new_columns, schema=pa.schema(new_fields))

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
            left_keys = list(left_index.index)
            right_keys = list(right_index.index)
        else:
            if left_index.index[0] != right_index.index[0]:
                # PyArrow drops the index column in all cases.
                # Thus, we create a copy of the index column and append it to the right_data to avoid this in case of different index columns.
                _right_index = "mloda_right_index"
                if _right_index in right_data.column_names:
                    raise ValueError(f"Column name {_right_index} already exists in right_data.")

                right_data = right_data.append_column(_right_index, right_data[right_index.index[0]])
                left_keys = [left_index.index[0]]
                right_keys = [_right_index]
            else:
                left_keys = [left_index.index[0]]
                right_keys = [right_index.index[0]]

        # Normalize string types in join key columns to ensure compatibility
        # (e.g., string vs large_string are incompatible in PyArrow joins)
        left_data = self._normalize_string_types(left_data, left_keys)
        right_data = self._normalize_string_types(right_data, right_keys)

        left_data = left_data.join(
            right_data,
            keys=left_keys,
            right_keys=right_keys,
            join_type=join_type,
        )

        return left_data
