from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.user import Index
from mloda.user import JoinType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    is_ordered_arrow_type,
    pick_helper_column_name,
)


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
        raise ValueError(f"JoinType union is not yet implemented in {self.__class__.__name__}")

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        self.validate_asof_time_columns(left_data, right_data, asof_config)
        if asof_config.direction == "nearest":
            raise ValueError(f"{self.__class__.__name__} asof does not support direction='nearest'.")

        if asof_config.allow_exact_matches is False:
            raise ValueError(
                f"{self.__class__.__name__} asof does not support allow_exact_matches=False; "
                "Acero's match range always includes exact matches."
            )

        tol = asof_config.tolerance
        if tol is not None:
            is_integer = isinstance(tol, int) and not isinstance(tol, bool)
            is_integer_valued_float = isinstance(tol, float) and tol.is_integer()
            if not (is_integer or is_integer_valued_float):
                raise ValueError(
                    f"{self.__class__.__name__} asof requires an integer tolerance; "
                    "timedelta, boolean and non-integer tolerances are not supported."
                )

        by_left = list(left_index.index)
        by_right = list(right_index.index)
        lt, rt = asof_config.left_time_column, asof_config.right_time_column

        left_cols = list(left_data.column_names)
        right_cols = list(right_data.column_names)

        right_match = set(by_right) | {rt}
        right_value_keep = [c for c in right_cols if c not in right_match and c not in left_cols]
        right_key_carry = [c for c in right_match if c not in left_cols]

        right_select = by_right + [rt] + right_value_keep
        right_join = right_data.select(right_select)

        taken = set(left_cols) | set(right_join.column_names)
        carry_to_original: dict[str, str] = {}
        for c in right_key_carry:
            carry_name = pick_helper_column_name(taken=taken, prefix="mloda_asof_carry")
            taken.add(carry_name)
            right_join = right_join.append_column(carry_name, right_data[c])
            carry_to_original[carry_name] = c

        left_data = self._normalize_string_types(left_data, by_left)
        right_join = self._normalize_string_types(right_join, by_right)

        if tol is not None:
            magnitude = int(cast(float, tol))
        else:
            left_on_i = pc.cast(left_data[lt], pa.int64())
            right_on_i = pc.cast(right_join[rt], pa.int64())
            combined = pa.concat_arrays([left_on_i.combine_chunks(), right_on_i.combine_chunks()])
            mm = pc.min_max(combined).as_py()
            magnitude = 0 if mm["min"] is None or mm["max"] is None else (mm["max"] - mm["min"])

        signed_tol = -magnitude if asof_config.direction == "backward" else magnitude
        _INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
        signed_tol = max(_INT64_MIN, min(signed_tol, _INT64_MAX))

        result = left_data.sort_by(lt).join_asof(
            right_join.sort_by(rt),
            on=lt,
            by=by_left,
            right_on=rt,
            right_by=by_right,
            tolerance=signed_tol,
        )

        if carry_to_original:
            new_names = [carry_to_original.get(name, name) for name in result.column_names]
            result = result.rename_columns(new_names)

        return result

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        t = data.schema.field(column).type
        return is_ordered_arrow_type(t)

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
                _right_index = pick_helper_column_name(taken=set(right_data.column_names), prefix="mloda_right_index")
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
