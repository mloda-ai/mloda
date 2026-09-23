# Annotations stay strings so the pa.Table signatures below do not dereference pa at import time.
from __future__ import annotations

from typing import Any, cast

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.user import Index
from mloda.user import JoinType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow import pyarrow_type_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    is_ordered_arrow_type,
    pick_helper_column_name,
)

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]
    pc = None


class PyArrowMergeEngine(BaseMergeEngine):
    provides_column_semantics = True

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

    @staticmethod
    def _joins_natively(t: pa.DataType, asof: bool) -> bool:
        """Whether Arrow's join/asof-join kernels can carry this payload type without a row-index swap."""
        always = (
            pa.types.is_boolean(t)
            or pa.types.is_integer(t)
            or pa.types.is_float32(t)
            or pa.types.is_float64(t)
            or pa.types.is_date(t)
            or pa.types.is_time(t)
            or pa.types.is_timestamp(t)
            or pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_binary(t)
            or pa.types.is_large_binary(t)
        )
        if always:
            return True
        if asof:
            return False
        if (
            pa.types.is_float16(t)
            or pa.types.is_decimal(t)
            or pa.types.is_duration(t)
            or pa.types.is_interval(t)
            or pa.types.is_fixed_size_binary(t)
        ):
            return True
        if pa.types.is_dictionary(t):
            return PyArrowMergeEngine._joins_natively(t.value_type, asof)
        if isinstance(t, pa.BaseExtensionType):
            return PyArrowMergeEngine._joins_natively(t.storage_type, asof)
        return False

    @staticmethod
    def _without_views(t: pa.DataType) -> pa.DataType:
        """Replace string_view/binary_view with large_string/large_binary, recursing through nested types."""
        if pa.types.is_string_view(t):
            return pa.large_string()
        if pa.types.is_binary_view(t):
            return pa.large_binary()
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            value_type = PyArrowMergeEngine._without_views(t.value_type)
            return pa.large_list(value_type) if pa.types.is_large_list(t) else pa.list_(value_type)
        if pa.types.is_fixed_size_list(t):
            return pa.list_(PyArrowMergeEngine._without_views(t.value_type), t.list_size)
        if pa.types.is_list_view(t) or pa.types.is_large_list_view(t):
            value_field = t.value_field.with_type(PyArrowMergeEngine._without_views(t.value_field.type))
            return pa.large_list_view(value_field) if pa.types.is_large_list_view(t) else pa.list_view(value_field)
        if pa.types.is_struct(t):
            return pa.struct([f.with_type(PyArrowMergeEngine._without_views(f.type)) for f in t])
        if pa.types.is_map(t):
            key_field = t.key_field.with_type(PyArrowMergeEngine._without_views(t.key_field.type))
            item_field = t.item_field.with_type(PyArrowMergeEngine._without_views(t.item_field.type))
            return pa.map_(key_field, item_field, keys_sorted=t.keys_sorted)
        return t

    @staticmethod
    def _swap_out_payload_columns(
        table: pa.Table,
        key_columns: list[str],
        taken: set[str],
        placeholders: dict[str, tuple[str, pa.ChunkedArray]],
        asof: bool,
    ) -> pa.Table:
        """Swap payload columns the join cannot carry for row-index placeholders."""
        row_index: pa.Array | None = None
        for i, field in enumerate(table.schema):
            if field.name in key_columns:
                continue
            if PyArrowMergeEngine._joins_natively(field.type, asof):
                continue
            if row_index is None:
                n = table.num_rows
                row_index = pc.subtract(pc.cumulative_sum(pa.repeat(pa.scalar(1, pa.int64()), n)), 1)
            placeholder_name = pick_helper_column_name(taken=taken, prefix="mloda_payload_row")
            taken.add(placeholder_name)
            placeholders[placeholder_name] = (field.name, table.column(i))
            table = table.set_column(i, placeholder_name, row_index)
        return table

    @classmethod
    def _take_rows(cls, column: pa.ChunkedArray, idx: pa.ChunkedArray) -> pa.ChunkedArray:
        """Take rows, working around missing take kernels for run-end-encoded, view and nested view types."""
        t = column.type
        if pa.types.is_run_end_encoded(t):
            taken = cls._take_rows(pc.run_end_decode(column), idx)
            run_end_type = t.run_end_type if len(taken) < 1 << (t.run_end_type.bit_width - 1) else pa.int64()
            return pc.run_end_encode(taken, run_end_type=run_end_type)
        # No take kernel for view types (nor nested types containing them): cast to their
        # non-view equivalent, take, and cast back.
        plain = cls._without_views(t)
        if plain != t:
            return column.cast(plain).take(idx).cast(t)
        return column.take(idx)

    @classmethod
    def _restore_payload_columns(
        cls, table: pa.Table, placeholders: dict[str, tuple[str, pa.ChunkedArray]]
    ) -> pa.Table:
        """Replace row-index placeholders with the original payload columns, reindexed by the join result."""
        for i, name in enumerate(table.column_names):
            if name in placeholders:
                original_name, original_column = placeholders[name]
                table = table.set_column(i, original_name, cls._take_rows(original_column, table.column(i)))
        return table

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
        left_data, right_data = self.validate_asof_time_columns(left_data, right_data, asof_config)
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

        placeholders: dict[str, tuple[str, pa.ChunkedArray]] = {}
        left_data = self._swap_out_payload_columns(left_data, by_left + [lt], taken, placeholders, asof=True)
        right_join = self._swap_out_payload_columns(right_join, by_right + [rt], taken, placeholders, asof=True)

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

        result = self._restore_payload_columns(result, placeholders)

        if carry_to_original:
            new_names = [carry_to_original.get(name, name) for name in result.column_names]
            result = result.rename_columns(new_names)

        return result

    def _coerce_asof_time_column(self, data: Any, column: str) -> Any:
        index = data.schema.get_field_index(column)
        casted = pc.cast(data.column(column), pa.timestamp("us"))
        return data.set_column(index, pa.field(column, pa.timestamp("us")), casted)

    def _column_semantics(self, data: Any, column: str) -> ColumnSemantics:
        return pyarrow_type_semantics.column_semantics(data, column)

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        t = data.schema.field(column).type
        return is_ordered_arrow_type(t)

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        helper_columns: set[str] = set()
        if left_index.is_multi_index() or right_index.is_multi_index():
            left_keys = list(left_index.index)
            right_keys = list(right_index.index)
        elif left_index.index[0] != right_index.index[0]:
            # Join on helper copies of BOTH keys so Arrow leaves the original lk/rk columns
            # untouched; each then keeps its real value or NULL on its unmatched side.
            taken = set(left_data.column_names) | set(right_data.column_names)
            left_helper = pick_helper_column_name(taken=taken, prefix="mloda_left_join_key")
            right_helper = pick_helper_column_name(taken=taken | {left_helper}, prefix="mloda_right_join_key")
            left_data = left_data.append_column(left_helper, left_data[left_index.index[0]])
            right_data = right_data.append_column(right_helper, right_data[right_index.index[0]])
            left_keys = [left_helper]
            right_keys = [right_helper]
            helper_columns = {left_helper, right_helper}
        else:
            left_keys = [left_index.index[0]]
            right_keys = [right_index.index[0]]

        # Normalize string types in join key columns to ensure compatibility
        # (e.g., string vs large_string are incompatible in PyArrow joins)
        left_data = self._normalize_string_types(left_data, left_keys)
        right_data = self._normalize_string_types(right_data, right_keys)

        taken = set(left_data.column_names) | set(right_data.column_names)
        placeholders: dict[str, tuple[str, pa.ChunkedArray]] = {}
        left_data = self._swap_out_payload_columns(left_data, left_keys, taken, placeholders, asof=False)
        right_data = self._swap_out_payload_columns(right_data, right_keys, taken, placeholders, asof=False)

        left_data = left_data.join(
            right_data,
            keys=left_keys,
            right_keys=right_keys,
            join_type=join_type,
        )

        left_data = self._restore_payload_columns(left_data, placeholders)

        if helper_columns:
            # Drop helpers by index: a shared non-key column name would make a name-based select ambiguous.
            keep = [i for i, name in enumerate(left_data.column_names) if name not in helper_columns]
            left_data = left_data.select(keep)

        return left_data
