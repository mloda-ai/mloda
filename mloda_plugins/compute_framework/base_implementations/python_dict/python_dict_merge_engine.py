from datetime import date, datetime, timedelta
from typing import Any
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda.user import Index
from mloda.user import JoinType


class PythonDictMergeEngine(BaseMergeEngine):
    """
    Merge engine for PythonDict framework using a COLUMNAR ``dict[str, list[Any]]``.

    Joins are computed over row-index views built from the columnar data and produce
    columnar dicts with the union of columns; the absent side is null-filled with ``None``.
    """

    def check_import(self) -> None:
        """
        No external dependencies required for PythonDict framework.
        """
        pass

    # -- columnar helpers ----------------------------------------------------

    @staticmethod
    def _row_count(data: dict[str, list[Any]]) -> int:
        for column in data.values():
            return len(column)
        return 0

    @classmethod
    def _to_rows(cls, data: dict[str, list[Any]]) -> list[dict[str, Any]]:
        n = cls._row_count(data)
        return [{col: data[col][i] for col in data} for i in range(n)]

    @staticmethod
    def _to_columnar(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, list[Any]]:
        return {col: [row.get(col) for row in rows] for col in columns}

    @staticmethod
    def _ordered_columns(*column_groups: Any) -> list[str]:
        ordered: list[str] = []
        for group in column_groups:
            for col in group:
                if col not in ordered:
                    ordered.append(col)
        return ordered

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("outer", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        columns = self._ordered_columns(left_data.keys(), right_data.keys())
        left_rows = self._to_rows(left_data)
        right_rows = self._to_rows(right_data)
        return self._to_columnar(left_rows + right_rows, columns)

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("union", left_data, right_data, left_index, right_index, JoinType.UNION)

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        left_data, right_data = self.validate_asof_time_columns(left_data, right_data, asof_config)
        left_by = list(left_index.index)
        right_by = list(right_index.index)
        lt, rt = asof_config.left_time_column, asof_config.right_time_column
        direction = asof_config.direction
        allow_exact = asof_config.allow_exact_matches
        tolerance = asof_config.tolerance

        left_rows = self._to_rows(left_data)
        right_rows = self._to_rows(right_data)

        right_by_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for r in right_rows:
            key = tuple(r.get(col) for col in right_by)
            right_by_groups.setdefault(key, []).append(r)

        left_keys = set(left_data.keys())
        right_extra_cols = [col for col in right_data.keys() if col not in left_keys]
        out_columns = self._ordered_columns(left_data.keys(), right_extra_cols)

        result_rows = []
        for left in left_rows:
            key = tuple(left.get(col) for col in left_by)
            left_time = left.get(lt)
            match = self._select_asof_match(
                right_by_groups.get(key, []), rt, left_time, direction, allow_exact, tolerance, right_by
            )
            merged = {**left}
            for col in right_extra_cols:
                merged[col] = match.get(col) if match is not None else None
            result_rows.append(merged)

        return self._to_columnar(result_rows, out_columns)

    def _coerce_asof_time_column(self, data: Any, column: str) -> Any:
        column_values = data.get(column, [])
        new_column: list[Any] = []
        parsed_values: list[datetime] = []
        for v in column_values:
            if v is None:
                new_column.append(None)
                continue
            if not isinstance(v, str):
                raise ValueError(
                    f"Cannot coerce as-of time column '{column}': expected ISO-8601 string or None, "
                    f"got {type(v).__name__}."
                )
            # Python 3.10's fromisoformat rejects a trailing 'Z'; normalize it to '+00:00'.
            parse_input = v[:-1] + "+00:00" if v.endswith("Z") else v
            parsed = datetime.fromisoformat(parse_input)
            new_column.append(parsed)
            parsed_values.append(parsed)
        has_naive = any(p.tzinfo is None for p in parsed_values)
        has_aware = any(p.tzinfo is not None for p in parsed_values)
        if has_naive and has_aware:
            raise ValueError(
                f"Cannot coerce as-of time column '{column}': mixed tz-naive and tz-aware values are not allowed."
            )
        coerced = dict(data)
        coerced[column] = new_column
        return coerced

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        for v in data.get(column, []):
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float, datetime, date, timedelta)):
                return False
        return True

    @staticmethod
    def _select_asof_match(
        candidates: list[dict[str, Any]],
        rt: str,
        left_time: Any,
        direction: str,
        allow_exact: bool,
        tolerance: float | int | timedelta | None,
        by_cols: list[str],
    ) -> dict[str, Any] | None:
        eligible = []
        for r in candidates:
            right_time = r.get(rt)
            if right_time is None or left_time is None:
                continue
            if direction == "backward":
                ok = right_time <= left_time if allow_exact else right_time < left_time
            elif direction == "forward":
                ok = right_time >= left_time if allow_exact else right_time > left_time
            else:
                ok = allow_exact or right_time != left_time
            if ok:
                eligible.append(r)

        if not eligible:
            return None

        # Pick the winning time per direction (largest for backward, smallest for forward,
        # nearest abs gap otherwise). max/min here only select the time value, not the row.
        if direction == "backward":
            best_time = max(r[rt] for r in eligible)
            tied = [r for r in eligible if r[rt] == best_time]
        elif direction == "forward":
            best_time = min(r[rt] for r in eligible)
            tied = [r for r in eligible if r[rt] == best_time]
        else:
            best_gap = min(abs(r[rt] - left_time) for r in eligible)
            tied = [r for r in eligible if abs(r[rt] - left_time) == best_gap]

        # Tie-break deterministically and independent of input order: among rows tied on the
        # primary time criterion, the smallest right non-key column values win (ascending),
        # mirroring the sqlite backend.
        by_set = set(by_cols)
        non_key_cols = sorted({c for r in tied for c in r.keys() if c not in by_set})
        best = min(tied, key=lambda r: tuple(r.get(c) for c in non_key_cols))

        if tolerance is not None and abs(best[rt] - left_time) > tolerance:
            return None
        return best

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        """
        Common join logic for all join types.

        Args:
            join_type: Type of join ("inner", "left", "right", "outer", "union")
            left_data: Left dataset as a columnar dict
            right_data: Right dataset as a columnar dict
            left_index: Index object containing left join columns
            right_index: Index object containing right join columns
            jointype: JoinType enum value

        Returns:
            dict[str, list[Any]]: Joined result as a columnar dict
        """
        left_cols = left_index.index
        right_cols = right_index.index

        if join_type == "inner":
            return self._inner_join(left_data, right_data, left_cols, right_cols)
        elif join_type == "left":
            return self._left_join(left_data, right_data, left_cols, right_cols)
        elif join_type == "right":
            return self._right_join(left_data, right_data, left_cols, right_cols)
        elif join_type == "outer":
            return self._outer_join(left_data, right_data, left_cols, right_cols)
        elif join_type == "union":
            return self._union_join(left_data, right_data, left_cols, right_cols)
        else:
            raise ValueError(f"Join type {join_type} is not supported")

    def _inner_join(
        self, left_data: Any, right_data: Any, left_cols: tuple[str, ...], right_cols: tuple[str, ...]
    ) -> Any:
        """Performs inner join."""
        left_rows = self._to_rows(left_data)
        right_rows = self._to_rows(right_data)
        right_index_map = {tuple(r.get(col) for col in right_cols): r for r in right_rows}

        out_columns = self._ordered_columns(left_data.keys(), right_data.keys())
        result = []
        for left in left_rows:
            key = tuple(left.get(col) for col in left_cols)
            if key in right_index_map:
                result.append({**left, **right_index_map[key]})

        return self._to_columnar(result, out_columns)

    def _left_join(
        self, left_data: Any, right_data: Any, left_cols: tuple[str, ...], right_cols: tuple[str, ...]
    ) -> Any:
        """Performs left join."""
        left_rows = self._to_rows(left_data)
        right_rows = self._to_rows(right_data)
        right_index_map = {tuple(r.get(col) for col in right_cols): r for r in right_rows}

        out_columns = self._ordered_columns(left_data.keys(), right_data.keys())
        right_columns = [col for col in right_data.keys() if col not in set(right_cols) and col not in left_data]

        result = []
        for left in left_rows:
            key = tuple(left.get(col) for col in left_cols)
            if key in right_index_map:
                result.append({**left, **right_index_map[key]})
            else:
                merged = {**left}
                for col in right_columns:
                    merged[col] = None
                result.append(merged)

        return self._to_columnar(result, out_columns)

    def _right_join(
        self, left_data: Any, right_data: Any, left_cols: tuple[str, ...], right_cols: tuple[str, ...]
    ) -> Any:
        """Performs right join."""
        left_rows = self._to_rows(left_data)
        right_rows = self._to_rows(right_data)
        left_index_map = {tuple(left.get(col) for col in left_cols): left for left in left_rows}

        out_columns = self._ordered_columns(right_data.keys(), left_data.keys())
        left_columns = [col for col in left_data.keys() if col not in set(left_cols) and col not in right_data]

        result = []
        for r in right_rows:
            key = tuple(r.get(col) for col in right_cols)
            if key in left_index_map:
                result.append({**left_index_map[key], **r})
            else:
                merged = {**r}
                for col in left_columns:
                    merged[col] = None
                result.append(merged)

        return self._to_columnar(result, out_columns)

    def _outer_join(
        self, left_data: Any, right_data: Any, left_cols: tuple[str, ...], right_cols: tuple[str, ...]
    ) -> Any:
        """Performs outer join."""
        left_rows = self._to_rows(left_data)
        right_rows = self._to_rows(right_data)
        left_index_map = {tuple(left.get(col) for col in left_cols): left for left in left_rows}
        right_index_map = {tuple(r.get(col) for col in right_cols): r for r in right_rows}

        all_keys = list(left_index_map.keys())
        for key in right_index_map.keys():
            if key not in left_index_map:
                all_keys.append(key)

        left_columns = list(left_data.keys())
        right_columns = list(right_data.keys())
        out_columns = self._ordered_columns(left_columns, right_columns)

        result = []
        for key in all_keys:
            left_row = left_index_map.get(key, {})
            right_row = right_index_map.get(key, {})

            merged: dict[str, Any] = {}

            # Start with the key values
            if left_cols == right_cols:
                for i, col in enumerate(left_cols):
                    merged[col] = key[i]
            else:
                if key in left_index_map:
                    for i, col in enumerate(left_cols):
                        merged[col] = key[i]
                if key in right_index_map:
                    for i, col in enumerate(right_cols):
                        merged[col] = key[i]

            for col in left_columns:
                if col not in left_cols:
                    merged[col] = left_row.get(col)

            for col in right_columns:
                if col not in right_cols:
                    merged[col] = right_row.get(col)

            result.append(merged)

        return self._to_columnar(result, out_columns)

    def _union_join(
        self, left_data: Any, right_data: Any, left_cols: tuple[str, ...], right_cols: tuple[str, ...]
    ) -> Any:
        """Performs union (removes duplicates based on join columns)."""
        left_rows = self._to_rows(left_data)
        right_rows = self._to_rows(right_data)
        out_columns = self._ordered_columns(left_data.keys(), right_data.keys())

        seen_keys: set[Any] = set()
        result = []

        for row in left_rows:
            key = tuple(row.get(col) for col in left_cols)
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(row)

        for row in right_rows:
            key = tuple(row.get(col) for col in right_cols)
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(row)

        return self._to_columnar(result, out_columns)
