from typing import Any, Set
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType


class PythonDictMergeEngine(BaseMergeEngine):
    """
    Merge engine for PythonDict framework using List[Dict[str, Any]] data structure.

    Implements hash-based joins using dictionary lookups.
    """

    def check_import(self) -> None:
        """
        No external dependencies required for PythonDict framework.
        """
        pass

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("outer", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return left_data + right_data

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("union", left_data, right_data, left_index, right_index, JoinType.UNION)

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        """
        Common join logic for all join types.

        Args:
            join_type: Type of join ("inner", "left", "right", "outer", "union")
            left_data: Left dataset as List[Dict]
            right_data: Right dataset as List[Dict]
            left_index: Index object containing left join columns
            right_index: Index object containing right join columns
            jointype: JoinType enum value

        Returns:
            List[Dict]: Joined result
        """
        if left_index.is_multi_index() or right_index.is_multi_index():
            raise ValueError(f"MultiIndex is not yet implemented {self.__class__.__name__}")

        left_col = left_index.index[0]
        right_col = right_index.index[0]

        if join_type == "inner":
            return self._inner_join(left_data, right_data, left_col, right_col)
        elif join_type == "left":
            return self._left_join(left_data, right_data, left_col, right_col)
        elif join_type == "right":
            return self._right_join(left_data, right_data, left_col, right_col)
        elif join_type == "outer":
            return self._outer_join(left_data, right_data, left_col, right_col)
        elif join_type == "union":
            return self._union_join(left_data, right_data, left_col, right_col)
        else:
            raise ValueError(f"Join type {join_type} is not supported")

    def _inner_join(self, left_data: Any, right_data: Any, left_col: str, right_col: str) -> Any:
        """Performs inner join."""
        right_index_map = {r.get(right_col): r for r in right_data}

        result = []
        for l in left_data:
            key = l.get(left_col)
            if key in right_index_map:
                merged = {**l, **right_index_map[key]}
                result.append(merged)

        return result

    def _left_join(self, left_data: Any, right_data: Any, left_col: str, right_col: str) -> Any:
        """Performs left join."""
        right_index_map = {r.get(right_col): r for r in right_data}

        # Get right columns for null filling
        right_columns = set()
        for r in right_data:
            right_columns.update(r.keys())
        right_columns = right_columns - {right_col}

        left_columns = set()
        for l in left_data:
            left_columns.update(l.keys())
        right_columns = right_columns - left_columns

        result = []
        for l in left_data:
            key = l.get(left_col)
            if key in right_index_map:
                merged = {**l, **right_index_map[key]}
                result.append(merged)
            else:
                merged = {**l}
                for col in right_columns:
                    merged[col] = None
                result.append(merged)

        return result

    def _right_join(self, left_data: Any, right_data: Any, left_col: str, right_col: str) -> Any:
        """Performs right join."""
        left_index_map = {l.get(left_col): l for l in left_data}

        # Get left columns for null filling
        left_columns = set()
        for l in left_data:
            left_columns.update(l.keys())
        left_columns = left_columns - {left_col}

        right_columns = set()
        for r in right_data:
            right_columns.update(r.keys())
        left_columns = left_columns - right_columns

        result = []
        for r in right_data:
            key = r.get(right_col)
            if key in left_index_map:
                merged = {**left_index_map[key], **r}
                result.append(merged)
            else:
                merged = {**r}
                for col in left_columns:
                    merged[col] = None
                result.append(merged)

        return result

    def _outer_join(self, left_data: Any, right_data: Any, left_col: str, right_col: str) -> Any:
        """Performs outer join."""
        left_index_map = {l.get(left_col): l for l in left_data}
        right_index_map = {r.get(right_col): r for r in right_data}

        all_keys = set(left_index_map.keys()) | set(right_index_map.keys())

        # Get all column names
        left_columns = set()
        right_columns = set()
        for l in left_data:
            left_columns.update(l.keys())
        for r in right_data:
            right_columns.update(r.keys())

        result = []
        for key in all_keys:
            left_row = left_index_map.get(key, {})
            right_row = right_index_map.get(key, {})

            merged = {}

            # Start with the key value
            if left_col == right_col:
                merged[left_col] = key
            else:
                if key in left_index_map:
                    merged[left_col] = key
                if key in right_index_map:
                    merged[right_col] = key

            # Add left columns (excluding join column)
            for col in left_columns:
                if col != left_col:
                    merged[col] = left_row.get(col)

            # Add right columns (excluding join column)
            for col in right_columns:
                if col != right_col:
                    merged[col] = right_row.get(col)

            result.append(merged)

        return result

    def _union_join(self, left_data: Any, right_data: Any, left_col: str, right_col: str) -> Any:
        """Performs union (removes duplicates based on join columns)."""
        seen_keys: Set[Any] = set()
        result = []

        # Add left rows
        for row in left_data:
            key = row.get(left_col)
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(row)

        # Add right rows that haven't been seen
        for row in right_data:
            key = row.get(right_col)
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(row)

        return result
