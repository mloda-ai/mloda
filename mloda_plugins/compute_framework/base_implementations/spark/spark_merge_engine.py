from typing import Any, Tuple

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine

try:
    from pyspark.sql import DataFrame
    import pyspark.sql.functions as F
except ImportError:
    DataFrame = None
    F = None


class SparkMergeEngine(BaseMergeEngine):
    def check_import(self) -> None:
        if DataFrame is None:
            raise ImportError("PySpark is not installed. To be able to use this framework, please install pyspark.")

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._join_logic("inner", left_data, right_data, left_index, right_index)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._join_logic("left", left_data, right_data, left_index, right_index)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._join_logic("right", left_data, right_data, left_index, right_index)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self._join_logic("outer", left_data, right_data, left_index, right_index)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        """Append (union all) two DataFrames."""
        return left_data.unionAll(right_data)

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        """Union two DataFrames (removes duplicates)."""
        return left_data.union(right_data).distinct()

    def _join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index
    ) -> Any:
        """Execute join logic for Spark DataFrames."""
        if left_index.is_multi_index() or right_index.is_multi_index():
            raise ValueError(f"MultiIndex is not yet implemented {self.__class__.__name__}")

        # Get the index column names
        left_idx = left_index.index[0]
        right_idx = right_index.index[0]

        # Handle case where index columns have the same name
        if left_idx == right_idx:
            # Join on the same column name
            join_condition = left_idx
        else:
            # Join on different column names
            join_condition = left_data[left_idx] == right_data[right_idx]

        return left_data.join(right_data, join_condition, join_type)

    def _handle_column_conflicts(
        self, left_data: Any, right_data: Any, left_index: Index, right_index: Index
    ) -> Tuple[Any, Any]:
        """Handle column name conflicts by renaming columns in right DataFrame."""
        left_columns = set(left_data.columns)
        right_columns = set(right_data.columns)

        # Find conflicting columns (excluding join keys)
        left_idx = left_index.index[0]
        right_idx = right_index.index[0]

        conflicts = (left_columns & right_columns) - {left_idx, right_idx}

        if conflicts:
            # Rename conflicting columns in right DataFrame
            for col in conflicts:
                right_data = right_data.withColumnRenamed(col, f"{col}_right")

        return left_data, right_data
