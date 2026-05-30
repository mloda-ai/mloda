from functools import reduce
from typing import Any

from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.user import Index
from mloda.provider import BaseMergeEngine

try:
    from pyspark.sql import DataFrame, Window
    import pyspark.sql.functions as F
except ImportError:
    DataFrame = None
    Window = None
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

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        self.check_import()
        if asof_config.direction == "nearest":
            raise ValueError("SparkMergeEngine asof does not support direction='nearest'.")

        by_left = list(left_index.index)
        by_right = list(right_index.index)
        lt = asof_config.left_time_column
        rt = asof_config.right_time_column

        left_ids = left_data.withColumn("_mloda_lid", F.monotonically_increasing_id())
        left_alias = left_ids.alias("l")
        right_alias = right_data.alias("r")

        conditions = [F.col(f"l.{lk}") == F.col(f"r.{rk}") for lk, rk in zip(by_left, by_right)]
        if asof_config.direction == "backward":
            if asof_config.allow_exact_matches:
                conditions.append(F.col(f"r.{rt}") <= F.col(f"l.{lt}"))
            else:
                conditions.append(F.col(f"r.{rt}") < F.col(f"l.{lt}"))
        else:
            if asof_config.allow_exact_matches:
                conditions.append(F.col(f"r.{rt}") >= F.col(f"l.{lt}"))
            else:
                conditions.append(F.col(f"r.{rt}") > F.col(f"l.{lt}"))

        if asof_config.tolerance is not None:
            conditions.append(F.abs(F.col(f"l.{lt}") - F.col(f"r.{rt}")) <= float(asof_config.tolerance))

        condition = reduce(lambda a, b: a & b, conditions)
        joined = left_alias.join(right_alias, condition, "left")

        if asof_config.direction == "backward":
            order = F.col(f"r.{rt}").desc_nulls_last()
        else:
            order = F.col(f"r.{rt}").asc_nulls_last()
        window = Window.partitionBy("_mloda_lid").orderBy(order)
        ranked = joined.withColumn("_mloda_rn", F.row_number().over(window)).filter(F.col("_mloda_rn") == 1)

        select_list = [F.col(f"l.{c}").alias(c) for c in left_data.columns]
        select_list += [F.col(f"r.{c}").alias(c) for c in right_data.columns if c not in left_data.columns]
        return ranked.select(*select_list)

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
    ) -> tuple[Any, Any]:
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
