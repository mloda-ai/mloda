from datetime import timedelta
from functools import reduce
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.user import Index
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.spark import spark_type_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name

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
        if isinstance(asof_config.tolerance, timedelta):
            raise ValueError(
                f"{self.__class__.__name__} ASOF does not support a timedelta tolerance; provide a numeric "
                "tolerance (e.g. epoch seconds matching the time column)."
            )
        left_data, right_data = self.validate_asof_time_columns(left_data, right_data, asof_config)
        self.check_import()
        if asof_config.direction == "nearest":
            raise ValueError("SparkMergeEngine asof does not support direction='nearest'.")

        by_left = list(left_index.index)
        by_right = list(right_index.index)
        lt = asof_config.left_time_column
        rt = asof_config.right_time_column

        left_cols = list(left_data.columns)
        right_cols = list(right_data.columns)
        # Right columns that survive in the output (left wins on a name collision).
        right_keep = [c for c in right_cols if c not in left_cols]

        # Rename every right column to a collision-free internal name BEFORE the join so the
        # join predicate, window and final projection never rely on Spark alias ("l."/"r.")
        # resolution surviving a withColumn/filter, which is brittle across Spark versions.
        prefix = "_mloda_r_"
        right_renamed = right_data
        for c in right_cols:
            right_renamed = right_renamed.withColumnRenamed(c, f"{prefix}{c}")

        left_ids = left_data.withColumn("_mloda_lid", F.monotonically_increasing_id())

        conditions = [F.col(lk) == F.col(f"{prefix}{rk}") for lk, rk in zip(by_left, by_right)]
        if asof_config.direction == "backward":
            if asof_config.allow_exact_matches:
                conditions.append(F.col(f"{prefix}{rt}") <= F.col(lt))
            else:
                conditions.append(F.col(f"{prefix}{rt}") < F.col(lt))
        else:
            if asof_config.allow_exact_matches:
                conditions.append(F.col(f"{prefix}{rt}") >= F.col(lt))
            else:
                conditions.append(F.col(f"{prefix}{rt}") > F.col(lt))

        if asof_config.tolerance is not None:
            conditions.append(F.abs(F.col(lt) - F.col(f"{prefix}{rt}")) <= float(asof_config.tolerance))

        condition = reduce(lambda a, b: a & b, conditions)
        joined = left_ids.join(right_renamed, condition, "left")

        # Rank candidate right rows per left row: nearest time first (direction-aware), ties
        # broken by the surviving right columns ascending so the winner is deterministic and
        # independent of input order (mirrors the sqlite and python_dict backends). Nulls sort
        # last so an unmatched left row keeps its single null-right row.
        time_col = F.col(f"{prefix}{rt}")
        if asof_config.direction == "backward":
            order_by = [time_col.desc_nulls_last()]
        else:
            order_by = [time_col.asc_nulls_last()]
        order_by += [F.col(f"{prefix}{c}").asc_nulls_last() for c in right_keep]

        window = Window.partitionBy("_mloda_lid").orderBy(*order_by)
        ranked = joined.withColumn("_mloda_rn", F.row_number().over(window)).filter(F.col("_mloda_rn") == 1)

        select_list = [F.col(c) for c in left_cols]
        select_list += [F.col(f"{prefix}{c}").alias(c) for c in right_keep]
        return ranked.select(*select_list)

    def _coerce_asof_time_column(self, data: Any, column: str) -> Any:
        """Coerce an ISO-8601 string time column to timestamp, failing hard on null introduction.

        to_timestamp silently yields NULL for unparseable values, so an eager count guards first.
        """
        helper = pick_helper_column_name(taken=set(data.columns), prefix="_mloda_coerced")
        coerced = data.withColumn(helper, F.to_timestamp(F.col(column)))
        bad_count = coerced.filter(F.col(column).isNotNull() & F.col(helper).isNull()).count()
        if bad_count > 0:
            raise ValueError(
                f"As-of time column '{column}' contains {bad_count} value(s) that are not ISO-8601 "
                f"and cannot be coerced to timestamp."
            )
        return data.withColumn(column, F.to_timestamp(F.col(column)))

    def _column_semantics(self, data: Any, column: str) -> ColumnSemantics:
        return spark_type_semantics.column_semantics(data, column)

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        dtype = dict(data.dtypes)[column]
        ordered_prefixes = (
            "tinyint",
            "smallint",
            "int",
            "bigint",
            "float",
            "double",
            "decimal",
            "timestamp",
            "date",
            "interval",
        )
        return any(dtype.startswith(p) for p in ordered_prefixes)

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
