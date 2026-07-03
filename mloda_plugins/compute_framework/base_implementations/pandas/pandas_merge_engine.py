from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.user import Index
from mloda.user import JoinType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pandas import pandas_type_semantics

try:
    import pandas as pd
    import pandas.api.types as pdt
except ImportError:
    pd = None
    pdt = None


class PandasMergeEngine(BaseMergeEngine):
    provides_column_semantics = True

    def check_import(self) -> None:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("outer", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.pd_concat()([left_data, right_data], ignore_index=True)

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        combined = self.merge_append(left_data, right_data, left_index, right_index)
        return combined.drop_duplicates()

    def merge_asof(
        self,
        left_data: Any,
        right_data: Any,
        left_index: Index,
        right_index: Index,
        asof_config: AsOfJoinConfig,
    ) -> Any:
        left_data, right_data = self.validate_asof_time_columns(left_data, right_data, asof_config)
        by_left = list(left_index.index)
        by_right = list(right_index.index)
        lt, rt = asof_config.left_time_column, asof_config.right_time_column
        left_sorted = left_data.sort_values(lt)
        right_sorted = right_data.sort_values(rt)
        kwargs: dict[str, Any] = {
            "direction": asof_config.direction,
            "allow_exact_matches": asof_config.allow_exact_matches,
        }
        if asof_config.tolerance is not None:
            kwargs["tolerance"] = asof_config.tolerance
        return self.pd_merge_asof()(
            left_sorted,
            right_sorted,
            left_on=lt,
            right_on=rt,
            left_by=by_left,
            right_by=by_right,
            **kwargs,
        )

    def _column_semantics(self, data: Any, column: str) -> ColumnSemantics:
        return pandas_type_semantics.column_semantics(data, column)

    def _coerce_asof_time_column(self, data: Any, column: str) -> Any:
        return data.assign(**{column: pd.to_datetime(data[column], format="ISO8601")})

    def _asof_time_column_is_ordered(self, data: Any, column: str) -> bool:
        dtype = data[column].dtype
        return bool(
            (pdt.is_numeric_dtype(dtype) or pdt.is_datetime64_any_dtype(dtype) or pdt.is_timedelta64_dtype(dtype))
            and not pdt.is_bool_dtype(dtype)
        )

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        left_idx: str | list[str]
        right_idx: str | list[str]
        if left_index.is_multi_index() or right_index.is_multi_index():
            left_idx = list(left_index.index)
            right_idx = list(right_index.index)
        else:
            left_idx = left_index.index[0]
            right_idx = right_index.index[0]

        left_data = self.pd_merge()(left_data, right_data, left_on=left_idx, right_on=right_idx, how=join_type)
        return left_data

    @classmethod
    def pd_merge(cls) -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.merge

    @classmethod
    def pd_concat(cls) -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.concat

    @classmethod
    def pd_merge_asof(cls) -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.merge_asof
