from typing import Any

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


class PolarsMergeEngine(BaseMergeEngine):
    def check_import(self) -> None:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")

    def merge_inner(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("inner", left_data, right_data, left_index, right_index, JoinType.INNER)

    def merge_left(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("left", left_data, right_data, left_index, right_index, JoinType.LEFT)

    def merge_right(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("right", left_data, right_data, left_index, right_index, JoinType.RIGHT)

    def merge_full_outer(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.join_logic("full", left_data, right_data, left_index, right_index, JoinType.OUTER)

    def merge_append(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        return self.pl_concat()([left_data, right_data], how="diagonal")

    def merge_union(self, left_data: Any, right_data: Any, left_index: Index, right_index: Index) -> Any:
        combined = self.merge_append(left_data, right_data, left_index, right_index)
        return combined.unique()

    def get_column_names(self, data: Any) -> list[str]:
        """Get column names from data. Override in subclasses for different data types."""
        return list(data.columns)

    def is_empty_data(self, data: Any) -> bool:
        """Check if data is empty. Override in subclasses for different data types."""
        return len(data) == 0

    def column_exists_in_result(self, result: Any, column_name: str) -> bool:
        """Check if column exists in result. Override in subclasses for different data types."""
        return column_name in result.columns

    def handle_empty_data(self, left_data: Any, right_data: Any, left_idx: str, right_idx: str) -> Any:
        """Handle empty data cases. Override in subclasses for different data types."""
        if self.is_empty_data(left_data) or self.is_empty_data(right_data):
            # For empty datasets, create compatible schemas
            if self.is_empty_data(left_data) and self.is_empty_data(right_data):
                # Both empty - return empty with combined schema
                combined_schema = {}
                for col in self.get_column_names(left_data):
                    combined_schema[col] = left_data[col].dtype
                for col in self.get_column_names(right_data):
                    if col not in combined_schema:
                        combined_schema[col] = right_data[col].dtype
                return pl.DataFrame(schema=combined_schema)
            elif self.is_empty_data(left_data):
                # Left empty - ensure left has compatible schema with right join column
                left_schema = dict(left_data.schema)
                if left_idx in self.get_column_names(right_data):
                    left_schema[left_idx] = right_data[right_idx].dtype
                return pl.DataFrame(schema=left_schema)
            else:
                # Right empty - ensure right has compatible schema with left join column
                right_schema = dict(right_data.schema)
                if right_idx in self.get_column_names(left_data):
                    right_schema[right_idx] = left_data[left_idx].dtype
                return pl.DataFrame(schema=right_schema)
        return None

    def join_logic(
        self, join_type: str, left_data: Any, right_data: Any, left_index: Index, right_index: Index, jointype: JoinType
    ) -> Any:
        if left_index.is_multi_index() or right_index.is_multi_index():
            raise ValueError(f"MultiIndex is not yet implemented {self.__class__.__name__}")

        left_idx = left_index.index[0]
        right_idx = right_index.index[0]

        # Handle empty data cases
        empty_result = self.handle_empty_data(left_data, right_data, left_idx, right_idx)
        if empty_result is not None:
            return empty_result

        # Perform the join with nulls_equal=True to match null values (updated parameter name)
        try:
            result = left_data.join(right_data, left_on=left_idx, right_on=right_idx, how=join_type, nulls_equal=True)
        except TypeError:
            # Fallback for older polars versions
            result = left_data.join(right_data, left_on=left_idx, right_on=right_idx, how=join_type, join_nulls=True)

        # For different join column names, add the right join column manually
        # because Polars drops it when column names are different
        if left_idx != right_idx:
            # Add the right join column by copying the left join column values
            # This works because the join ensures they have matching values
            result = result.with_columns(pl.col(left_idx).alias(right_idx))

        # Handle duplicate join columns only for full outer joins when column names are the same
        right_col_name = f"{right_idx}_right"
        if self.column_exists_in_result(result, right_col_name) and join_type == "full" and left_idx == right_idx:
            # For full outer joins with same column names, coalesce the columns
            # Use the right column value when left is null, otherwise use left
            result = result.with_columns(
                pl.when(pl.col(left_idx).is_null())
                .then(pl.col(right_col_name))
                .otherwise(pl.col(left_idx))
                .alias(left_idx)
            ).drop(right_col_name)

        # Ensure consistent column ordering: join column first, then left columns, then right columns
        left_cols = [col for col in self.get_column_names(left_data) if col != left_idx]
        right_cols = [
            col
            for col in self.get_column_names(right_data)
            if col != right_idx and col not in self.get_column_names(left_data)
        ]

        # For different join column names, include the right join column in the ordering
        if left_idx != right_idx:
            right_cols = [right_idx] + right_cols

        # Build the desired column order
        desired_order = [left_idx] + left_cols + right_cols

        # Select columns in the desired order (only if they exist in result)
        result_columns = self.get_column_names(result)
        existing_cols = [col for col in desired_order if col in result_columns]
        result = result.select(existing_cols)

        return result

    @staticmethod
    def pl_concat() -> Any:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.concat
