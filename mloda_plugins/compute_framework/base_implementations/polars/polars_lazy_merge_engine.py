from typing import Any

from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


class PolarsLazyMergeEngine(PolarsMergeEngine):
    """
    Lazy merge engine for Polars LazyFrame operations.

    Inherits from PolarsMergeEngine and overrides helper methods for
    LazyFrame-specific handling of schema operations.
    """

    def get_column_names(self, data: Any) -> list[str]:
        """Get column names from LazyFrame using collect_schema()."""
        return list(data.collect_schema().names())

    def is_empty_data(self, data: Any) -> bool:
        """For LazyFrames, we can't easily check if empty without collecting.
        Skip empty data handling and let Polars handle it efficiently."""
        return False

    def column_exists_in_result(self, result: Any, column_name: str) -> bool:
        """Check if column exists in LazyFrame result using collect_schema()."""
        return column_name in result.collect_schema().names()

    def handle_empty_data(self, left_data: Any, right_data: Any, left_idx: str, right_idx: str) -> Any:
        """For LazyFrames, skip empty data handling and let Polars handle it efficiently."""
        return None
