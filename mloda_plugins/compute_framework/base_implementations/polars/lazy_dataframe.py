from typing import Any, Set, Type
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataframe
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


class PolarsLazyDataframe(PolarsDataframe):
    """
    Lazy evaluation version of PolarsDataframe using pl.LazyFrame.

    This compute framework defers execution of operations until results are explicitly
    requested, enabling query optimization and reduced memory usage for large datasets.
    """

    @staticmethod
    def expected_data_framework() -> Any:
        return PolarsLazyDataframe.pl_lazy_frame()

    def merge_engine(self) -> Type[BaseMergeEngine]:
        return PolarsLazyMergeEngine

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        column_names = set(data.collect_schema().names())
        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)
        # Select the columns and collect the lazy evaluation since this is the final result step
        lazy_result = data.select(list(_selected_feature_names))
        return lazy_result.collect()

    def set_column_names(self) -> None:
        if hasattr(self.data, "collect_schema"):
            # For LazyFrame, use collect_schema() to get column names without executing
            self.column_names = set(self.data.collect_schema().names())
        else:
            raise ValueError("Data does not have a collect_schema method, cannot set column names.")

    @staticmethod
    def pl_lazy_frame() -> Any:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.LazyFrame

    @staticmethod
    def pl_dataframe() -> Any:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.DataFrame

    @staticmethod
    def pl_series() -> Any:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.Series

    def transform(
        self,
        data: Any,
        feature_names: Set[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to lazy frame"""
            return self.pl_lazy_frame()(data)

        if isinstance(data, self.pl_series()):
            """Added data: Add column to lazy frame"""
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                # Check if feature already exists by examining schema
                existing_columns = set(self.data.collect_schema().names())
                if feature_name in existing_columns:
                    raise ValueError(f"Feature {feature_name} already exists in the dataframe")

                # In Polars lazy mode, we use with_columns to add new columns
                return self.data.with_columns(data.alias(feature_name))
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        # Handle DataFrame to LazyFrame conversion
        if isinstance(data, self.pl_dataframe()):
            return data.lazy()

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")
