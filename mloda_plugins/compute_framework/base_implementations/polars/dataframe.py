from typing import Any, Set, Type
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


class PolarsDataFrame(ComputeFramework):
    @staticmethod
    def is_available() -> bool:
        """Check if Polars is installed and available."""
        try:
            import polars

            return True
        except ImportError:
            return False

    @classmethod
    def expected_data_framework(cls) -> Any:
        return cls.pl_dataframe()

    @classmethod
    def merge_engine(cls) -> Type[BaseMergeEngine]:
        return PolarsMergeEngine

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)
        return data.select(list(_selected_feature_names))

    def set_column_names(self) -> None:
        self.column_names = set(self.data.columns)

    @classmethod
    def pl_dataframe(cls) -> Any:
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")
        return pl.DataFrame

    @classmethod
    def pl_series(cls) -> Any:
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
            """Initial data: Transform dict to table"""
            return self.pl_dataframe()(data)

        if isinstance(data, self.pl_series()):
            """Added data: Add column to table"""
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                if feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the dataframe")

                # In Polars, we use with_columns to add new columns
                return self.data.with_columns(data.alias(feature_name))
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def filter_engine(cls) -> Type[BaseFilterEngine]:
        return PolarsFilterEngine
