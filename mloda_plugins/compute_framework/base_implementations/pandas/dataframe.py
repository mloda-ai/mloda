from typing import Any, Set, Type
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine

try:
    import pandas as pd
except ImportError:
    pd = None


class PandasDataframe(ComputeFrameWork):
    @staticmethod
    def is_available() -> bool:
        """Check if Pandas is installed and available."""
        try:
            import pandas

            return True
        except ImportError:
            return False

    @staticmethod
    def expected_data_framework() -> Any:
        return PandasDataframe.pd_dataframe()

    def merge_engine(self) -> Type[BaseMergeEngine]:
        return PandasMergeEngine

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)
        return data[[f for f in _selected_feature_names]]

    def set_column_names(self) -> None:
        self.column_names = set(self.data.columns)

    @staticmethod
    def pd_dataframe() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.DataFrame

    @staticmethod
    def pd_series() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.Series

    @staticmethod
    def pd_merge() -> Any:
        if pd is None:
            raise ImportError("Pandas is not installed. To be able to use this framework, please install pandas.")
        return pd.merge

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
            return self.pd_dataframe().from_dict(data)

        if isinstance(data, self.pd_series()):
            """Added data: Add column to table"""
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                if feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the dataframe")

                self.data[feature_name] = data
                return self.data
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    def filter_engine(self) -> Type[BaseFilterEngine]:
        return PandasFilterEngine
