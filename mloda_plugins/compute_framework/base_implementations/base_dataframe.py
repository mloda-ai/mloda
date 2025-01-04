from typing import Any, Set
from mloda_core.abstract_plugins.components.cfw_transformer import ComputeFrameworkTransformMap
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe


class BaseDataFrame(PandasDataframe):
    def transform(
        self,
        data: Any,
        feature_names: Set[str],
        transform_map: ComputeFrameworkTransformMap = ComputeFrameworkTransformMap(),
    ) -> Any:
        transformed_data = self.transform_refactored(data, transform_map)
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
