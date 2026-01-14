from typing import Any, Optional, Set
import os


from mloda.user import FeatureName

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.user import Options


from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


try:
    import pandas as pd
except ImportError:
    pd = None


class MixedCfwFeature(FeatureGroup):
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_set = set()

        py_arrow_features = ["id", "V1", "V2"]
        for py_f in py_arrow_features:
            feature_set.add(
                Feature(
                    name=py_f,
                    options={CsvReader.__name__: self.file_path, "123": 2},
                    compute_framework="PyArrowTable",
                )
            )

        pd_features = ["id", "V3", "V4"]
        for py_f in pd_features:
            feature_set.add(
                Feature(
                    name=py_f,
                    options={CsvReader.__name__: self.file_path, "123": 2},
                    compute_framework="PandasDataFrame",
                )
            )
        return feature_set

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data is not a pandas dataframe. Got {type(data)}")

        for col in ["id", "V1", "V2", "V3", "V4"]:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data.")

        data[cls.get_class_name()] = data["V1"] + data["V2"] + data["V3"] + data["V4"]
        return data


class DuplicateFeatureSetup(MixedCfwFeature):
    file_path = f"{os.getcwd()}/tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_set = set()

        py_arrow_features = ["id", "V1", "V2"]
        for py_f in py_arrow_features:
            feature_set.add(
                Feature(
                    name=py_f,
                    options={CsvReader.__name__: self.file_path, "self_left_alias": "dummy"},
                    compute_framework="PandasDataFrame",
                )
            )

        pd_features = ["id", "V3", "V4"]
        for py_f in pd_features:
            feature_set.add(
                Feature(
                    name=py_f,
                    options={CsvReader.__name__: self.file_path, "self_right_alias": "dummy"},
                    compute_framework="PandasDataFrame",
                )
            )
        return feature_set
