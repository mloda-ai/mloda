"""A same-compute-framework, non-APPEND/UNION join whose child requires both raw sides directly
resolves through ExecutionPlan.add_tfs's inner_ep.tfs_ids branch and produces correct joined data.
"""

from typing import Any

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import Link
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


LEFT_KEYS = [1, 2, 3]
LEFT_VALUES = [10, 20, 30]
RIGHT_KEYS = [1, 2, 3]
RIGHT_VALUES = [100, 200, 300]
SUMMED_VALUES = [110, 220, 330]


class SftfsLeft(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"sftfs_left_key", "sftfs_left_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sftfs_left_key": LEFT_KEYS, "sftfs_left_value": LEFT_VALUES}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("sftfs_left_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SftfsRight(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"sftfs_right_key", "sftfs_right_value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"sftfs_right_key": RIGHT_KEYS, "sftfs_right_value": RIGHT_VALUES}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("sftfs_right_key",))]

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SftfsChild(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name="sftfs_left_key"),
            Feature(name="sftfs_left_value"),
            Feature(name="sftfs_right_key"),
            Feature(name="sftfs_right_value"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        summed = (data["sftfs_left_value"] + data["sftfs_right_value"]).tolist()
        return {cls.get_class_name(): summed}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


def test_same_framework_inner_join_child_requiring_both_sides_computes_correct_sums() -> None:
    link = Link.inner_on(SftfsLeft, SftfsRight)

    results = mloda.run_all(
        [Feature(name=SftfsChild.get_class_name())],
        links={link},
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.enabled_feature_groups({SftfsLeft, SftfsRight, SftfsChild}),
    )

    matching = [frame for frame in results if SftfsChild.get_class_name() in frame.columns]
    assert len(matching) == 1

    values = sorted(matching[0][SftfsChild.get_class_name()].tolist())
    assert values == sorted(SUMMED_VALUES)
