"""Picklable joined-plan fixtures without importing test modules in spawned workers."""

from typing import Any

from mloda.provider import ApiDataFG, ComputeFramework, DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Index, JoinSpec, Link, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class PrepareRunDictCreator(DataCreatorFeatureGroup):
    """Dict-framework root keyed by prj_creator_id."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"prj_creator_id", "prj_creator_value"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"prj_creator_id": list(range(8)), "prj_creator_value": [f"c{i}" for i in range(8)]}


class PrepareRunJoinedFeature(FeatureGroup):
    """Pandas group left-joining api data with the dict creator, so the plan transforms and joins."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        link = Link.left(
            JoinSpec(ApiDataFG, Index(("api_id",))), JoinSpec(PrepareRunDictCreator, Index(("prj_creator_id",)))
        )
        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="prj_creator_value", index=Index(("prj_creator_id",))),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["PrepareRunJoinedFeature"] = data["api_value"].fillna("") + "_" + data["prj_creator_value"].fillna("")
        return data

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}
