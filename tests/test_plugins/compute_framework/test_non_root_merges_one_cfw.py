from typing import Any, Optional, Set

import pytest

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Index
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Link, JoinSpec
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda


class NonRootJoinTestFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"]}


class NonRootJoinTestFeatureB(NonRootJoinTestFeature):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"]}


class SecondNonRootJoinTestFeature(NonRootJoinTestFeature):
    pass


class GroupedNonRootJoinTestFeature(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        if options.get("test_non_root_merge_multiple_join"):
            return {Feature(name="NonRootJoinTestFeature"), Feature(name="NonRootJoinTestFeatureB")}

        return {Feature(name="NonRootJoinTestFeature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.get_options_key("test_non_root_merge_multiple_join"):
            data["GroupedNonRootJoinTestFeature"] = data["NonRootJoinTestFeature"] + data["NonRootJoinTestFeatureB"]
        else:
            data.columns = [cls.get_class_name()]
        return data


class GroupedSecondNonRootJoinTestFeature(GroupedNonRootJoinTestFeature):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature(name="SecondNonRootJoinTestFeature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.get_options_key("test_non_root_merge_multiple_join"):
            data[cls.get_class_name()] = data["SecondNonRootJoinTestFeature"] + data["SecondNonRootJoinTestFeature"]
        else:
            data.columns = [cls.get_class_name()]
        return data


class Call2GroupedNonRootJoinTestFeature(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature(name=GroupedNonRootJoinTestFeature.get_class_name()),
            Feature(name=GroupedSecondNonRootJoinTestFeature.get_class_name()),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if "GroupedNonRootJoinTestFeature" not in data.columns:
            raise ValueError("GroupedNonRootJoinTestFeature not in data")

        if "GroupedSecondNonRootJoinTestFeature" not in data.columns:
            raise ValueError("GroupedSecondNonRootJoinTestFeature not in data")

        data[cls.get_class_name()] = data["GroupedNonRootJoinTestFeature"] + data["GroupedSecondNonRootJoinTestFeature"]
        return data


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationMode.SYNC}),
        ({ParallelizationMode.THREADING}),
        # ({ParallelizationMode.MULTIPROCESSING}),
    ],
)
class TestNonRootMerge:
    def test_non_root_merge_simple(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        """
        This test is for testing a merge on the second level of the feature graph.
        """

        feature = Feature(name=Call2GroupedNonRootJoinTestFeature.get_class_name())

        link = Link.inner(
            left=JoinSpec(GroupedNonRootJoinTestFeature, Index(("GroupedNonRootJoinTestFeature",))),
            right=JoinSpec(GroupedSecondNonRootJoinTestFeature, Index(("GroupedSecondNonRootJoinTestFeature",))),
        )

        result = mloda.run_all(
            [feature],
            links={link},
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    NonRootJoinTestFeature,
                    SecondNonRootJoinTestFeature,
                    GroupedNonRootJoinTestFeature,
                    GroupedSecondNonRootJoinTestFeature,
                    Call2GroupedNonRootJoinTestFeature,
                }
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        for res in result:
            assert res["Call2GroupedNonRootJoinTestFeature"].values == ["Same ValueSame Value"]
            assert len(res.columns) == 1
            assert len(res) == 1

    def test_non_root_merge_multiple_join(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        """
        This test is for testing a merge on the first and second level of the feature graph.
        """

        feature = Feature(
            name=Call2GroupedNonRootJoinTestFeature.get_class_name(),
            options={"test_non_root_merge_multiple_join": True},
        )

        link = Link.inner(
            left=JoinSpec(GroupedNonRootJoinTestFeature, Index(("GroupedNonRootJoinTestFeature",))),
            right=JoinSpec(GroupedSecondNonRootJoinTestFeature, Index(("GroupedSecondNonRootJoinTestFeature",))),
        )

        link_first_level = Link.inner(
            left=JoinSpec(NonRootJoinTestFeature, Index(("NonRootJoinTestFeature",))),
            right=JoinSpec(NonRootJoinTestFeatureB, Index(("NonRootJoinTestFeatureB",))),
        )

        links = set([link, link_first_level])

        result = mloda.run_all(
            [feature],
            links=links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    NonRootJoinTestFeature,
                    SecondNonRootJoinTestFeature,
                    GroupedNonRootJoinTestFeature,
                    GroupedSecondNonRootJoinTestFeature,
                    Call2GroupedNonRootJoinTestFeature,
                    NonRootJoinTestFeatureB,
                }
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        for res in result:
            assert res["Call2GroupedNonRootJoinTestFeature"].values == ["Same ValueSame ValueSame ValueSame Value"]
            assert len(res.columns) == 1
            assert len(res) == 1
