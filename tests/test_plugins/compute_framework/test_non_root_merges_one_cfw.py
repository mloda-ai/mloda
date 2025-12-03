from typing import Any, Optional, Set

import pytest

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.link import Link, JoinSpec
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI


class NonRootJoinTestFeature(AbstractFeatureGroup):
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


class GroupedNonRootJoinTestFeature(AbstractFeatureGroup):
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


class Call2GroupedNonRootJoinTestFeature(AbstractFeatureGroup):
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
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        # ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestNonRootMerge:
    def test_non_root_merge_simple(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        """
        This test is for testing a merge on the second level of the feature graph.
        """

        feature = Feature(name=Call2GroupedNonRootJoinTestFeature.get_class_name())

        link = Link.inner(
            left=JoinSpec(GroupedNonRootJoinTestFeature, Index(("GroupedNonRootJoinTestFeature",))),
            right=JoinSpec(GroupedSecondNonRootJoinTestFeature, Index(("GroupedSecondNonRootJoinTestFeature",))),
        )

        result = mlodaAPI.run_all(
            [feature],
            links={link},
            compute_frameworks=["PandasDataframe"],
            plugin_collector=PlugInCollector.enabled_feature_groups(
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

    def test_non_root_merge_multiple_join(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
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

        result = mlodaAPI.run_all(
            [feature],
            links=links,
            compute_frameworks=["PandasDataframe"],
            plugin_collector=PlugInCollector.enabled_feature_groups(
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
