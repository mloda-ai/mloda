from typing import Any, Optional, Set, Type, Union

from mloda import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
import pytest
import pyarrow as pa

from mloda import FeatureGroup
from mloda import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Index
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.user import Link, JoinSpec
from mloda import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
import mloda


class NonCfwRootJoinTestFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], "dummy": ["dummy"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class NonCfwRootJoinTestFeatureB(NonCfwRootJoinTestFeature):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], "dummy2": ["dummy3"]}


class SecondNonCfwRootJoinTestFeature(NonCfwRootJoinTestFeature):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], "dummy4": ["dummy3"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class GroupedNonCfwRootJoinTestFeature(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        if options.get("test_non_root_merge_multiple_join"):
            return {Feature(name="NonCfwRootJoinTestFeature"), Feature(name="NonCfwRootJoinTestFeatureB")}

        return {Feature(name="NonCfwRootJoinTestFeature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.get_options_key("test_non_root_merge_multiple_join"):
            col_a = data.column("NonCfwRootJoinTestFeature").to_pandas()
            data = data.append_column("GroupedNonCfwRootJoinTestFeature", pa.array(col_a))
        else:
            data = data.rename_columns([cls.get_class_name(), "dummy"])
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


class GroupedSecondNonCfwRootJoinTestFeature(GroupedNonCfwRootJoinTestFeature):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature(name="SecondNonCfwRootJoinTestFeature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = "Same Value"
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class Call2GroupedNonCfwRootJoinTestFeature(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {
            Feature(name=GroupedNonCfwRootJoinTestFeature.get_class_name()),
            Feature(name=GroupedSecondNonCfwRootJoinTestFeature.get_class_name()),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if "GroupedNonCfwRootJoinTestFeature" not in data.column_names:
            raise ValueError("GroupedNonCfwRootJoinTestFeature not in data")

        if "GroupedSecondNonCfwRootJoinTestFeature" not in data.column_names:
            raise ValueError("GroupedSecondNonCfwRootJoinTestFeature not in data")

        data = data.append_column(cls.get_class_name(), data["GroupedNonCfwRootJoinTestFeature"])
        return data


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationMode.SYNC}),
        ({ParallelizationMode.THREADING}),
        # ({ParallelizationMode.MULTIPROCESSING}),
    ],
)
class TestNonCfWRootMerge:
    def test_non_cfw_root_merge_simple(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        """
        This test is for testing a merge on the second level of the feature graph with mixed cfw.
        """

        feature = Feature(name=Call2GroupedNonCfwRootJoinTestFeature.get_class_name())

        link = Link.inner(
            left=JoinSpec(GroupedNonCfwRootJoinTestFeature, Index(("GroupedNonCfwRootJoinTestFeature",))),
            right=JoinSpec(GroupedSecondNonCfwRootJoinTestFeature, Index(("GroupedSecondNonCfwRootJoinTestFeature",))),
        )

        result = mloda.run_all(
            [feature],
            links={link},
            compute_frameworks=["PyArrowTable", "PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    NonCfwRootJoinTestFeature,
                    SecondNonCfwRootJoinTestFeature,
                    GroupedNonCfwRootJoinTestFeature,
                    GroupedSecondNonCfwRootJoinTestFeature,
                    Call2GroupedNonCfwRootJoinTestFeature,
                }
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        for res in result:
            res = res.to_pydict()
            assert res["Call2GroupedNonCfwRootJoinTestFeature"] == ["Same Value"]
            assert len(res) == 1

    def test_non_cfw_root_multiple(self, modes: Set[ParallelizationMode], flight_server: Any) -> None:
        feature = Feature(
            name=Call2GroupedNonCfwRootJoinTestFeature.get_class_name(),
            options={"test_non_root_merge_multiple_join": True},
        )

        link = Link.inner(
            left=JoinSpec(GroupedNonCfwRootJoinTestFeature, Index(("GroupedNonCfwRootJoinTestFeature",))),
            right=JoinSpec(GroupedSecondNonCfwRootJoinTestFeature, Index(("GroupedSecondNonCfwRootJoinTestFeature",))),
        )

        link_first_level = Link.inner(
            left=JoinSpec(NonCfwRootJoinTestFeature, Index(("NonCfwRootJoinTestFeature",))),
            right=JoinSpec(NonCfwRootJoinTestFeatureB, Index(("NonCfwRootJoinTestFeatureB",))),
        )

        links = set([link, link_first_level])

        result = mloda.run_all(
            [feature],
            links=links,
            compute_frameworks=["PyArrowTable", "PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    NonCfwRootJoinTestFeature,
                    NonCfwRootJoinTestFeatureB,
                    SecondNonCfwRootJoinTestFeature,
                    GroupedNonCfwRootJoinTestFeature,
                    GroupedSecondNonCfwRootJoinTestFeature,
                    Call2GroupedNonCfwRootJoinTestFeature,
                }
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )

        for res in result:
            assert res.to_pydict() == {"Call2GroupedNonCfwRootJoinTestFeature": ["Same Value"]}
