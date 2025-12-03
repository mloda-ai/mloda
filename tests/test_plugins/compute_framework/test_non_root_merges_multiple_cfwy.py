from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
import pytest
import pyarrow as pa

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_core.api.request import mlodaAPI


class NonCfwRootJoinTestFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["Same Value"], "dummy": ["dummy"]}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
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
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataFrame}


class GroupedNonCfwRootJoinTestFeature(AbstractFeatureGroup):
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
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyArrowTable}


class GroupedSecondNonCfwRootJoinTestFeature(GroupedNonCfwRootJoinTestFeature):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature(name="SecondNonCfwRootJoinTestFeature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = "Same Value"
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataFrame}


class Call2GroupedNonCfwRootJoinTestFeature(AbstractFeatureGroup):
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
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        # ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestNonCfWRootMerge:
    def test_non_cfw_root_merge_simple(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        """
        This test is for testing a merge on the second level of the feature graph with mixed cfw.
        """

        feature = Feature(name=Call2GroupedNonCfwRootJoinTestFeature.get_class_name())

        link = Link.inner(
            left=(GroupedNonCfwRootJoinTestFeature, Index(("GroupedNonCfwRootJoinTestFeature",))),
            right=(GroupedSecondNonCfwRootJoinTestFeature, Index(("GroupedSecondNonCfwRootJoinTestFeature",))),
        )

        result = mlodaAPI.run_all(
            [feature],
            links={link},
            compute_frameworks=["PyArrowTable", "PandasDataFrame"],
            plugin_collector=PlugInCollector.enabled_feature_groups(
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

    def test_non_cfw_root_multiple(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        feature = Feature(
            name=Call2GroupedNonCfwRootJoinTestFeature.get_class_name(),
            options={"test_non_root_merge_multiple_join": True},
        )

        link = Link.inner(
            left=(GroupedNonCfwRootJoinTestFeature, Index(("GroupedNonCfwRootJoinTestFeature",))),
            right=(GroupedSecondNonCfwRootJoinTestFeature, Index(("GroupedSecondNonCfwRootJoinTestFeature",))),
        )

        link_first_level = Link.inner(
            left=(NonCfwRootJoinTestFeature, Index(("NonCfwRootJoinTestFeature",))),
            right=(NonCfwRootJoinTestFeatureB, Index(("NonCfwRootJoinTestFeatureB",))),
        )

        links = set([link, link_first_level])

        result = mlodaAPI.run_all(
            [feature],
            links=links,
            compute_frameworks=["PyArrowTable", "PandasDataFrame"],
            plugin_collector=PlugInCollector.enabled_feature_groups(
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
