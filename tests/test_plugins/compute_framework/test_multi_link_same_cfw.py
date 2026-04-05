from typing import Any, Optional

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


# === Scenario 1: Same class used 3x with different feature names per batch ===


class ReadFGS1(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["v1"]}


class AggFGS1(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"agg_sum_s1", "agg_count_s1", "agg_avg_s1"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        name = next(iter(features.get_all_names()))
        return {name: ["v1"]}


class ConsumerFGS1(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name=ReadFGS1.get_class_name()),
            Feature(name="agg_sum_s1", options={"agg_type": "sum"}),
            Feature(name="agg_count_s1", options={"agg_type": "count"}),
            Feature(name="agg_avg_s1", options={"agg_type": "avg"}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["ok"]}


# === Scenario 2: Subclasses with base-class Links ===


class BaseAggS2(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["v1"]}


class SumAggS2(BaseAggS2):
    pass


class CountAggS2(BaseAggS2):
    pass


class AvgAggS2(BaseAggS2):
    pass


class ReadFGS2(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["v1"]}


class ConsumerFGS2(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name=ReadFGS2.get_class_name()),
            Feature(name=SumAggS2.get_class_name()),
            Feature(name=CountAggS2.get_class_name()),
            Feature(name=AvgAggS2.get_class_name()),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["ok"]}


# === Scenario 3: Fully distinct classes (verify still works with same cfw) ===


class ReadFGS3(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["v1"]}


class SumFGS3(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["v1"]}


class CountFGS3(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["v1"]}


class AvgFGS3(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["v1"]}


class ConsumerFGS3(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name=ReadFGS3.get_class_name()),
            Feature(name=SumFGS3.get_class_name()),
            Feature(name=CountFGS3.get_class_name()),
            Feature(name=AvgFGS3.get_class_name()),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): ["ok"]}


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationMode.SYNC}),
        ({ParallelizationMode.THREADING}),
    ],
)
class TestMultiLinkSameCfw:
    def test_multi_link_same_class_same_cfw(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """Scenario 1: same FG class used 3x (different feature names per batch), same compute framework."""
        feature = Feature(name=ConsumerFGS1.get_class_name())
        links = {
            Link.inner(
                JoinSpec(ReadFGS1, Index(("ReadFGS1",))),
                JoinSpec(AggFGS1, Index(("agg_sum_s1",))),
            ),
            Link.inner(
                JoinSpec(ReadFGS1, Index(("ReadFGS1",))),
                JoinSpec(AggFGS1, Index(("agg_count_s1",))),
            ),
            Link.inner(
                JoinSpec(ReadFGS1, Index(("ReadFGS1",))),
                JoinSpec(AggFGS1, Index(("agg_avg_s1",))),
            ),
        }
        result = mloda.run_all(
            [feature],
            links=links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups({ReadFGS1, AggFGS1, ConsumerFGS1}),
            flight_server=flight_server,
            parallelization_modes=modes,
        )
        for res in result:
            assert len(res) == 1
            assert ConsumerFGS1.get_class_name() in res.columns

    def test_multi_link_subclass_same_cfw(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """Scenario 2: subclasses of a common base, Links defined on the base class, same compute framework."""
        feature = Feature(name=ConsumerFGS2.get_class_name())
        links = {
            Link.inner(
                JoinSpec(ReadFGS2, Index(("ReadFGS2",))),
                JoinSpec(BaseAggS2, Index(("SumAggS2",))),
            ),
            Link.inner(
                JoinSpec(ReadFGS2, Index(("ReadFGS2",))),
                JoinSpec(BaseAggS2, Index(("CountAggS2",))),
            ),
            Link.inner(
                JoinSpec(ReadFGS2, Index(("ReadFGS2",))),
                JoinSpec(BaseAggS2, Index(("AvgAggS2",))),
            ),
        }
        result = mloda.run_all(
            [feature],
            links=links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {ReadFGS2, SumAggS2, CountAggS2, AvgAggS2, ConsumerFGS2}
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )
        for res in result:
            assert len(res) == 1
            assert ConsumerFGS2.get_class_name() in res.columns

    def test_multi_link_distinct_class_same_cfw(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        """Scenario 3: fully distinct FG classes, same compute framework (verify no regression)."""
        feature = Feature(name=ConsumerFGS3.get_class_name())
        links = {
            Link.inner(
                JoinSpec(ReadFGS3, Index(("ReadFGS3",))),
                JoinSpec(SumFGS3, Index(("SumFGS3",))),
            ),
            Link.inner(
                JoinSpec(ReadFGS3, Index(("ReadFGS3",))),
                JoinSpec(CountFGS3, Index(("CountFGS3",))),
            ),
            Link.inner(
                JoinSpec(ReadFGS3, Index(("ReadFGS3",))),
                JoinSpec(AvgFGS3, Index(("AvgFGS3",))),
            ),
        }
        result = mloda.run_all(
            [feature],
            links=links,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {ReadFGS3, SumFGS3, CountFGS3, AvgFGS3, ConsumerFGS3}
            ),
            flight_server=flight_server,
            parallelization_modes=modes,
        )
        for res in result:
            assert len(res) == 1
            assert ConsumerFGS3.get_class_name() in res.columns
