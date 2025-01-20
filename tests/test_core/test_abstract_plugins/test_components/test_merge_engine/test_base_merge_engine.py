from uuid import UUID
import uuid
import pandas as pd
import pandas.testing as pdt
from typing import Any, List, Optional, Set, Tuple
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.components.options import Options, DefaultOptionKeys
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI


class AppendMergeTestFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={f"{cls.get_class_name()}{i}" for i in range(0, 99)})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = features.get_all_names()

        # We check that this feature is alone. It is only for the design of the test.
        if len(feature_name) != 1:
            raise ValueError(f"Test Failed: {feature_name}")
        else:
            single_feature_name = feature_name.pop()
        return {single_feature_name: [f"{single_feature_name}_value"]}


class SecondAppendMergeTestFeature(AppendMergeTestFeature):
    pass


class GroupedAppendMergeTestFeature(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        def add_run_id(some_uuid: UUID, right: int = 0) -> Tuple[str]:
            return (f"{some_uuid}{str(cnt + right)}",)

        run_uuid = uuid.uuid4()

        source_features = options.get(DefaultOptionKeys.mloda_source_feature)
        left_links_cls = options.get(DefaultOptionKeys.left_link_cls)
        right_links_cls = options.get(DefaultOptionKeys.right_link_cls)
        features = set()

        for cnt, value in enumerate(source_features):
            _link = None
            if cnt < len(source_features) - 1:
                _link = Link.append(
                    (left_links_cls, Index(add_run_id(run_uuid))),
                    (right_links_cls, Index(add_run_id(run_uuid, 1))),
                )

            features.add(
                Feature(
                    name=value,
                    link=_link,
                    index=Index(add_run_id(run_uuid)),
                    options={"Arbritrary options": add_run_id(run_uuid)},
                )
            )
        return features

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data.apply(lambda row: "".join(filter(lambda x: pd.notna(x), row)), axis=1)
        data = data[[cls.get_class_name()]]
        return data


class Call2GroupedAppendMergeTestFeature(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        iteration = options.get("iteration")

        features = frozenset({f"AppendMergeTestFeature{i}" for i in range(iteration)})
        feature = Feature(
            name=GroupedAppendMergeTestFeature.get_class_name(),
            index=Index((GroupedAppendMergeTestFeature.get_class_name(),)),
            options={
                DefaultOptionKeys.mloda_source_feature: features,
                DefaultOptionKeys.left_link_cls: AppendMergeTestFeature,
                DefaultOptionKeys.right_link_cls: AppendMergeTestFeature,
            },
        )

        features2 = frozenset({f"SecondAppendMergeTestFeature{i}" for i in range(iteration)})
        feature2 = Feature(
            name=GroupedAppendMergeTestFeature.get_class_name(),
            index=Index((f"{GroupedAppendMergeTestFeature.get_class_name()}2",)),
            options={
                DefaultOptionKeys.mloda_source_feature: features2,
                DefaultOptionKeys.left_link_cls: SecondAppendMergeTestFeature,
                DefaultOptionKeys.right_link_cls: SecondAppendMergeTestFeature,
            },
        )

        return {feature, feature2}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data.apply(lambda row: "".join(filter(lambda x: pd.notna(x), row)), axis=1)
        return data


class TestBaseMergeEngine:
    def test_prep_append(self) -> None:
        feature_list: List[str | Feature] = ["AppendMergeTestFeature1"]

        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks=["PandasDataframe"],
            plugin_collector=PlugInCollector.enabled_feature_groups({AppendMergeTestFeature}),
        )
        assert len(result) == 1

    def test_dependent_append_multiple_features_v1(self) -> None:
        iteration = 5
        features = frozenset({f"AppendMergeTestFeature{i}" for i in range(iteration)})

        feature = Feature(
            name=GroupedAppendMergeTestFeature.get_class_name(),
            options={
                DefaultOptionKeys.mloda_source_feature.value: features,
                DefaultOptionKeys.left_link_cls: AppendMergeTestFeature,
                DefaultOptionKeys.right_link_cls: AppendMergeTestFeature,
            },
        )

        result = mlodaAPI.run_all(
            [feature],
            compute_frameworks=["PandasDataframe"],
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {GroupedAppendMergeTestFeature, AppendMergeTestFeature}
            ),
        )
        assert len(result) == 1

        data = {
            GroupedAppendMergeTestFeature.get_class_name(): [
                f"{AppendMergeTestFeature.get_class_name()}{i}_value" for i in range(iteration)
            ]
        }
        expected_df = pd.DataFrame(data)
        expected_df = expected_df.sort_values("GroupedAppendMergeTestFeature").set_index(
            "GroupedAppendMergeTestFeature"
        )

        for res in result:
            res = res.sort_values("GroupedAppendMergeTestFeature").set_index("GroupedAppendMergeTestFeature")
            pdt.assert_frame_equal(res, expected_df)

    def test_dependent_append_multiple_features_v2(self) -> None:
        iteration = 5

        feature_name = GroupedAppendMergeTestFeature.get_class_name()

        features = frozenset({f"AppendMergeTestFeature{i}" for i in range(iteration)})
        feature = Feature(
            name=feature_name,
            options={
                DefaultOptionKeys.mloda_source_feature: features,
                DefaultOptionKeys.left_link_cls: AppendMergeTestFeature,
                DefaultOptionKeys.right_link_cls: AppendMergeTestFeature,
            },
        )

        features2 = frozenset({f"SecondAppendMergeTestFeature{i}" for i in range(iteration)})
        feature2 = Feature(
            name=feature_name,
            options={
                DefaultOptionKeys.mloda_source_feature: features2,
                DefaultOptionKeys.left_link_cls: SecondAppendMergeTestFeature,
                DefaultOptionKeys.right_link_cls: SecondAppendMergeTestFeature,
            },
        )

        result = mlodaAPI.run_all(
            [feature, feature2],
            compute_frameworks=["PandasDataframe"],
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {GroupedAppendMergeTestFeature, AppendMergeTestFeature, SecondAppendMergeTestFeature}
            ),
        )

        for res in result:
            assert len(res) == 5

    def test_dependent_append_multiple_features_v3(self) -> None:
        feature = Feature(
            name=Call2GroupedAppendMergeTestFeature.get_class_name(),
            options={
                "iteration": 3,
            },
        )

        link = Link.union(
            (GroupedAppendMergeTestFeature, Index(("GroupedAppendMergeTestFeature",))),
            (GroupedAppendMergeTestFeature, Index(("GroupedAppendMergeTestFeature2",))),
        )

        result = mlodaAPI.run_all(
            [feature],
            links={link},
            compute_frameworks=["PandasDataframe"],
            plugin_collector=PlugInCollector.enabled_feature_groups(
                {
                    GroupedAppendMergeTestFeature,
                    AppendMergeTestFeature,
                    SecondAppendMergeTestFeature,
                    Call2GroupedAppendMergeTestFeature,
                }
            ),
        )
        for res in result:
            assert len(res) == 6
            assert res.columns == ["Call2GroupedAppendMergeTestFeature"]
