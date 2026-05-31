from uuid import UUID
import uuid
import pytest
import pandas as pd
import pandas.testing as pdt
from typing import Any, Optional
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Index
from mloda.provider import BaseInputData
from mloda.provider import BaseMergeEngine
from mloda.provider import DataCreator
from mloda.user import Link, JoinSpec
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda.provider import DefaultOptionKeys
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link


class AppendMergeTestFeature(FeatureGroup):
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


class GroupedAppendMergeTestFeature(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        def add_run_id(some_uuid: UUID, right: int = 0) -> tuple[str]:
            return (f"{some_uuid}{str(cnt + right)}",)

        run_uuid = uuid.uuid4()

        source_features = options.get(DefaultOptionKeys.in_features)
        left_links_cls = options.get("left_link_cls")
        right_links_cls = options.get("right_link_cls")
        features = set()

        for cnt, value in enumerate(source_features):
            _link = None
            if cnt < len(source_features) - 1:
                _link = Link.append(
                    JoinSpec(left_links_cls, Index(add_run_id(run_uuid))),
                    JoinSpec(right_links_cls, Index(add_run_id(run_uuid, 1))),
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


class Call2GroupedAppendMergeTestFeature(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        iteration = options.get("iteration")

        features = frozenset({f"AppendMergeTestFeature{i}" for i in range(iteration)})
        feature = Feature(
            name=GroupedAppendMergeTestFeature.get_class_name(),
            index=Index((GroupedAppendMergeTestFeature.get_class_name(),)),
            options={
                DefaultOptionKeys.in_features: features,
                "left_link_cls": AppendMergeTestFeature,
                "right_link_cls": AppendMergeTestFeature,
            },
        )

        features2 = frozenset({f"SecondAppendMergeTestFeature{i}" for i in range(iteration)})
        feature2 = Feature(
            name=GroupedAppendMergeTestFeature.get_class_name(),
            index=Index((f"{GroupedAppendMergeTestFeature.get_class_name()}2",)),
            options={
                DefaultOptionKeys.in_features: features2,
                "left_link_cls": SecondAppendMergeTestFeature,
                "right_link_cls": SecondAppendMergeTestFeature,
            },
        )

        return {feature, feature2}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[cls.get_class_name()] = data.apply(lambda row: "".join(filter(lambda x: pd.notna(x), row)), axis=1)
        return data


class TestBaseMergeEngine:
    def test_prep_append(self) -> None:
        feature_list: list[str | Feature] = ["AppendMergeTestFeature1"]

        result = mloda.run_all(
            feature_list,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups({AppendMergeTestFeature}),
        )
        assert len(result) == 1

    def test_dependent_append_multiple_features_v1(self) -> None:
        iteration = 5
        features = frozenset({f"AppendMergeTestFeature{i}" for i in range(iteration)})

        feature = Feature(
            name=GroupedAppendMergeTestFeature.get_class_name(),
            options={
                DefaultOptionKeys.in_features: features,
                "left_link_cls": AppendMergeTestFeature,
                "right_link_cls": AppendMergeTestFeature,
            },
        )

        result = mloda.run_all(
            [feature],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
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
                DefaultOptionKeys.in_features: features,
                "left_link_cls": AppendMergeTestFeature,
                "right_link_cls": AppendMergeTestFeature,
            },
        )

        features2 = frozenset({f"SecondAppendMergeTestFeature{i}" for i in range(iteration)})
        feature2 = Feature(
            name=feature_name,
            options={
                DefaultOptionKeys.in_features: features2,
                "left_link_cls": SecondAppendMergeTestFeature,
                "right_link_cls": SecondAppendMergeTestFeature,
            },
        )

        result = mloda.run_all(
            [feature, feature2],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {GroupedAppendMergeTestFeature, AppendMergeTestFeature, SecondAppendMergeTestFeature}
            ),
        )

        for res in result:
            assert len(res) == 5

    def test_merge_asof_default_raises_value_error(self) -> None:
        """A bare BaseMergeEngine subclass must raise ValueError mentioning 'asof' for merge_asof."""
        from mloda.user import JoinType
        from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

        class BareMergeEngine(BaseMergeEngine):
            pass

        engine = BareMergeEngine()
        idx = Index(("k",))
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t")

        with pytest.raises(ValueError, match="asof"):
            engine.merge_asof(None, None, idx, idx, cfg)

        # Sanity: JoinType.ASOF must exist for the dispatch tests below.
        assert JoinType.ASOF.value == "asof"

    def test_merge_asof_link_without_config_raises_value_error(self) -> None:
        """An ASOF-jointype Link whose asof_config is None must raise ValueError mentioning 'asof_config'."""
        from mloda.user import JoinType

        class BareMergeEngine(BaseMergeEngine):
            pass

        engine = BareMergeEngine()
        idx = Index(("k",))
        link = make_merge_link(JoinType.ASOF, idx, idx)

        with pytest.raises(ValueError, match="asof_config"):
            engine.merge(None, None, link)

    def test_merge_non_asof_dispatches_via_link(self) -> None:
        """A non-ASOF Link dispatches to the matching leaf hook (INNER -> merge_inner)."""
        from mloda.user import JoinType
        from mloda.core.abstract_plugins.components.index.index import Index as CoreIndex

        sentinel = object()

        class NonAsofMergeEngine(BaseMergeEngine):
            def merge_inner(
                self, left_data: Any, right_data: Any, left_index: CoreIndex, right_index: CoreIndex
            ) -> Any:
                return sentinel

        engine = NonAsofMergeEngine()
        idx = CoreIndex(("id",))

        result = engine.merge(None, None, make_merge_link(JoinType.INNER, idx, idx))
        assert result is sentinel

    def test_merge_asof_dispatches_to_merge_asof(self) -> None:
        """merge() with a real ASOF Link (carrying asof_config) dispatches to merge_asof."""
        sentinel = object()

        class DispatchMergeEngine(BaseMergeEngine):
            def merge_asof(
                self, left_data: Any, right_data: Any, left_index: Index, right_index: Index, asof_config: Any
            ) -> Any:
                return sentinel

        engine = DispatchMergeEngine()
        link = Link.asof(
            JoinSpec(AppendMergeTestFeature, "k"),
            JoinSpec(SecondAppendMergeTestFeature, "k"),
            left_time_column="t",
            right_time_column="t",
        )

        result = engine.merge(None, None, link)
        assert result is sentinel

    def test_dependent_append_multiple_features_v3(self) -> None:
        feature = Feature(
            name=Call2GroupedAppendMergeTestFeature.get_class_name(),
            options={
                "iteration": 3,
            },
        )

        link = Link.union(
            JoinSpec(GroupedAppendMergeTestFeature, Index(("GroupedAppendMergeTestFeature",))),
            JoinSpec(GroupedAppendMergeTestFeature, Index(("GroupedAppendMergeTestFeature2",))),
        )

        result = mloda.run_all(
            [feature],
            links={link},
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(
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
