"""E2E integration tests for propagate_context_keys in chained features."""

from typing import Any, ClassVar

import pandas as pd
import pytest

from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda
from mloda.provider import (
    ComputeFramework,
    DataCreatorFeatureGroup,
    DefaultOptionKeys,
    FeatureGroup,
    FeatureSet,
    PropertySpec,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.integration_plugins.chainer.propagate_context_feature import (
    CASCADE_ENV_ROOT,
    CASCADE_FOLD_CASCADE_CONSUMER,
    CASCADE_FOLD_PINNED_CONSUMER,
    CASCADE_GUARD_CASCADE_CONSUMER,
    CASCADE_GUARD_HOP_CONSUMER,
    CASCADE_GUARD_PULL_CONSUMER,
    CASCADE_GUARD_ROOT,
    CASCADE_GUARD_SHARED,
    CascadeEnvRootGroupTest,
    CascadeFoldConsumerGroupTest,
    CascadeFoldSharedGroupTest,
    CascadeGuardConsumerGroupTest,
    CascadeGuardRootGroupTest,
    CascadeGuardSharedGroupTest,
    NameChainedCascadeGroupTest,
    PropagateContextFeatureGroupTest,
)
from tests.test_plugins.integration_plugins.chainer.context.test_chained_optional_features import (
    ChainerContextParserTestDataCreator,
)


NAME_CHAINED_TOP = "Sales__scale_ncc9x__scale_ncc9x__scale_ncc9x"


def find_column(results: list[Any], column_name: str) -> list[Any]:
    for df in results:
        if column_name in df.columns:
            return list(df[column_name].values)
    raise ValueError(f"Column '{column_name}' not found in any result dataframe")


def _cascade_chain(
    prefix: str, blocked: str | None = None, deepest_env: str | None = None, root: str = "Sales"
) -> tuple[Feature, Feature, Feature]:
    """Build the x -> y -> z chain cascading env=prod from z; blocked is the level with forward_group=False."""
    x_context: dict[str, Any] = {DefaultOptionKeys.in_features: root, "ident": "identifier1"}
    if deepest_env is not None:
        x_context["env"] = deepest_env
    feat_x = Feature(
        name=f"{prefix}_x",
        options=Options(context=x_context),
        forward_group=False if blocked == "x" else None,
    )
    feat_y = Feature(
        name=f"{prefix}_y",
        options=Options(context={DefaultOptionKeys.in_features: feat_x, "ident": "identifier2"}),
        forward_group=False if blocked == "y" else None,
    )
    feat_z = Feature(
        name=f"{prefix}_z",
        options=Options(
            context={DefaultOptionKeys.in_features: feat_y, "ident": "identifier1", "env": "prod"},
            cascade_context_keys=frozenset({"env"}),
        ),
    )
    return feat_x, feat_y, feat_z


def _guard_requests(target: str, other: str) -> list[Feature]:
    """The cascading consumer and a consumer delivering the same k=v to the shared target another way."""
    return [
        Feature(
            CASCADE_GUARD_CASCADE_CONSUMER,
            options=Options(context={"k": "v", "target": target}, cascade_context_keys=frozenset({"k"})),
        ),
        Feature(
            other,
            options=Options(
                context={"k": "v", "target": target},
                propagate_context_keys=frozenset({"k"}) if other == CASCADE_GUARD_HOP_CONSUMER else None,
            ),
        ),
    ]


def _run_guard(requests: list[Feature]) -> list[Any]:
    return mloda.run_all(
        list[Feature | str](requests),
        compute_frameworks=[PandasDataFrame],
        plugin_collector=PluginCollector.enabled_feature_groups(
            {CascadeGuardRootGroupTest, CascadeGuardSharedGroupTest, CascadeGuardConsumerGroupTest}
        ),
    )


PREVIEW_INPUT = "pvw_input"
PREVIEW_CONSUMER = "pvw_consumer"


class PreviewInputGroupTest(DataCreatorFeatureGroup):
    """Input that records the options it actually receives."""

    RECEIVED: ClassVar[dict[str, Options]] = {}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {PREVIEW_INPUT}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.RECEIVED[PREVIEW_INPUT] = next(iter(features.features)).options
        return pd.DataFrame({PREVIEW_INPUT: [1, 2, 3]})


class PreviewConsumerGroupTest(FeatureGroup):
    """Consumer that records the preview of what core forwards into its input."""

    PREVIEWED: ClassVar[dict[str, Options]] = {}
    PROPERTY_MAPPING = {
        "mode": PropertySpec("Mode", context=False),
        "vault": PropertySpec("Vault", context=False, forward_to_inputs=False),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any | None = None,
    ) -> bool:
        return str(feature_name) == PREVIEW_CONSUMER

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        feature = Feature(PREVIEW_INPUT)
        type(self).PREVIEWED[PREVIEW_CONSUMER] = feature.options_forwarded_from(options, type(self))
        return {feature}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[PREVIEW_CONSUMER] = data[PREVIEW_INPUT] + 1
        return data


class TestForwardedOptionsPreviewE2E:
    def test_preview_matches_the_options_the_input_group_receives(self) -> None:
        PreviewInputGroupTest.RECEIVED.clear()
        PreviewConsumerGroupTest.PREVIEWED.clear()
        request = Feature(
            PREVIEW_CONSUMER,
            options=Options(
                group={"mode": "fast", "vault": "v1"},
                context={"env": "prod"},
                cascade_context_keys=frozenset({"env"}),
            ),
        )

        mloda.run_all(
            [request],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=PluginCollector.enabled_feature_groups({PreviewInputGroupTest, PreviewConsumerGroupTest}),
        )

        received = PreviewInputGroupTest.RECEIVED[PREVIEW_INPUT]
        preview = PreviewConsumerGroupTest.PREVIEWED[PREVIEW_CONSUMER]
        assert preview.group["mode"] == "fast"
        assert preview.context["env"] == "prod"
        assert "vault" not in preview.group
        assert "vault" not in received.group
        assert received.group == preview.group
        assert received.context == preview.context
        assert received.inherited_group_keys == preview.inherited_group_keys
        assert received.inherited_context_keys == preview.inherited_context_keys
        assert "env" in received.context


class TestPropagateContextKeysE2E:
    plugin_collector = PluginCollector.enabled_feature_groups(
        {PropagateContextFeatureGroupTest, ChainerContextParserTestDataCreator}
    )

    @pytest.mark.parametrize("copy_features", [True, False])
    @pytest.mark.parametrize("order", ["cb", "bc"])
    def test_propagation_affects_upstream_computation(self, order: str, copy_features: bool) -> None:
        feat_b = Feature(
            name="feat_b",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                },
            ),
        )
        feat_c = Feature(
            name="feat_c",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: feat_b,
                    "ident": "identifier2",
                    "env": "prod",
                },
                propagate_context_keys=frozenset({"env"}),
            ),
        )
        by_letter = {"b": feat_b, "c": feat_c}
        result = mloda.run_all(
            [*[by_letter[letter] for letter in order], "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
            copy_features=copy_features,
        )
        assert find_column(result, "feat_b") == [1200, 1400, 1600, 1800, 2000]
        assert find_column(result, "feat_c") == [4600, 5200, 5800, 6400, 7000]
        if not copy_features:
            assert "env" not in feat_b.options.context
            assert feat_b.child_options is None

    @staticmethod
    def _dup_requests(prefix: str, order: str, in_form: str, d_env: str = "prod") -> list[Feature | str]:
        """b is the shared input; c and d name it (plain or inside a list) and push env into it."""
        feat_b = Feature(
            name=f"{prefix}_b",
            options=Options(context={DefaultOptionKeys.in_features: "Sales", "ident": "identifier1"}),
        )
        in_value: Any = {"plain": feat_b, "list": [feat_b], "tuple": (feat_b,)}[in_form]
        by_letter = {
            "b": feat_b,
            "c": Feature(
                name=f"{prefix}_c",
                options=Options(
                    context={DefaultOptionKeys.in_features: in_value, "ident": "identifier2", "env": "prod"},
                    propagate_context_keys=frozenset({"env"}),
                ),
            ),
            "d": Feature(
                name=f"{prefix}_d",
                options=Options(
                    context={DefaultOptionKeys.in_features: in_value, "ident": "identifier2", "env": d_env},
                    propagate_context_keys=frozenset({"env"}),
                ),
            ),
        }
        return [*[by_letter[letter] for letter in order], "Sales"]

    @pytest.mark.parametrize("in_form", ["plain", "list", "tuple"])
    @pytest.mark.parametrize("consumers", ["c", "cd"])
    @pytest.mark.parametrize("copy_features", [True, False])
    @pytest.mark.parametrize("order", ["bc", "cb"])
    def test_requested_input_named_by_a_later_request_is_stored_once(
        self, order: str, copy_features: bool, consumers: str, in_form: str
    ) -> None:
        letters = order.replace("c", consumers)
        prefix = f"dup_{order}_{int(copy_features)}_{consumers}_{in_form}"
        session = mloda.prepare(
            self._dup_requests(prefix, letters, in_form),
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
            copy_features=copy_features,
        )
        assert session.engine is not None
        assert session.engine._requested == {}
        stored = [f for group in session.engine.feature_group_collection.values() for f in group]
        for letter in letters:
            assert len([f for f in stored if str(f.name) == f"{prefix}_{letter}"]) == 1

        result = mloda.run_all(
            self._dup_requests(prefix, letters, in_form),
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
            copy_features=copy_features,
        )
        assert find_column(result, f"{prefix}_b") == [1200, 1400, 1600, 1800, 2000]
        for letter in letters.replace("b", ""):
            assert find_column(result, f"{prefix}_{letter}") == [4600, 5200, 5800, 6400, 7000]

    @pytest.mark.parametrize("copy_features", [True, False])
    @pytest.mark.parametrize("order", ["bcd", "cdb", "dcb"])
    def test_requested_input_pushed_differing_values_raises_in_every_order(
        self, order: str, copy_features: bool
    ) -> None:
        prefix = f"twin_{order}_{int(copy_features)}"
        with pytest.raises(ValueError, match="reached twice with differing values"):
            mloda.run_all(
                self._dup_requests(prefix, order, "plain", d_env="staging"),
                compute_frameworks=[PandasDataFrame],
                plugin_collector=self.plugin_collector,
                copy_features=copy_features,
            )

    def test_no_propagation_without_propagate_context_keys(self) -> None:
        feat_b = Feature(
            name="feat_b_noprop",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                },
            ),
        )
        feat_d = Feature(
            name="feat_d",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: feat_b,
                    "ident": "identifier2",
                    "env": "prod",
                },
            ),
        )
        result = mloda.run_all(
            [feat_d, feat_b, "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result, "feat_b_noprop") == [200, 400, 600, 800, 1000]
        assert find_column(result, "feat_d") == [1600, 2200, 2800, 3400, 4000]

    def test_selective_propagation_only_specified_keys(self) -> None:
        feat_b = Feature(
            name="feat_b_sel",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                },
            ),
        )
        feat_e = Feature(
            name="feat_e",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: feat_b,
                    "ident": "identifier2",
                    "env": "staging",
                    "extra_key": "val",
                },
                propagate_context_keys=frozenset({"env"}),
            ),
        )
        result = mloda.run_all(
            [feat_e, feat_b, "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result, "feat_b_sel") == [700, 900, 1100, 1300, 1500]
        assert find_column(result, "feat_e") == [2600, 3200, 3800, 4400, 5000]

    def test_propagation_does_not_affect_sibling_chains(self) -> None:
        feat_base = Feature(
            name="feat_base",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                },
            ),
        )
        feat_with_prop = Feature(
            name="feat_with_prop",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: feat_base,
                    "ident": "identifier2",
                    "env": "staging",
                },
                propagate_context_keys=frozenset({"env"}),
            ),
        )
        # feat_base receives env="staging" from feat_with_prop -> offset=500
        # feat_base values: Sales*2+500 = [700, 900, 1100, 1300, 1500]
        # feat_with_prop values: feat_base*3+500 = [2600, 3200, 3800, 4400, 5000]
        result = mloda.run_all(
            [feat_with_prop, feat_base, "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result, "feat_base") == [700, 900, 1100, 1300, 1500]
        assert find_column(result, "feat_with_prop") == [2600, 3200, 3800, 4400, 5000]

        # Now create an independent chain with the same base pattern but no propagation
        feat_base2 = Feature(
            name="feat_base2",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                },
            ),
        )
        feat_no_prop = Feature(
            name="feat_no_prop",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: feat_base2,
                    "ident": "identifier2",
                    "env": "staging",
                },
            ),
        )
        # feat_base2 does NOT receive env -> offset=0
        # feat_base2 values: Sales*2+0 = [200, 400, 600, 800, 1000]
        # feat_no_prop values: feat_base2*3+500 = [1100, 1700, 2300, 2900, 3500]
        result2 = mloda.run_all(
            [feat_no_prop, feat_base2, "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result2, "feat_base2") == [200, 400, 600, 800, 1000]
        assert find_column(result2, "feat_no_prop") == [1100, 1700, 2300, 2900, 3500]

    def test_propagation_single_hop_in_multi_level_chain(self) -> None:
        """Verify propagate_context_keys only propagates one hop, not cascading further.

        3-level chain: feat_z -> feat_y -> feat_x -> Sales

        Only feat_z declares propagate_context_keys=frozenset({"env"}) with env="prod".
        This should propagate env="prod" to feat_y (direct dependency) but NOT to feat_x.

        Expected:
          feat_x: ident="identifier1" (mult=2), no env -> offset=0
            Sales*2 = [200, 400, 600, 800, 1000]
          feat_y: ident="identifier2" (mult=3), receives env="prod" from feat_z -> offset=1000
            feat_x*3+1000 = [1600, 2200, 2800, 3400, 4000]
          feat_z: ident="identifier1" (mult=2), env="prod" natively -> offset=1000
            feat_y*2+1000 = [4200, 5400, 6600, 7800, 9000]
        """
        feat_x = Feature(
            name="feat_x",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                },
            ),
        )
        feat_y = Feature(
            name="feat_y",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: feat_x,
                    "ident": "identifier2",
                },
            ),
        )
        feat_z = Feature(
            name="feat_z",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: feat_y,
                    "ident": "identifier1",
                    "env": "prod",
                },
                propagate_context_keys=frozenset({"env"}),
            ),
        )
        result = mloda.run_all(
            [feat_z, feat_y, feat_x, "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result, "feat_x") == [200, 400, 600, 800, 1000]
        assert find_column(result, "feat_y") == [1600, 2200, 2800, 3400, 4000]
        assert find_column(result, "feat_z") == [4200, 5400, 6600, 7800, 9000]

    def _run_name_chained(self, options: Options) -> list[Any]:
        return mloda.run_all(
            [Feature(name=NAME_CHAINED_TOP, options=options)],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {NameChainedCascadeGroupTest, ChainerContextParserTestDataCreator}
            ),
        )

    def test_cascade_through_name_chained_features(self) -> None:
        result = self._run_name_chained(
            Options(context={"env": "prod"}, cascade_context_keys=frozenset({"env"})),
        )
        assert find_column(result, NAME_CHAINED_TOP) == [7800, 8600, 9400, 10200, 11000]

    def test_one_hop_propagation_fails_in_name_chained_features(self) -> None:
        with pytest.raises(Exception, match="needs env=prod"):
            self._run_name_chained(Options(context={"env": "prod"}, propagate_context_keys=frozenset({"env"})))

    @pytest.mark.parametrize(
        "blocked, expected_x, expected_y, expected_z",
        [
            ("x", [200, 400, 600, 800, 1000], [1600, 2200, 2800, 3400, 4000], [4200, 5400, 6600, 7800, 9000]),
            ("y", [200, 400, 600, 800, 1000], [600, 1200, 1800, 2400, 3000], [2200, 3400, 4600, 5800, 7000]),
            (None, [1200, 1400, 1600, 1800, 2000], [4600, 5200, 5800, 6400, 7000], [10200, 11400, 12600, 13800, 15000]),
        ],
    )
    def test_cascade_reaches_or_stops_at_forward_group_false(
        self, blocked: str | None, expected_x: list[int], expected_y: list[int], expected_z: list[int]
    ) -> None:
        prefix = f"casc_stop_{blocked}"
        stop_x, stop_y, stop_z = _cascade_chain(prefix, blocked=blocked)
        result = mloda.run_all(
            [stop_z, stop_y, stop_x, "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result, f"{prefix}_x") == expected_x
        assert find_column(result, f"{prefix}_y") == expected_y
        assert find_column(result, f"{prefix}_z") == expected_z

    def test_cascade_conflict_at_deepest_feature_raises(self) -> None:
        deep_x, deep_y, deep_z = _cascade_chain("casc_conflict", deepest_env="staging")
        with pytest.raises(ValueError, match="cascade_context_keys") as excinfo:
            mloda.run_all(
                [deep_z, deep_y, deep_x, "Sales"],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=self.plugin_collector,
            )
        assert "casc_conflict_x" in str(excinfo.value)

    @pytest.mark.parametrize("other", [CASCADE_GUARD_PULL_CONSUMER, CASCADE_GUARD_HOP_CONSUMER])
    @pytest.mark.parametrize("cascade_first", [True, False])
    def test_shared_input_with_differing_cascade_raises(self, other: str, cascade_first: bool) -> None:
        requests = _guard_requests(CASCADE_GUARD_SHARED, other)
        with pytest.raises(ValueError) as excinfo:
            _run_guard(requests if cascade_first else requests[::-1])
        assert CASCADE_GUARD_SHARED in str(excinfo.value)
        assert "cascade_context_keys" in str(excinfo.value)
        assert "inputs were already resolved" in str(excinfo.value)

    @pytest.mark.parametrize("cascade_first", [True, False])
    def test_inputs_differing_in_compute_framework_fail_at_plan_time(self, cascade_first: bool) -> None:
        requests = [
            Feature(
                CASCADE_FOLD_CASCADE_CONSUMER,
                options=Options(context={"k": "v"}, cascade_context_keys=frozenset({"k"})),
            ),
            Feature(CASCADE_FOLD_PINNED_CONSUMER, options=Options(context={"k": "v"})),
        ]
        with pytest.raises(ValueError) as excinfo:
            mloda.run_all(
                list[Feature | str](requests if cascade_first else requests[::-1]),
                compute_frameworks=[PandasDataFrame, PyArrowTable],
                plugin_collector=PluginCollector.enabled_feature_groups(
                    {CascadeGuardRootGroupTest, CascadeFoldSharedGroupTest, CascadeFoldConsumerGroupTest}
                ),
            )
        assert "from two steps" in str(excinfo.value)
        assert "differ in options" in str(excinfo.value)

    @pytest.mark.parametrize("other", [CASCADE_GUARD_PULL_CONSUMER, CASCADE_GUARD_HOP_CONSUMER])
    @pytest.mark.parametrize("cascade_first", [True, False])
    def test_shared_root_with_differing_cascade_does_not_raise(self, other: str, cascade_first: bool) -> None:
        requests = _guard_requests(CASCADE_GUARD_ROOT, other)
        result = _run_guard(requests if cascade_first else requests[::-1])
        assert find_column(result, CASCADE_GUARD_CASCADE_CONSUMER) == [2, 3, 4]
        assert find_column(result, other) == [2, 3, 4]

    _ENV_ROOT_COLLECTOR = PluginCollector.enabled_feature_groups(
        {PropagateContextFeatureGroupTest, CascadeEnvRootGroupTest}
    )
    _CHAIN_ORDERS_OK = ["z", "zy", "yz", "zx", "xz", "zyx", "zxy", "yzx", "yxz", "xzy", "xyz"]
    _CHAIN_EXPECTED = {
        "x": [1020, 1040, 1060],
        "y": [4060, 4120, 4180],
        "z": [9120, 9240, 9360],
    }

    def _run_env_root(self, requests: list[Feature], copy_features: bool = True) -> list[Any]:
        return mloda.run_all(
            list[Feature | str](requests),
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self._ENV_ROOT_COLLECTOR,
            copy_features=copy_features,
        )

    @pytest.mark.parametrize("copy_features", [True, False])
    @pytest.mark.parametrize("order", _CHAIN_ORDERS_OK)
    def test_shared_chain_cascade_reaches_root_for_any_order(self, order: str, copy_features: bool) -> None:
        prefix = f"cer_ok_{order}_{int(copy_features)}"
        chain = dict(zip("xyz", _cascade_chain(prefix, root=CASCADE_ENV_ROOT)))
        result = self._run_env_root([chain[level] for level in order], copy_features)
        for level in order:
            assert find_column(result, f"{prefix}_{level}") == self._CHAIN_EXPECTED[level]

    @pytest.mark.parametrize("copy_features", [True, False])
    def test_chain_yz_stores_each_object_once(self, copy_features: bool) -> None:
        def prepare(order: str) -> Any:
            chain = dict(zip("xyz", _cascade_chain(f"cer_yz_once_{int(copy_features)}", root=CASCADE_ENV_ROOT)))
            return mloda.prepare(
                list[Feature | str]([chain[level] for level in order]),
                compute_frameworks=[PandasDataFrame],
                plugin_collector=self._ENV_ROOT_COLLECTOR,
                copy_features=copy_features,
            )

        session = prepare("yz")
        assert session.engine is not None
        for features in session.engine.feature_group_collection.values():
            stored = list(features)
            assert len(stored) == len({id(f) for f in stored})
            assert len(stored) == len({f.uuid for f in stored})
        assert len(session.resolved_plan()) == len(prepare("zy").resolved_plan())

    @pytest.mark.parametrize("order", ["ab", "ba"])
    def test_shared_mid_gaining_cascade_from_second_consumer(self, order: str) -> None:
        mid = Feature(
            name=f"cer_mid_{order}",
            options=Options(context={DefaultOptionKeys.in_features: CASCADE_ENV_ROOT, "ident": "identifier1"}),
        )
        consumers = {
            "a": Feature(
                name=f"cer_a_{order}",
                options=Options(context={DefaultOptionKeys.in_features: mid, "ident": "identifier1"}),
            ),
            "b": Feature(
                name=f"cer_b_{order}",
                options=Options(
                    context={DefaultOptionKeys.in_features: mid, "ident": "identifier1", "env": "prod"},
                    cascade_context_keys=frozenset({"env"}),
                ),
            ),
        }
        result = self._run_env_root([consumers[name] for name in order])
        assert find_column(result, f"cer_a_{order}") == [4, 8, 12]
        assert find_column(result, f"cer_b_{order}") == [3040, 3080, 3120]
