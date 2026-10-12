"""E2E integration tests for propagate_context_keys in chained features."""

from typing import Any

import pytest

from mloda.user import Feature, Options, PluginCollector, mloda
from mloda.provider import DefaultOptionKeys
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_plugins.integration_plugins.chainer.propagate_context_feature import (
    CASCADE_ENV_ROOT,
    CASCADE_GUARD_CASCADE_CONSUMER,
    CASCADE_GUARD_HOP_CONSUMER,
    CASCADE_GUARD_PULL_CONSUMER,
    CASCADE_GUARD_ROOT,
    CASCADE_GUARD_SHARED,
    CascadeEnvRootGroupTest,
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


class TestPropagateContextKeysE2E:
    plugin_collector = PluginCollector.enabled_feature_groups(
        {PropagateContextFeatureGroupTest, ChainerContextParserTestDataCreator}
    )

    def test_propagation_affects_upstream_computation(self) -> None:
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
        result = mloda.run_all(
            [feat_c, feat_b, "Sales"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result, "feat_b") == [1200, 1400, 1600, 1800, 2000]
        assert find_column(result, "feat_c") == [4600, 5200, 5800, 6400, 7000]

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
    _CHAIN_ORDERS_OK = ["zyx", "zxy", "z", "zy", "zx", "xzy", "xz"]
    _CHAIN_ORDERS_RAISE = ["yzx", "yxz", "xyz", "yz"]
    _CHAIN_EXPECTED = {
        "x": [1020, 1040, 1060],
        "y": [4060, 4120, 4180],
        "z": [9120, 9240, 9360],
    }

    def _run_env_root(self, requests: list[Feature]) -> list[Any]:
        return mloda.run_all(
            list[Feature | str](requests),
            compute_frameworks=[PandasDataFrame],
            plugin_collector=self._ENV_ROOT_COLLECTOR,
        )

    @pytest.mark.parametrize("order", _CHAIN_ORDERS_OK)
    def test_shared_chain_cascade_reaches_root_for_any_successful_order(self, order: str) -> None:
        prefix = f"cer_ok_{order}"
        chain = dict(zip("xyz", _cascade_chain(prefix, root=CASCADE_ENV_ROOT)))
        result = self._run_env_root([chain[level] for level in order])
        for level in order:
            assert find_column(result, f"{prefix}_{level}") == self._CHAIN_EXPECTED[level]

    @pytest.mark.parametrize("order", _CHAIN_ORDERS_RAISE)
    def test_shared_chain_gaining_cascade_after_inputs_resolved_raises(self, order: str) -> None:
        prefix = f"cer_raise_{order}"
        chain = dict(zip("xyz", _cascade_chain(prefix, root=CASCADE_ENV_ROOT)))
        with pytest.raises(ValueError) as excinfo:
            self._run_env_root([chain[level] for level in order])
        assert "cascade_context_keys" in str(excinfo.value)
        assert any(f"{prefix}_{level}" in str(excinfo.value) for level in "xy")

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
        requests = [consumers[name] for name in order]
        if order == "ab":
            with pytest.raises(ValueError) as excinfo:
                self._run_env_root(requests)
            assert f"cer_mid_{order}" in str(excinfo.value)
            assert "cascade_context_keys" in str(excinfo.value)
        else:
            result = self._run_env_root(requests)
            assert find_column(result, f"cer_a_{order}") == [2040, 2080, 2120]
            assert find_column(result, f"cer_b_{order}") == [3040, 3080, 3120]
