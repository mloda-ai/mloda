"""E2E integration tests for propagate_context_keys in chained features."""

from typing import Any, List

from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_plugins.integration_plugins.chainer.propagate_context_feature import PropagateContextFeatureGroupTest
from tests.test_plugins.integration_plugins.chainer.context.test_chained_optional_features import (
    ChainerContextParserTestDataCreator,
)


def find_column(results: List[Any], column_name: str) -> List[Any]:
    for df in results:
        if column_name in df.columns:
            return list(df[column_name].values)
    raise ValueError(f"Column '{column_name}' not found in any result dataframe")


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
            compute_frameworks={PandasDataFrame},
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
            compute_frameworks={PandasDataFrame},
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
            compute_frameworks={PandasDataFrame},
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
            compute_frameworks={PandasDataFrame},
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
            compute_frameworks={PandasDataFrame},
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
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )
        assert find_column(result, "feat_x") == [200, 400, 600, 800, 1000]
        assert find_column(result, "feat_y") == [1600, 2200, 2800, 3400, 4000]
        assert find_column(result, "feat_z") == [4200, 5400, 6600, 7800, 9000]
