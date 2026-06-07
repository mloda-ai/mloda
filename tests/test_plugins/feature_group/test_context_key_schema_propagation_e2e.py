"""E2E regression tests for the chained-propagation context-key exemption.

These tests exercise the real engine run path (``mloda.run_all``), NOT a direct
call to ``Options.update_with_protected_keys``. They lock in the live-path
behavior on this branch.

Scenario: a parent feature ``P`` declares
``propagate_context_keys=frozenset({"bespoke_env"})`` with ``bespoke_env`` in its
context. ``P`` has an input feature ``D`` whose FeatureGroup opts into
context-key validation via ``context_key_schema() -> derive_context_key_schema()``
but does NOT declare ``bespoke_env`` in its ``PROPERTY_MAPPING``.

During the run the engine merges ``P``'s propagated context value INTO ``D``'s
context BEFORE ``D``'s own planning/validation runs, populating
``D.options.received_propagated_keys``. The validator exempts received-propagated
keys, so the run must succeed and must NOT raise "unknown context key
'bespoke_env'".
"""

from typing import Any, Optional

import pytest

from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureChainParserMixin
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ContextSchemaPropagationDataCreator(ATestDataCreator):
    """Source feature group producing a small Sales frame on PandasDataFrame."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {"Sales": [100, 200, 300, 400, 500]}


class ContextSchemaPropagationConsumer(FeatureChainParserMixin, FeatureGroup):
    """Config-based consumer that opts into context-key schema validation.

    IMPORTANT: ``bespoke_env`` is intentionally NOT part of ``PROPERTY_MAPPING``.
    The derived schema therefore does not contain ``bespoke_env``; the run can
    only succeed if the propagated key is exempt via
    ``received_propagated_keys`` on the live path.
    """

    PROPERTY_MAPPING = {
        "ident": {
            "identifier1": "multiplier 2",
            "identifier2": "multiplier 3",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "label": {
            "explanation": "Optional, unused label used by the typo guard test",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.default: None,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature(s)",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def context_key_schema(cls) -> Optional[dict[str, Any]]:
        return cls.derive_context_key_schema()

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        features: set[Feature] = set()
        for source_feature in options.get_in_features():
            source_feature.options.add_to_group(
                DefaultOptionKeys.feature_chainer_parser_key,
                frozenset(["ident", DefaultOptionKeys.in_features]),
            )
            features.add(source_feature)
        if features:
            return features
        raise ValueError("No input features found")

    @classmethod
    def perform_operation(cls, data: Any, feature: Feature) -> Any:
        source_feature = next(iter(feature.options.get_in_features()))
        source_feature_name: str = source_feature.name

        ident = feature.options.get("ident")
        if ident == "identifier1":
            multiplier = 2
        elif ident == "identifier2":
            multiplier = 3
        else:
            raise ValueError(f"Unknown ident value: {ident}")

        bespoke_env = feature.options.get("bespoke_env")
        if bespoke_env == "prod":
            offset = 1000
        elif bespoke_env == "staging":
            offset = 500
        else:
            offset = 0

        data[feature.name] = data[source_feature_name] * multiplier + offset
        return data

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data = cls.perform_operation(data, feature)
        return data


def find_column(results: list[Any], column_name: str) -> list[Any]:
    for df in results:
        if column_name in df.columns:
            return list(df[column_name].values)
    raise ValueError(f"Column '{column_name}' not found in any result dataframe")


class TestContextKeySchemaPropagationE2E:
    plugin_collector = PluginCollector.enabled_feature_groups(
        {ContextSchemaPropagationConsumer, ContextSchemaPropagationDataCreator}
    )

    def test_propagated_context_key_is_exempt_on_live_path(self) -> None:
        """The propagated key 'bespoke_env' is not in the consumer's schema, yet
        the run succeeds because it is exempt as a received-propagated key.

        consumer_child: Sales*2 + offset(bespoke_env=prod=1000)
            = [200,400,600,800,1000] + 1000 = [1200,1400,1600,1800,2000]
        consumer_parent: child*3 + offset(prod=1000)
            = [3600,4200,4800,5400,6000] + 1000 = [4600,5200,5800,6400,7000]

        Asserting the child equals [1200,...] proves both that planning did not
        raise "unknown context key 'bespoke_env'" AND that the value was actually
        propagated into the child's context (without propagation it would be
        [200,400,600,800,1000]).
        """
        consumer_child = Feature(
            name="consumer_child",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                },
            ),
        )
        consumer_parent = Feature(
            name="consumer_parent",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: consumer_child,
                    "ident": "identifier2",
                    "bespoke_env": "prod",
                },
                propagate_context_keys=frozenset({"bespoke_env"}),
            ),
        )

        result = mloda.run_all(
            [consumer_parent, consumer_child, "Sales"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=self.plugin_collector,
        )

        assert find_column(result, "consumer_child") == [1200, 1400, 1600, 1800, 2000]
        assert find_column(result, "consumer_parent") == [4600, 5200, 5800, 6400, 7000]

    def test_typo_on_optional_context_key_still_raises(self) -> None:
        """Guard: prove validation is actually active on the opt-in consumer.

        'label' is a legitimate optional PROPERTY_MAPPING key. Requesting the
        consumer directly with a typo'd 'labal' (close to 'label') still resolves
        to the consumer (matching ignores extra context keys and 'label' is
        optional), so context-key validation fires and raises a ValueError with a
        "did you mean" suggestion.
        """
        typo_feature = Feature(
            name="consumer_typo",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: "Sales",
                    "ident": "identifier1",
                    "labal": "oops",
                },
            ),
        )

        with pytest.raises(ValueError, match="labal"):
            mloda.run_all(
                [typo_feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=self.plugin_collector,
            )

        with pytest.raises(ValueError, match="did you mean"):
            mloda.run_all(
                [typo_feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=self.plugin_collector,
            )
