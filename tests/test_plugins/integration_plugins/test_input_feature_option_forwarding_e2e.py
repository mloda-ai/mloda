"""End-to-end coverage for input-feature option forwarding (issue #579).

Pins the forward-by-default definition of done at the mloda.run_all API level:
1. Forward by default: consumer group options flow to upstream input features that
   carry no directive (the unspecified None sentinel); in_features never flows.
2. Allowlist: Feature.forward_group forwards exactly the listed consumer group keys.
3. Isolation: Feature.forward_group=False keeps every consumer group option behind.
4. Exclude: Feature.forward_group_exclude carves single keys out of the default copy.
5. Context pull: Feature.inherit_context_keys pulls only the listed consumer context keys.
6. Chained features: FeatureChainParserMixin children receive full consumer inheritance
   via the engine default.
7. Upstream dedup: consumers differing only in an excluded consumer group option share
   one upstream computation because the excluded upstream features are identical.

The upstream source group records the options it is resolved with into class-level
state, which every test resets before running. All fixture feature names carry a
"forwarding_e2e_579" marker so they cannot collide with other tests in the global
plugin registry.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import pandas as pd

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureChainParserMixin
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


SOURCE_NAME = "forwarding_e2e_579_source"


class ForwardingE2ESourceGroup(FeatureGroup):
    """Upstream root group that records the options its resolved features carry.

    Returns [10, 20, 30] when the "backend" option arrives with value "premium",
    otherwise [1, 2, 3], so option forwarding is visible in the computed output.
    """

    seen_options: ClassVar[list[dict[str, dict[str, Any]]]] = []
    invocation_count: ClassVar[int] = 0

    @classmethod
    def reset_recording(cls) -> None:
        cls.seen_options = []
        cls.invocation_count = 0

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({SOURCE_NAME})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.invocation_count += 1
        backend = None
        for feature in features.features:
            cls.seen_options.append(
                {
                    "group": dict(feature.options.group),
                    "context": dict(feature.options.context),
                }
            )
            backend = feature.options.get("backend")
        values = [10, 20, 30] if backend == "premium" else [1, 2, 3]
        return pd.DataFrame({SOURCE_NAME: values})


class _ForwardingE2EConsumerBase(FeatureGroup):
    """Shared consumer behavior: multiply the upstream column by the top_k group option."""

    FEATURE_NAME: ClassVar[str] = ""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == cls.FEATURE_NAME

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            factor = feature.options.get("top_k")
            if factor is None:
                factor = 1
            data[feature.name] = data[SOURCE_NAME] * factor
        return data


class ForwardingE2EDefaultConsumer(_ForwardingE2EConsumerBase):
    """Consumer with default forwarding: no directive on its input feature."""

    FEATURE_NAME = "forwarding_e2e_579_default_consumer"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME)}


class ForwardingE2EAllowlistConsumer(_ForwardingE2EConsumerBase):
    """Consumer that allowlists exactly the "backend" group key for its input feature."""

    FEATURE_NAME = "forwarding_e2e_579_allowlist_consumer"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME, forward_group={"backend"})}


class ForwardingE2EIsolatedConsumer(_ForwardingE2EConsumerBase):
    """Consumer whose input feature opts out of all forwarding via forward_group=False."""

    FEATURE_NAME = "forwarding_e2e_579_isolated_consumer"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME, forward_group=False)}


class ForwardingE2EExcludeConsumer(_ForwardingE2EConsumerBase):
    """Consumer whose input feature carves "top_k" out of the default forwarding."""

    FEATURE_NAME = "forwarding_e2e_579_exclude_consumer"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME, forward_group_exclude={"top_k"})}


class ForwardingE2EContextConsumer(_ForwardingE2EConsumerBase):
    """Consumer whose input feature pulls only the "tenant" context key from the consumer."""

    FEATURE_NAME = "forwarding_e2e_579_context_consumer"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME, inherit_context_keys={"tenant"})}


class ForwardingE2EDedupConsumer(_ForwardingE2EConsumerBase):
    """Consumer used twice with different consumer-only group options to test upstream dedup.

    Its input feature excludes "top_k", so consumers that differ only in top_k build
    identical upstream features. Distinct output names per top_k variant: both consumer
    FeatureSets operate on the shared upstream frame, so writing one shared column name
    would let the second computation overwrite the first.
    """

    FEATURE_NAME = "forwarding_e2e_579_dedup_consumer"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_NAME, forward_group_exclude={"top_k"})}

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        return FeatureName(f"{self.FEATURE_NAME}_top{config.get('top_k')}")


class ForwardingE2EChainedGroup(FeatureChainParserMixin, FeatureGroup):
    """Minimal chainer group: <source>__double_e2echain579fwd doubles the source values."""

    PREFIX_PATTERN = r".*__([\w]+)_e2echain579fwd$"
    PROPERTY_MAPPING = {
        "operation": {
            DefaultOptionKeys.allowed_values: {"double": "Doubles the source values"},
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source features",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            source = cls._extract_source_features(feature)[0]
            data[feature.name] = data[source] * 2
        return data


def _frame_with_column(results: list[Any], column: str) -> Any:
    for frame in results:
        if column in frame.columns:
            return frame
    raise AssertionError(f"Column '{column}' not found in any result frame: {[list(r.columns) for r in results]}")


def _run(features: list[Feature | str], groups: set[type[FeatureGroup]]) -> list[Any]:
    return mloda.run_all(
        features,
        compute_frameworks={PandasDataFrame},
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )


class TestInputFeatureOptionForwardingE2E:
    def test_default_forwards_consumer_group_options_upstream(self) -> None:
        """Forward by default: backend AND top_k reach the upstream source; in_features never does."""
        ForwardingE2ESourceGroup.reset_recording()

        consumer = Feature(
            ForwardingE2EDefaultConsumer.FEATURE_NAME,
            options=Options(group={"backend": "premium", "top_k": 3}),
        )
        results = _run([consumer], {ForwardingE2ESourceGroup, ForwardingE2EDefaultConsumer})

        # Upstream saw backend=premium, so the source is [10, 20, 30]; consumer applies top_k=3.
        frame = _frame_with_column(results, ForwardingE2EDefaultConsumer.FEATURE_NAME)
        assert frame[ForwardingE2EDefaultConsumer.FEATURE_NAME].tolist() == [30, 60, 90]

        assert ForwardingE2ESourceGroup.invocation_count == 1
        assert len(ForwardingE2ESourceGroup.seen_options) == 1
        upstream = ForwardingE2ESourceGroup.seen_options[0]
        assert upstream["group"].get("backend") == "premium"
        assert upstream["group"].get("top_k") == 3
        assert DefaultOptionKeys.in_features not in upstream["group"]
        assert DefaultOptionKeys.in_features not in upstream["context"]

    def test_forward_group_allowlists_backend_to_upstream(self) -> None:
        """forward_group={"backend"} forwards exactly backend; top_k stays consumer-only."""
        ForwardingE2ESourceGroup.reset_recording()

        consumer = Feature(
            ForwardingE2EAllowlistConsumer.FEATURE_NAME,
            options=Options(group={"backend": "premium", "top_k": 3}),
        )
        results = _run([consumer], {ForwardingE2ESourceGroup, ForwardingE2EAllowlistConsumer})

        # Upstream saw backend=premium, so the source is [10, 20, 30]; consumer applies top_k=3.
        frame = _frame_with_column(results, ForwardingE2EAllowlistConsumer.FEATURE_NAME)
        assert frame[ForwardingE2EAllowlistConsumer.FEATURE_NAME].tolist() == [30, 60, 90]

        assert len(ForwardingE2ESourceGroup.seen_options) == 1
        upstream = ForwardingE2ESourceGroup.seen_options[0]
        assert upstream["group"].get("backend") == "premium"
        assert "top_k" not in upstream["group"]
        assert "top_k" not in upstream["context"]

    def test_forward_group_false_isolates_upstream(self) -> None:
        """forward_group=False: neither backend nor top_k reaches the upstream source."""
        ForwardingE2ESourceGroup.reset_recording()

        consumer = Feature(
            ForwardingE2EIsolatedConsumer.FEATURE_NAME,
            options=Options(group={"backend": "premium", "top_k": 3}),
        )
        results = _run([consumer], {ForwardingE2ESourceGroup, ForwardingE2EIsolatedConsumer})

        # Upstream never saw backend=premium, so the source stays [1, 2, 3]; consumer applies top_k=3.
        frame = _frame_with_column(results, ForwardingE2EIsolatedConsumer.FEATURE_NAME)
        assert frame[ForwardingE2EIsolatedConsumer.FEATURE_NAME].tolist() == [3, 6, 9]

        assert len(ForwardingE2ESourceGroup.seen_options) == 1
        upstream = ForwardingE2ESourceGroup.seen_options[0]
        assert "backend" not in upstream["group"]
        assert "backend" not in upstream["context"]
        assert "top_k" not in upstream["group"]
        assert "top_k" not in upstream["context"]

    def test_forward_group_exclude_carves_top_k_out_of_default(self) -> None:
        """forward_group_exclude={"top_k"}: backend flows via the default; top_k is carved out."""
        ForwardingE2ESourceGroup.reset_recording()

        consumer = Feature(
            ForwardingE2EExcludeConsumer.FEATURE_NAME,
            options=Options(group={"backend": "premium", "top_k": 3}),
        )
        results = _run([consumer], {ForwardingE2ESourceGroup, ForwardingE2EExcludeConsumer})

        # Upstream saw backend=premium, so the source is [10, 20, 30]; consumer applies top_k=3.
        frame = _frame_with_column(results, ForwardingE2EExcludeConsumer.FEATURE_NAME)
        assert frame[ForwardingE2EExcludeConsumer.FEATURE_NAME].tolist() == [30, 60, 90]

        assert len(ForwardingE2ESourceGroup.seen_options) == 1
        upstream = ForwardingE2ESourceGroup.seen_options[0]
        assert upstream["group"].get("backend") == "premium"
        assert "top_k" not in upstream["group"]
        assert "top_k" not in upstream["context"]

    def test_inherit_context_keys_pulls_tenant_only(self) -> None:
        """inherit_context_keys={"tenant"} pulls tenant; other context keys stay behind."""
        ForwardingE2ESourceGroup.reset_recording()

        consumer = Feature(
            ForwardingE2EContextConsumer.FEATURE_NAME,
            options=Options(context={"tenant": "acme", "query_text": "hello"}),
        )
        results = _run([consumer], {ForwardingE2ESourceGroup, ForwardingE2EContextConsumer})

        frame = _frame_with_column(results, ForwardingE2EContextConsumer.FEATURE_NAME)
        assert frame[ForwardingE2EContextConsumer.FEATURE_NAME].tolist() == [1, 2, 3]

        assert len(ForwardingE2ESourceGroup.seen_options) == 1
        upstream = ForwardingE2ESourceGroup.seen_options[0]
        assert upstream["context"].get("tenant") == "acme"
        assert "query_text" not in upstream["context"]
        assert "query_text" not in upstream["group"]

    def test_chained_feature_still_forwards_group_options(self) -> None:
        """FeatureChainParserMixin children receive consumer group options via the engine
        default (no forward_group stamping needed): chains are unaffected by the flip."""
        ForwardingE2ESourceGroup.reset_recording()

        chained_name = f"{SOURCE_NAME}__double_e2echain579fwd"
        chained = Feature(chained_name, options=Options(group={"passthrough_marker": "carried"}))
        results = _run([chained], {ForwardingE2ESourceGroup, ForwardingE2EChainedGroup})

        frame = _frame_with_column(results, chained_name)
        assert frame[chained_name].tolist() == [2, 4, 6]

        assert len(ForwardingE2ESourceGroup.seen_options) == 1
        upstream = ForwardingE2ESourceGroup.seen_options[0]
        assert upstream["group"].get("passthrough_marker") == "carried"
        assert DefaultOptionKeys.in_features not in upstream["group"]

    def test_excluding_consumers_share_single_upstream_computation(self) -> None:
        """Consumers differing only in the excluded top_k share one upstream run: the
        forwarded backend is equal on both, so the upstream features stay identical."""
        ForwardingE2ESourceGroup.reset_recording()

        consumers: list[Feature | str] = [
            Feature(
                ForwardingE2EDedupConsumer.FEATURE_NAME,
                options=Options(group={"backend": "premium", "top_k": 3}),
            ),
            Feature(
                ForwardingE2EDedupConsumer.FEATURE_NAME,
                options=Options(group={"backend": "premium", "top_k": 5}),
            ),
        ]
        results = _run(consumers, {ForwardingE2ESourceGroup, ForwardingE2EDedupConsumer})

        # Upstream saw backend=premium once, so the shared source is [10, 20, 30].
        frame_top3 = _frame_with_column(results, f"{ForwardingE2EDedupConsumer.FEATURE_NAME}_top3")
        frame_top5 = _frame_with_column(results, f"{ForwardingE2EDedupConsumer.FEATURE_NAME}_top5")
        assert frame_top3[f"{ForwardingE2EDedupConsumer.FEATURE_NAME}_top3"].tolist() == [30, 60, 90]
        assert frame_top5[f"{ForwardingE2EDedupConsumer.FEATURE_NAME}_top5"].tolist() == [50, 100, 150]

        # Excluding top_k keeps both upstream input features identical, so the source computes once.
        assert ForwardingE2ESourceGroup.invocation_count == 1
        for upstream in ForwardingE2ESourceGroup.seen_options:
            assert upstream["group"].get("backend") == "premium"
            assert "top_k" not in upstream["group"]
            assert "top_k" not in upstream["context"]
