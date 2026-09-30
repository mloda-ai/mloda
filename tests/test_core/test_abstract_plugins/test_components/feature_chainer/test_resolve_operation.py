"""Tests for FeatureChainParserMixin._resolve_operation helper."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature
from mloda.user import mloda
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda.provider import DefaultOptionKeys, PropertySpec
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

from tests.test_core.test_abstract_plugins.test_components.feature_chainer.test_parsed_name_bindings import (
    SIZE_KEY,
    OptionalCaptureGroup,
)
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class MockResolverFG(FeatureChainParserMixin):
    """Mock feature group for testing _resolve_operation."""

    PREFIX_PATTERN = r".*__([\w]+)_op$"
    AGGREGATION_TYPE = "aggregation_type"

    PROPERTY_MAPPING = {
        "aggregation_type": PropertySpec(
            "Aggregation to apply",
            allowed_values={"sum": "Sum", "avg": "Average"},
            context=True,
            strict_validation=True,
        ),
    }


class TestResolveOperationUnit:
    """Unit tests for _resolve_operation."""

    def test_returns_parsed_operation_from_string(self) -> None:
        """When feature name matches PREFIX_PATTERN, returns parsed operation."""
        options = Options(context={"aggregation_type": "sum"})
        result = MockResolverFG._resolve_operation("source__sum_op", options, "aggregation_type")
        assert result == "sum"

    def test_config_fallback_unpacks_a_singleton_option(self) -> None:
        feature = Feature("config_feature", Options(context={"aggregation_type": ["sum"]}))

        assert AggregatedFeatureGroup._resolve_operation(feature, "aggregation_type") == "sum"

    def test_returns_config_when_pattern_does_not_match(self) -> None:
        """When feature name does not match, falls back to options[config_key]."""
        options = Options(context={"aggregation_type": "avg"})
        result = MockResolverFG._resolve_operation("plain_name", options, "aggregation_type")
        assert result == "avg"

    def test_returns_none_when_neither_matches(self) -> None:
        """When neither pattern nor config key matches, returns None."""
        options = Options(context={})
        result = MockResolverFG._resolve_operation("plain_name", options, "aggregation_type")
        assert result is None

    def test_string_path_takes_precedence_over_config(self) -> None:
        """String-based resolution takes precedence even if config key is also set."""
        options = Options(context={"aggregation_type": "avg"})
        result = MockResolverFG._resolve_operation("source__sum_op", options, "aggregation_type")
        assert result == "sum"

    def test_works_with_feature_name_object(self) -> None:
        """Accepts FeatureName objects as well as strings."""
        options = Options(context={"aggregation_type": "sum"})
        result = MockResolverFG._resolve_operation(FeatureName("source__sum_op"), options, "aggregation_type")
        assert result == "sum"


class TestResolveOperationKeyAware:
    """A named pattern resolves each key from its own capture; a positional one keeps first-capture semantics."""

    @pytest.mark.parametrize(
        "key,expected",
        [
            (TimeWindowFeatureGroup.TIME_UNIT, "day"),
            (TimeWindowFeatureGroup.WINDOW_SIZE, "7"),
            (TimeWindowFeatureGroup.WINDOW_FUNCTION, "sum"),
        ],
    )
    def test_named_pattern_resolves_each_key_from_its_own_capture(self, key: str, expected: str) -> None:
        assert TimeWindowFeatureGroup._resolve_operation(Feature("x__sum_7_day_window"), key) == expected

    def test_named_pattern_wins_over_a_present_option(self) -> None:
        options = Options(context={TimeWindowFeatureGroup.TIME_UNIT: "hour"})
        result = TimeWindowFeatureGroup._resolve_operation(
            "x__sum_7_day_window", options, TimeWindowFeatureGroup.TIME_UNIT
        )
        assert result == "day"

    def test_named_key_that_is_not_captured_falls_back_to_the_option(self) -> None:
        options = Options(context={"not_a_capture": "fallback"})
        result = TimeWindowFeatureGroup._resolve_operation("x__sum_7_day_window", options, "not_a_capture")
        assert result == "fallback"

    def test_named_key_that_did_not_participate_falls_back_to_the_option(self) -> None:
        options = Options(context={SIZE_KEY: "9"})
        result = OptionalCaptureGroup._resolve_operation("f0__pca_optional_pnb770", options, SIZE_KEY)
        assert result == "9"

    def test_named_key_that_did_not_participate_without_option_is_none(self) -> None:
        result = OptionalCaptureGroup._resolve_operation("f0__pca_optional_pnb770", Options(), SIZE_KEY)
        assert result is None

    @pytest.mark.parametrize("key", ["aggregation_type", "any_other_key"])
    def test_positional_pattern_returns_the_first_capture_for_any_key(self, key: str) -> None:
        assert MockResolverFG._resolve_operation("source__sum_op", Options(), key) == "sum"


class TestResolveOperationIntegration:
    """Integration test: use _resolve_operation in a concrete subclass."""

    def test_string_and_config_resolve_same_value(self) -> None:
        """Both paths should yield the same operation for equivalent inputs."""
        string_options = Options(context={"aggregation_type": "sum"})
        config_options = Options(context={"aggregation_type": "sum"})

        string_result = MockResolverFG._resolve_operation("source__sum_op", string_options, "aggregation_type")
        config_result = MockResolverFG._resolve_operation("my_result", config_options, "aggregation_type")

        assert string_result == config_result == "sum"

    def test_config_returns_string_value(self) -> None:
        """Config-based resolution converts the option value to string."""
        options = Options(context={"aggregation_type": "avg"})
        result = MockResolverFG._resolve_operation("my_result", options, "aggregation_type")
        assert isinstance(result, str)
        assert result == "avg"


class TestResolveOperationFeatureShorthand:
    """Tests for the Feature-based calling convention."""

    def test_feature_shorthand_string_match(self) -> None:
        """Feature shorthand resolves from the feature name pattern."""
        feature = Feature("source__sum_op", options=Options(context={"aggregation_type": "sum"}))
        result = MockResolverFG._resolve_operation(feature, "aggregation_type")
        assert result == "sum"

    def test_feature_shorthand_config_fallback(self) -> None:
        """Feature shorthand falls back to options when pattern does not match."""
        feature = Feature("my_result", options=Options(context={"aggregation_type": "avg"}))
        result = MockResolverFG._resolve_operation(feature, "aggregation_type")
        assert result == "avg"

    def test_feature_shorthand_returns_none(self) -> None:
        """Feature shorthand returns None when neither path resolves."""
        feature = Feature("my_result", options=Options(context={}))
        result = MockResolverFG._resolve_operation(feature, "aggregation_type")
        assert result is None

    def test_feature_shorthand_matches_three_arg_form(self) -> None:
        """Both calling conventions return the same result."""
        feature = Feature("source__sum_op", options=Options(context={"aggregation_type": "sum"}))
        shorthand = MockResolverFG._resolve_operation(feature, "aggregation_type")
        explicit = MockResolverFG._resolve_operation("source__sum_op", feature.options, "aggregation_type")
        assert shorthand == explicit == "sum"


class ResolveOperationTestDataCreator(ATestDataCreator):
    """Test data creator for _resolve_operation integration tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {"Sales": [100, 200, 300, 400, 500]}


class TestResolveOperationRunAll:
    """End-to-end test: _resolve_operation works through mloda.run_all()."""

    def test_string_based_feature_via_run_all(self) -> None:
        """String-based aggregation feature resolves correctly through the engine."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {ResolveOperationTestDataCreator, PandasAggregatedFeatureGroup}
        )

        results = mloda.run_all(
            ["Sales__sum_aggr"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1
        assert "Sales__sum_aggr" in results[0].columns
        assert results[0]["Sales__sum_aggr"].iloc[0] == 1500

    def test_config_based_feature_via_run_all(self) -> None:
        """Config-based aggregation feature resolves correctly through the engine."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {ResolveOperationTestDataCreator, PandasAggregatedFeatureGroup}
        )

        feature = Feature(
            "total_sales",
            Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
                    DefaultOptionKeys.in_features: "Sales",
                }
            ),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1
        assert "total_sales" in results[0].columns
        assert results[0]["total_sales"].iloc[0] == 1500
