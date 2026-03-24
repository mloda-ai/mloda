"""Tests for FeatureChainParserMixin._resolve_operation helper."""

from __future__ import annotations

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class MockResolverFG(FeatureChainParserMixin):
    """Mock feature group for testing _resolve_operation."""

    PREFIX_PATTERN = r".*__([\w]+)_op$"
    AGGREGATION_TYPE = "aggregation_type"

    PROPERTY_MAPPING = {
        "aggregation_type": {
            "sum": "Sum",
            "avg": "Average",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
    }


class TestResolveOperationUnit:
    """Unit tests for _resolve_operation."""

    def test_returns_parsed_operation_from_string(self) -> None:
        """When feature name matches PREFIX_PATTERN, returns parsed operation."""
        options = Options(context={"aggregation_type": "sum"})
        result = MockResolverFG._resolve_operation("source__sum_op", options, "aggregation_type")
        assert result == "sum"

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
        from mloda.core.abstract_plugins.components.feature_name import FeatureName

        options = Options(context={"aggregation_type": "sum"})
        result = MockResolverFG._resolve_operation(
            FeatureName("source__sum_op"), options, "aggregation_type"
        )
        assert result == "sum"


class TestResolveOperationIntegration:
    """Integration test: use _resolve_operation in a concrete subclass."""

    def test_string_and_config_resolve_same_value(self) -> None:
        """Both paths should yield the same operation for equivalent inputs."""
        string_options = Options(context={"aggregation_type": "sum"})
        config_options = Options(context={"aggregation_type": "sum"})

        string_result = MockResolverFG._resolve_operation(
            "source__sum_op", string_options, "aggregation_type"
        )
        config_result = MockResolverFG._resolve_operation(
            "my_result", config_options, "aggregation_type"
        )

        assert string_result == config_result == "sum"

    def test_config_returns_string_value(self) -> None:
        """Config-based resolution converts the option value to string."""
        options = Options(context={"aggregation_type": "avg"})
        result = MockResolverFG._resolve_operation("my_result", options, "aggregation_type")
        assert isinstance(result, str)
        assert result == "avg"
