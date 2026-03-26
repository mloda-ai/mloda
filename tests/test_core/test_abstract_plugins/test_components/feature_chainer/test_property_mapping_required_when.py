"""Tests for required_when conditional option support in PROPERTY_MAPPING."""

from __future__ import annotations

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


_ORDER_DEPENDENT = {"first", "last"}


def _needs_order_by(options: Options) -> bool:
    """Predicate: order_by is required when aggregation_type is first or last."""
    agg_type = options.get("aggregation_type")
    return agg_type in _ORDER_DEPENDENT


class MockWithConditionalRequired(FeatureChainParserMixin):
    """Feature group with a conditionally required order_by in PROPERTY_MAPPING."""

    PREFIX_PATTERN = r".*__([\w]+)_windowed$"

    PROPERTY_MAPPING = {
        "aggregation_type": {
            "sum": "Sum of values",
            "avg": "Average of values",
            "first": "First value (requires order_by)",
            "last": "Last value (requires order_by)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "order_by": {
            "explanation": "Column to order by within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.required_when: _needs_order_by,
        },
    }


class TestRequiredWhenUnit:
    """Unit tests for required_when predicate in PROPERTY_MAPPING."""

    def test_rejects_when_predicate_true_and_option_absent(self) -> None:
        """When predicate returns True and the option is absent, matching fails."""
        options = Options(context={"aggregation_type": "first"})
        result = MockWithConditionalRequired.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_accepts_when_predicate_true_and_option_present(self) -> None:
        """When predicate returns True and the option is present, matching succeeds."""
        options = Options(context={"aggregation_type": "first", "order_by": "timestamp"})
        result = MockWithConditionalRequired.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_accepts_when_predicate_false_and_option_absent(self) -> None:
        """When predicate returns False, the option is not required."""
        options = Options(context={"aggregation_type": "sum"})
        result = MockWithConditionalRequired.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_accepts_when_predicate_false_and_option_present(self) -> None:
        """When predicate returns False, having the option is fine too."""
        options = Options(context={"aggregation_type": "sum", "order_by": "timestamp"})
        result = MockWithConditionalRequired.match_feature_group_criteria("my_feature", options)
        assert result is True


class TestRequiredWhenIntegration:
    """Integration tests: order_by required for first/last but not sum/avg."""

    def test_first_without_order_by_rejected(self) -> None:
        options = Options(context={"aggregation_type": "first"})
        assert MockWithConditionalRequired.match_feature_group_criteria("my_feat", options) is False

    def test_last_without_order_by_rejected(self) -> None:
        options = Options(context={"aggregation_type": "last"})
        assert MockWithConditionalRequired.match_feature_group_criteria("my_feat", options) is False

    def test_first_with_order_by_accepted(self) -> None:
        options = Options(context={"aggregation_type": "first", "order_by": "ts"})
        assert MockWithConditionalRequired.match_feature_group_criteria("my_feat", options) is True

    def test_last_with_order_by_accepted(self) -> None:
        options = Options(context={"aggregation_type": "last", "order_by": "ts"})
        assert MockWithConditionalRequired.match_feature_group_criteria("my_feat", options) is True

    def test_sum_without_order_by_accepted(self) -> None:
        options = Options(context={"aggregation_type": "sum"})
        assert MockWithConditionalRequired.match_feature_group_criteria("my_feat", options) is True

    def test_avg_without_order_by_accepted(self) -> None:
        options = Options(context={"aggregation_type": "avg"})
        assert MockWithConditionalRequired.match_feature_group_criteria("my_feat", options) is True

    def test_string_match_first_without_order_by_rejected(self) -> None:
        """String-based matching also enforces required_when."""
        options = Options(context={"aggregation_type": "first"})
        result = MockWithConditionalRequired.match_feature_group_criteria(
            "source__first_windowed", options
        )
        assert result is False

    def test_string_match_first_with_order_by_accepted(self) -> None:
        """String-based matching passes when required_when is satisfied."""
        options = Options(context={"aggregation_type": "first", "order_by": "ts"})
        result = MockWithConditionalRequired.match_feature_group_criteria(
            "source__first_windowed", options
        )
        assert result is True
