"""Tests for required_when conditional option support in PROPERTY_MAPPING."""

from __future__ import annotations

from typing import Any, Optional, Set

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.provider import DefaultOptionKeys


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

    def test_extract_property_values_strips_required_when(self) -> None:
        """required_when callable must be stripped from extracted property values."""
        mapping_entry = {
            "explanation": "Column to order by within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.required_when: _needs_order_by,
        }
        extracted = FeatureChainParser._extract_property_values(mapping_entry)
        assert DefaultOptionKeys.required_when not in extracted
        assert "explanation" not in extracted

    def test_extract_property_values_strips_explanation(self) -> None:
        """'explanation' is documentation metadata and must be stripped from extracted property values."""
        mapping_entry = {
            "explanation": "Column to order by within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
        }
        extracted = FeatureChainParser._extract_property_values(mapping_entry)
        assert "explanation" not in extracted

    def test_can_skip_required_check_with_required_when(self) -> None:
        """Properties with required_when should be skippable in the base required check."""
        prop_with_required_when = {
            "explanation": "test",
            DefaultOptionKeys.required_when: _needs_order_by,
        }
        assert FeatureChainParser._can_skip_required_check(prop_with_required_when) is True

    def test_can_skip_required_check_with_default(self) -> None:
        """Properties with default should be skippable in the base required check."""
        prop_with_default = {
            "val1": "desc",
            DefaultOptionKeys.default: "val1",
        }
        assert FeatureChainParser._can_skip_required_check(prop_with_default) is True

    def test_can_skip_required_check_without_either(self) -> None:
        """Properties without default or required_when are required."""
        prop_required = {
            "val1": "desc",
            DefaultOptionKeys.strict_validation: True,
        }
        assert FeatureChainParser._can_skip_required_check(prop_required) is False

    def test_non_callable_required_when_is_skipped(self) -> None:
        """A non-callable required_when value should not crash; the option is treated as optional."""

        class MockWithBadPredicate(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__([\w]+)_windowed$"
            PROPERTY_MAPPING = {
                "aggregation_type": {
                    "sum": "Sum",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                },
                "order_by": {
                    "explanation": "sort column",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.required_when: "not_a_callable",
                },
            }

        options = Options(context={"aggregation_type": "sum"})
        result = MockWithBadPredicate.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_required_when_with_default_value_interaction(self) -> None:
        """When a property has both required_when and default, default takes effect in base parser.

        The base parser treats the property as optional (can_skip_required_check is True).
        The required_when predicate still fires in the mixin layer.
        """

        def _always_required(options: Options) -> bool:
            return True

        class MockWithBothDefaultAndRequiredWhen(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__([\w]+)_windowed$"
            PROPERTY_MAPPING = {
                "aggregation_type": {
                    "sum": "Sum",
                    "first": "First",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                },
                "order_by": {
                    "explanation": "sort column",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: False,
                    DefaultOptionKeys.default: "id",
                    DefaultOptionKeys.required_when: _always_required,
                },
            }

        # The property has a default, but required_when predicate always returns True.
        # Since order_by is absent, the required_when check rejects the match.
        options_absent = Options(context={"aggregation_type": "sum"})
        assert MockWithBothDefaultAndRequiredWhen.match_feature_group_criteria("my_feature", options_absent) is False

        # When order_by is provided, predicate is satisfied.
        options_present = Options(context={"aggregation_type": "sum", "order_by": "ts"})
        assert MockWithBothDefaultAndRequiredWhen.match_feature_group_criteria("my_feature", options_present) is True


class TestRequiredWhenIntegration:
    """Integration tests for string-based and string-only matching with required_when."""

    def test_string_match_first_without_order_by_rejected(self) -> None:
        """String-based matching also enforces required_when."""
        options = Options(context={"aggregation_type": "first"})
        result = MockWithConditionalRequired.match_feature_group_criteria("source__first_windowed", options)
        assert result is False

    def test_string_match_first_with_order_by_accepted(self) -> None:
        """String-based matching passes when required_when is satisfied."""
        options = Options(context={"aggregation_type": "first", "order_by": "ts"})
        result = MockWithConditionalRequired.match_feature_group_criteria("source__first_windowed", options)
        assert result is True

    def test_string_only_first_without_order_by_rejected(self) -> None:
        """Regression: aggregation_type parsed from name only (not in options) still triggers required_when."""
        options = Options(context={})
        result = MockWithConditionalRequired.match_feature_group_criteria("source__first_windowed", options)
        assert result is False

    def test_string_only_first_with_order_by_accepted(self) -> None:
        """Regression: aggregation_type from name + order_by in options satisfies required_when."""
        options = Options(context={"order_by": "ts"})
        result = MockWithConditionalRequired.match_feature_group_criteria("source__first_windowed", options)
        assert result is True

    def test_string_only_sum_without_order_by_accepted(self) -> None:
        """Regression: aggregation_type=sum from name only, no order_by needed."""
        options = Options(context={})
        result = MockWithConditionalRequired.match_feature_group_criteria("source__sum_windowed", options)
        assert result is True

    def test_effective_options_preserve_propagate_context_keys(self) -> None:
        """Effective options built during required_when evaluation must preserve propagate_context_keys."""
        options = Options(
            context={"order_by": "ts"},
            propagate_context_keys=frozenset({"order_by"}),
        )
        effective = MockWithConditionalRequired._build_effective_options(
            "source__first_windowed",
            [MockWithConditionalRequired.PREFIX_PATTERN],
            MockWithConditionalRequired.PROPERTY_MAPPING,
            options,
        )
        assert effective.propagate_context_keys == frozenset({"order_by"})
        assert effective.get("aggregation_type") == "first"


_RUN_ALL_ORDER_DEPENDENT = {"first", "last"}


def _run_all_needs_order_by(options: Options) -> bool:
    """Predicate for the mloda.run_all integration test FeatureGroup."""
    agg_type = options.get("aggregation_type")
    return agg_type in _RUN_ALL_ORDER_DEPENDENT


class ConditionalRequiredFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Full FeatureGroup with required_when for mloda.run_all integration testing."""

    PREFIX_PATTERN = r".*__([\w]+)_windowed$"

    PROPERTY_MAPPING = {
        "aggregation_type": {
            "sum": "Sum of values",
            "first": "First value (requires order_by)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "order_by": {
            "explanation": "Column to order by within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.required_when: _run_all_needs_order_by,
        },
    }

    @classmethod
    def input_data(cls) -> Optional[DataCreator]:
        return DataCreator({"result_feature"})

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Any]]:
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = str(features.get_name_of_one_feature())
        agg_type = features.get_options_key("aggregation_type")
        return pd.DataFrame({feature_name: [f"computed_{agg_type}"]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class TestRequiredWhenRunAll:
    """Integration test: mloda.run_all enforces required_when end-to-end."""

    def test_run_all_accepts_sum_without_order_by(self) -> None:
        """mloda.run_all succeeds for sum (predicate False, order_by not required)."""
        plugin_collector = PluginCollector.enabled_feature_groups({ConditionalRequiredFeatureGroup})
        feature = Feature(
            "result_feature",
            Options(context={"aggregation_type": "sum"}),
        )
        results = mloda.run_all(
            features=[feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )
        assert len(results) == 1
        assert results[0]["result_feature"].iloc[0] == "computed_sum"

    def test_run_all_accepts_first_with_order_by(self) -> None:
        """mloda.run_all succeeds for first when order_by is provided."""
        plugin_collector = PluginCollector.enabled_feature_groups({ConditionalRequiredFeatureGroup})
        feature = Feature(
            "result_feature",
            Options(context={"aggregation_type": "first", "order_by": "timestamp"}),
        )
        results = mloda.run_all(
            features=[feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )
        assert len(results) == 1
        assert results[0]["result_feature"].iloc[0] == "computed_first"

    def test_run_all_rejects_first_without_order_by(self) -> None:
        """mloda.run_all raises ValueError when first is used without order_by."""
        plugin_collector = PluginCollector.enabled_feature_groups({ConditionalRequiredFeatureGroup})
        feature = Feature(
            "result_feature",
            Options(context={"aggregation_type": "first"}),
        )
        with pytest.raises(ValueError, match="No feature groups found"):
            mloda.run_all(
                features=[feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=plugin_collector,
            )
