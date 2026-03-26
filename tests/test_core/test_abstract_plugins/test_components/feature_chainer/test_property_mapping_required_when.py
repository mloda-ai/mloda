"""Tests for required_when conditional option support in PROPERTY_MAPPING."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pandas as pd
import pytest

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
        feature_name = features.get_name_of_one_feature().name
        agg_type = features.get_options_key("aggregation_type")
        return pd.DataFrame({feature_name: [f"computed_{agg_type}"]})

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
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
        """mloda.run_all raises when first is used without order_by (predicate True, option absent)."""
        plugin_collector = PluginCollector.enabled_feature_groups({ConditionalRequiredFeatureGroup})
        feature = Feature(
            "result_feature",
            Options(context={"aggregation_type": "first"}),
        )
        with pytest.raises(Exception):
            mloda.run_all(
                features=[feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=plugin_collector,
            )
