"""Tests for PROPERTY_MAPPING type constraint support."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import ComputeFramework, FeatureGroup
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.provider import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


def _is_list_of_strings(value: Any) -> bool:
    """Validator: value must be a list of strings."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) for item in value)


def _raising_validator(value: Any) -> bool:
    """Validator that raises instead of returning False for invalid input."""
    if not isinstance(value, list):
        raise TypeError(f"Expected list, got {type(value).__name__}")
    return True


def _max_length_3(value: Any) -> bool:
    """Validator that rejects strings longer than 3 characters."""
    return isinstance(value, str) and len(value) <= 3


class MockWithTypeConstraint(FeatureChainParserMixin):
    """Feature group with a type-constrained PROPERTY_MAPPING entry."""

    PREFIX_PATTERN = r".*__([\w]+)_typed$"

    PARTITION_BY = "partition_by"

    PROPERTY_MAPPING = {
        "operation": {
            "sum": "Sum of values",
            "avg": "Average of values",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "partition_by": {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.type_validator: _is_list_of_strings,
        },
    }


class MockStrictWithTypeValidator(FeatureChainParserMixin):
    """Feature group combining strict_validation=True with type_validator on the same entry."""

    PREFIX_PATTERN = r".*__([\w]+)_strict$"

    PROPERTY_MAPPING = {
        "mode": {
            "fast": "Fast mode",
            "slow": "Slow mode",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.type_validator: lambda v: isinstance(v, str),
        },
    }


class MockStrictWithOrthogonalTypeValidator(FeatureChainParserMixin):
    """Feature group where type_validator rejects values that pass strict membership.

    The mapping allows "fast" and "slow", but the type_validator requires
    strings of at most 3 characters. "fast" (4 chars) and "slow" (4 chars) both
    pass strict membership but fail the type_validator length check.
    """

    PREFIX_PATTERN = r".*__([\w]+)_ortho$"

    PROPERTY_MAPPING = {
        "mode": {
            "fast": "Fast mode",
            "slow": "Slow mode",
            "ok": "OK mode",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.type_validator: _max_length_3,
        },
    }


class MockWithRaisingValidator(FeatureChainParserMixin):
    """Feature group whose type_validator raises on invalid input."""

    PREFIX_PATTERN = r".*__([\w]+)_raise$"

    PROPERTY_MAPPING = {
        "items": {
            "explanation": "A list of items",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.type_validator: _raising_validator,
        },
    }


class MockWithoutTypeConstraint(FeatureChainParserMixin):
    """Feature group without any type constraints (baseline)."""

    PREFIX_PATTERN = r".*__([\w]+)_typed$"

    PROPERTY_MAPPING = {
        "operation": {
            "sum": "Sum of values",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
    }


class TestTypeConstraintValidation:
    """Tests for type_validator in PROPERTY_MAPPING."""

    def test_rejects_invalid_type(self) -> None:
        """match_feature_group_criteria returns False when type_validator fails."""
        options = Options(context={"operation": "sum", "partition_by": "not_a_list"})
        result = MockWithTypeConstraint.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_accepts_valid_type(self) -> None:
        """match_feature_group_criteria returns True when type_validator passes."""
        options = Options(context={"operation": "sum", "partition_by": ["region", "category"]})
        result = MockWithTypeConstraint.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_rejects_list_of_non_strings(self) -> None:
        """Validator should reject a list containing non-string items."""
        options = Options(context={"operation": "sum", "partition_by": [1, 2, 3]})
        result = MockWithTypeConstraint.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_accepts_empty_list(self) -> None:
        """Validator should accept an empty list (all() returns True for empty)."""
        options = Options(context={"operation": "sum", "partition_by": []})
        result = MockWithTypeConstraint.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_no_type_constraint_unaffected(self) -> None:
        """Entries without type_validator are unaffected."""
        options = Options(context={"operation": "sum"})
        result = MockWithoutTypeConstraint.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_missing_option_with_type_constraint_fails_property_mapping(self) -> None:
        """When a PROPERTY_MAPPING entry is absent from options, matching fails."""
        options = Options(context={"operation": "sum"})
        result = MockWithTypeConstraint.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_string_match_with_type_constraint(self) -> None:
        """String-based feature matching works alongside type constraints."""
        options = Options(context={"operation": "sum", "partition_by": ["region"]})
        result = MockWithTypeConstraint.match_feature_group_criteria("source__sum_typed", options)
        assert result is True

    def test_string_match_rejects_invalid_type(self) -> None:
        """String-based match also rejects when type_validator fails."""
        options = Options(context={"operation": "sum", "partition_by": "not_a_list"})
        result = MockWithTypeConstraint.match_feature_group_criteria("source__sum_typed", options)
        assert result is False

    def test_strict_validation_with_type_validator_accepts_valid(self) -> None:
        """Both strict membership and type_validator pass for a valid mapping value."""
        options = Options(context={"mode": "fast"})
        result = MockStrictWithTypeValidator.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_strict_validation_with_type_validator_rejects_non_string(self) -> None:
        """strict_validation rejects a non-string before type_validator runs."""
        options = Options(context={"mode": 123})
        result = MockStrictWithTypeValidator.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_strict_validation_with_type_validator_rejects_unknown_value(self) -> None:
        """strict_validation rejects a value not in the mapping, even if type_validator passes."""
        options = Options(context={"mode": "turbo"})
        result = MockStrictWithTypeValidator.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_type_validator_rejects_value_that_passes_strict_membership(self) -> None:
        """type_validator can reject a value that passes strict membership.

        "fast" is in the mapping (passes membership) but has 4 characters
        (fails the max-length-3 type_validator). This proves type_validator
        runs independently of strict validation.
        """
        options = Options(context={"mode": "fast"})
        result = MockStrictWithOrthogonalTypeValidator.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_type_validator_accepts_value_that_passes_both(self) -> None:
        """A value that passes both strict membership and type_validator is accepted.

        "ok" is in the mapping (2 chars <= 3), so both checks pass.
        """
        options = Options(context={"mode": "ok"})
        result = MockStrictWithOrthogonalTypeValidator.match_feature_group_criteria("my_feature", options)
        assert result is True

    def test_raising_validator_treated_as_non_match(self) -> None:
        """A validator that raises an exception is treated as a non-match (returns False)."""
        options = Options(context={"items": "not_a_list"})
        result = MockWithRaisingValidator.match_feature_group_criteria("my_feature", options)
        assert result is False

    def test_raising_validator_accepts_valid_input(self) -> None:
        """A raising validator that returns True for valid input works normally."""
        options = Options(context={"items": ["a", "b"]})
        result = MockWithRaisingValidator.match_feature_group_criteria("my_feature", options)
        assert result is True


class TypeValidatorTestDataCreator(ATestDataCreator):
    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {
            "Sales": [100, 200, 300, 400, 500],
            "region": ["A", "A", "B", "B", "A"],
        }


class TypeValidatedAggregation(FeatureChainParserMixin, FeatureGroup):
    """Feature group that uses type_validator to enforce partition_by is list[str]."""

    PREFIX_PATTERN = r".*__([\w]+)_tvaggr$"
    AGGREGATION_TYPE = "aggregation_type"
    PARTITION_BY = "partition_by"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        "aggregation_type": {
            "sum": "Sum of values",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        "partition_by": {
            "explanation": "List of columns to partition by",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.type_validator: _is_list_of_strings,
        },
    }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        table = data
        for feature in features.features:
            agg_type = str(feature.options.get(cls.AGGREGATION_TYPE))
            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
            partition_by = feature.options.get(cls.PARTITION_BY)

            if agg_type == "sum":
                result = table.groupby(partition_by, dropna=False)[source_col].transform("sum")
                table[feature.name] = result
        return table

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class TestTypeConstraintIntegrationRunAll:
    """End-to-end test: type_validator works through mloda.run_all()."""

    def test_valid_type_runs_successfully(self) -> None:
        """Feature with valid partition_by type runs through the full pipeline."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {TypeValidatorTestDataCreator, TypeValidatedAggregation}
        )

        feature = Feature(
            "sales_sum",
            Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "Sales",
                    "partition_by": ["region"],
                }
            ),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1
        assert "sales_sum" in results[0].columns

    def test_invalid_type_raises_value_error(self) -> None:
        """Feature with invalid partition_by type is rejected and raises ValueError."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {TypeValidatorTestDataCreator, TypeValidatedAggregation}
        )

        feature = Feature(
            "sales_sum",
            Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "Sales",
                    "partition_by": "region",
                }
            ),
        )

        with pytest.raises(ValueError, match="No feature groups found"):
            mloda.run_all(
                [feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=plugin_collector,
            )
