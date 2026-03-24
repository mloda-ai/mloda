"""Tests for PROPERTY_MAPPING type constraint support."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


def _is_list_of_strings(value: Any) -> bool:
    """Validator: value must be a list of strings."""
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) for item in value)


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
        result = MockWithTypeConstraint.match_feature_group_criteria(
            "source__sum_typed", options
        )
        assert result is True

    def test_string_match_rejects_invalid_type(self) -> None:
        """String-based match also rejects when type_validator fails."""
        options = Options(context={"operation": "sum", "partition_by": "not_a_list"})
        result = MockWithTypeConstraint.match_feature_group_criteria(
            "source__sum_typed", options
        )
        assert result is False
