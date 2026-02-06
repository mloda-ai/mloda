"""Tests that strict_validation ValueError is caught and returns False."""

import pytest

from mloda.user import FeatureName, Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseMode(FeatureChainParserMixin):
    """Base class with strict validation on 'mode' key."""

    PROPERTY_MAPPING = {
        "mode": {
            "mode_a": "Mode A",
            "mode_b": "Mode B",
            DefaultOptionKeys.strict_validation: True,
        }
    }


class SubGroupA(BaseMode):
    """Subclass that only accepts mode_a."""

    PROPERTY_MAPPING = {
        "mode": {
            "mode_a": "Mode A",
            DefaultOptionKeys.strict_validation: True,
        }
    }


class SubGroupB(BaseMode):
    """Subclass that only accepts mode_b."""

    PROPERTY_MAPPING = {
        "mode": {
            "mode_b": "Mode B",
            DefaultOptionKeys.strict_validation: True,
        }
    }


class SubGroupWithValidationFunction(BaseMode):
    """Subclass with a custom validation function."""

    PROPERTY_MAPPING = {
        "mode": {
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.validation_function: lambda v: v.startswith("valid_"),
        }
    }


class TestStrictValidationReturnsFalse:
    """Tests that strict_validation failures return False instead of raising ValueError."""

    def test_returns_false_for_nonmatching_subclass(self) -> None:
        """SubGroupA with mode=mode_b should return False, not raise ValueError."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "mode_b"})

        result = SubGroupA.match_feature_group_criteria(feature_name, options)

        assert result is False

    def test_returns_true_for_matching_subclass(self) -> None:
        """SubGroupA with mode=mode_a should return True."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "mode_a"})

        result = SubGroupA.match_feature_group_criteria(feature_name, options)

        assert result is True

    def test_validation_function_failure_returns_false(self) -> None:
        """Custom validation_function rejection should return False, not raise ValueError."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "invalid_prefix"})

        result = SubGroupWithValidationFunction.match_feature_group_criteria(feature_name, options)

        assert result is False
