"""
Unit tests for FeatureConfig model.

This module tests the validation and behavior of the FeatureConfig model
defined in mloda.core.api.feature_config.models.
"""

from typing import Any

import pytest

from mloda.core.api.feature_config.models import FeatureConfig


def test_valid_feature_config_with_name_only() -> None:
    """Test creating a valid FeatureConfig with only a name."""
    config = FeatureConfig(name="test_feature")

    assert config.name == "test_feature"
    assert config.options == {}


def test_valid_feature_config_with_options() -> None:
    """Test creating a valid FeatureConfig with name and options."""
    options = {"enabled": True, "threshold": 0.75}
    config = FeatureConfig(name="test_feature", options=options)

    assert config.name == "test_feature"
    assert config.options == options


def test_feature_config_options_default_empty() -> None:
    """Test that options defaults to an empty dict when not provided."""
    config = FeatureConfig(name="default_test")

    assert config.options == {}
    assert isinstance(config.options, dict)


def test_feature_config_rejects_empty_name() -> None:
    """Test that an empty name is rejected (shared invariant for the top-level and nested paths)."""
    with pytest.raises(ValueError, match="name"):
        FeatureConfig(name="")


def test_feature_config_rejects_whitespace_only_name() -> None:
    """Test that a whitespace-only name strips to nothing and is rejected."""
    with pytest.raises(ValueError, match="name"):
        FeatureConfig(name="   ")


def test_feature_config_rejects_non_string_name() -> None:
    """Test that a non-string name is rejected: 'name' is a non-empty string."""
    bad_value: Any = 123

    with pytest.raises(ValueError, match="name"):
        FeatureConfig(name=bad_value)
