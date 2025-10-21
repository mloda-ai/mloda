"""
Unit tests for FeatureConfig model.

This module tests the validation and behavior of the FeatureConfig model
defined in mloda_plugins.config.feature.models.
"""

from mloda_plugins.config.feature.models import FeatureConfig


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
