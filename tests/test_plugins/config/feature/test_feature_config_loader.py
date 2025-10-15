"""
Unit tests for feature configuration loader.

This module tests the load_features_from_config function from mloda_plugins.config.feature.loader.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.config.feature.loader import load_features_from_config


def test_load_features_string_only() -> None:
    """Test loading config with only string features."""
    config_str = '["feature1", "feature2", "feature3"]'

    result = load_features_from_config(config_str)

    assert len(result) == 3
    assert result[0] == "feature1"
    assert result[1] == "feature2"
    assert result[2] == "feature3"
    assert all(isinstance(item, str) for item in result)


def test_load_features_with_objects() -> None:
    """Test loading config with FeatureConfig objects."""
    config_str = """[
        {"name": "feature1", "options": {"enabled": true}},
        {"name": "feature2", "options": {"threshold": 0.8}}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2
    assert isinstance(result[0], Feature)
    assert isinstance(result[1], Feature)
    assert result[0].name.name == "feature1"
    assert result[0].options.get("enabled") == True
    assert result[1].name.name == "feature2"
    assert result[1].options.get("threshold") == 0.8


def test_load_features_mixed() -> None:
    """Test loading config with both strings and objects."""
    config_str = """[
        "simple_feature",
        {"name": "complex_feature", "options": {"param": "value"}},
        "another_simple"
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], Feature)
    assert isinstance(result[2], str)
    assert result[0] == "simple_feature"
    assert result[1].name.name == "complex_feature"
    assert result[1].options.get("param") == "value"
    assert result[2] == "another_simple"


def test_load_features_unsupported_format() -> None:
    """Test that ValueError is raised for unsupported format."""
    config_str = '["feature1"]'

    with pytest.raises(ValueError, match="Unsupported format: yaml"):
        load_features_from_config(config_str, format="yaml")


def test_load_features_preserves_options() -> None:
    """Test that options dict is correctly passed to Feature objects."""
    config_str = """[
        {"name": "feature_with_options", "options": {"key1": "value1", "key2": 42, "key3": true}}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "feature_with_options"
    assert result[0].options.get("key1") == "value1"
    assert result[0].options.get("key2") == 42
    assert result[0].options.get("key3") == True
