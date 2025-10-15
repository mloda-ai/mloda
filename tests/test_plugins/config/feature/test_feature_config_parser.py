"""
Unit tests for feature configuration parser.

This module tests the parse_json function from mloda_plugins.config.feature.parser.
"""

import pytest

from mloda_plugins.config.feature.models import FeatureConfig
from mloda_plugins.config.feature.parser import parse_json


def test_parse_json_with_string_features() -> None:
    """Test parsing JSON array with simple string features."""
    config_str = '["feature1", "feature2", "feature3"]'

    result = parse_json(config_str)

    assert len(result) == 3
    assert result[0] == "feature1"
    assert result[1] == "feature2"
    assert result[2] == "feature3"
    assert all(isinstance(item, str) for item in result)


def test_parse_json_with_object_features() -> None:
    """Test parsing JSON array with feature objects (name + options)."""
    config_str = """[
        {"name": "feature1", "options": {"enabled": true}},
        {"name": "feature2", "options": {"threshold": 0.8}}
    ]"""

    result = parse_json(config_str)

    assert len(result) == 2
    assert isinstance(result[0], FeatureConfig)
    assert isinstance(result[1], FeatureConfig)
    assert result[0].name == "feature1"
    assert result[0].options == {"enabled": True}
    assert result[1].name == "feature2"
    assert result[1].options == {"threshold": 0.8}


def test_parse_json_with_mixed_features() -> None:
    """Test parsing JSON with both strings and objects."""
    config_str = """[
        "simple_feature",
        {"name": "complex_feature", "options": {"param": "value"}},
        "another_simple"
    ]"""

    result = parse_json(config_str)

    assert len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], FeatureConfig)
    assert isinstance(result[2], str)
    assert result[0] == "simple_feature"
    assert result[1].name == "complex_feature"
    assert result[1].options == {"param": "value"}
    assert result[2] == "another_simple"


def test_parse_json_invalid_not_array() -> None:
    """Test that ValueError is raised when input is not a JSON array."""
    config_str = '{"name": "feature1"}'

    with pytest.raises(ValueError, match="Configuration must be a JSON array"):
        parse_json(config_str)


def test_parse_json_invalid_item_type() -> None:
    """Test that ValueError is raised when array contains invalid types."""
    config_str = "[123, 456]"

    with pytest.raises(ValueError, match="Invalid configuration item"):
        parse_json(config_str)
