"""
Unit tests for group/context options separation in feature configuration.

This module tests the new Options architecture that separates group_options
and context_options for performance optimization.
"""

import pytest

from mloda.core.api.feature_config.models import FeatureConfig
from mloda.core.api.feature_config.parser import parse_json


def test_parse_group_options() -> None:
    """Test parsing JSON with group_options field instead of simple options.

    The new Options architecture separates group_options and context_options
    for performance optimization. This test verifies that parse_json correctly
    handles the group_options field.

    Example:
        {
            "name": "production_feature",
            "group_options": {
                "data_source": "production"
            }
        }
    """
    config_str = """[
        {
            "name": "production_feature",
            "group_options": {
                "data_source": "production"
            }
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "production_feature"
    assert result[0].group_options == {"data_source": "production"}
    assert result[0].options == {}


def test_parse_context_options() -> None:
    """Test parsing JSON with context_options field.

    The new Options architecture separates group_options and context_options
    for performance optimization. This test verifies that parse_json correctly
    handles the context_options field.

    Example:
        {
            "name": "cached_feature",
            "context_options": {
                "cache_enabled": true
            }
        }
    """
    config_str = """[
        {
            "name": "cached_feature",
            "context_options": {
                "cache_enabled": true
            }
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "cached_feature"
    assert result[0].context_options == {"cache_enabled": True}
    assert result[0].options == {}


def test_parse_group_and_context_together() -> None:
    """Test parsing JSON with both group_options and context_options fields.

    The new Options architecture separates group_options and context_options.
    This test verifies that parse_json correctly handles both fields when
    present in the same feature configuration.

    Example:
        {
            "name": "advanced_feature",
            "group_options": {
                "data_source": "production"
            },
            "context_options": {
                "cache_enabled": true
            }
        }
    """
    config_str = """[
        {
            "name": "advanced_feature",
            "group_options": {
                "data_source": "production"
            },
            "context_options": {
                "cache_enabled": true
            }
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "advanced_feature"
    assert result[0].group_options == {"data_source": "production"}
    assert result[0].context_options == {"cache_enabled": True}
    assert result[0].options == {}


def test_load_creates_options_with_group_context() -> None:
    """Test that the loader creates proper Options objects with group/context separation.

    When a feature config includes both group_options and context_options,
    the loader should create a Feature object with an Options instance that
    properly separates the two types of options.

    The Options object should have:
    - options.group containing the group_options
    - options.context containing the context_options
    """
    from mloda.core.api.feature_config.loader import load_features_from_config
    from mloda.user import Feature

    config_str = """[
        {
            "name": "optimized_feature",
            "group_options": {
                "data_source": "production",
                "threshold": 0.5
            },
            "context_options": {
                "cache_enabled": true,
                "debug_mode": false
            }
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "optimized_feature"

    # Verify Options object has correct group options
    assert result[0].options.group == {"data_source": "production", "threshold": 0.5}

    # Verify Options object has correct context options
    assert result[0].options.context == {"cache_enabled": True, "debug_mode": False}


def test_validate_options_mutual_exclusion() -> None:
    """Test that options and group_options/context_options are mutually exclusive.

    The Options architecture uses either:
    - The simple `options` field (simple dict), OR
    - The `group_options` and/or `context_options` fields

    Users should NOT mix both formats together. This test verifies that
    FeatureConfig raises a ValueError when both formats are used.

    Invalid examples that should raise ValueError:
        - options + group_options together
        - options + context_options together
        - options + both group_options and context_options
    """
    # Test case 1: options + group_options should fail
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="bad_feature_1", options={"foo": "bar"}, group_options={"baz": "qux"})
    assert "options" in str(exc_info.value).lower()

    # Test case 2: options + context_options should fail
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="bad_feature_2", options={"foo": "bar"}, context_options={"baz": "qux"})
    assert "options" in str(exc_info.value).lower()

    # Test case 3: options + both group_options and context_options should fail
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(
            name="bad_feature_3",
            options={"foo": "bar"},
            group_options={"baz": "qux"},
            context_options={"alpha": "beta"},
        )
    assert "options" in str(exc_info.value).lower()
