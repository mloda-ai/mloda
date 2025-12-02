"""
Unit tests for chained feature configuration support.

This module tests parsing and handling of chained features using the
operation__source_feature pattern (e.g., "scale__age").
"""

import pytest

from mloda_plugins.config.feature.models import FeatureConfig
from mloda_plugins.config.feature.parser import parse_json


def test_parse_simple_chained_feature() -> None:
    """Test parsing JSON with simple chained feature string like 'age__scale'.

    Chained features follow the pattern: source_feature__operation
    This test verifies that the parser recognizes the chained pattern and
    extracts the mloda_sources field correctly.
    """
    config_str = """[
        {
            "name": "age__scale",
            "mloda_sources": ["age"]
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "age__scale"
    assert result[0].mloda_sources == ["age"]


def test_parse_multi_level_chained_feature() -> None:
    """Test parsing JSON with multi-level chained feature.

    Multi-level chained features follow the pattern: source_feature__op2__op1
    (e.g., "age__mean_imputed__standard_scaled").
    This test verifies that the parser correctly extracts the mloda_sources
    from deeply nested chained operations.
    """
    config_str = """[
        {
            "name": "age__mean_imputed__standard_scaled",
            "mloda_sources": ["age"]
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "age__mean_imputed__standard_scaled"
    assert result[0].mloda_sources == ["age"]


def test_load_chained_feature_as_string() -> None:
    """Test loading a chained feature configuration into a Feature object.

    When loading a chained feature config (e.g., "age__scale" with mloda_sources=["age"]),
    the loader should create a Feature object with in_features added to the
    options context, enabling the mloda runtime to properly resolve the dependency.

    This test verifies that:
    1. The loader processes FeatureConfig.mloda_sources field
    2. The resulting Feature has in_features in its options
    3. The mloda_sources value is correctly passed as a frozenset
    """
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_plugins.config.feature.loader import load_features_from_config
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    config_str = """[
        {
            "name": "age__scale",
            "mloda_sources": ["age"],
            "options": {"param": "value"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)

    feature = result[0]
    assert feature.name.name == "age__scale"

    # The mloda_sources should be added to options as in_features
    # It should be in the context section, not group, as a frozenset
    mloda_sources_value = feature.options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset)
    assert mloda_sources_value == frozenset({"age"})

    # Original options should still be preserved in group
    assert feature.options.group.get("param") == "value"


def test_load_chained_feature_from_config() -> None:
    """Test loading a complete config with mixed feature types including chained features.

    This test validates that the loader correctly handles a realistic JSON configuration
    containing:
    1. Simple string features (e.g., "age")
    2. Regular features with options (e.g., {"name": "weight", "options": {...}})
    3. Chained features with mloda_sources (e.g., {"name": "age__scale", "mloda_sources": ["age"]})

    The test verifies that:
    - All feature types are loaded correctly
    - Chained features have in_features in options.context
    - Regular features preserve their options in options.group
    - Simple string features remain as strings
    """
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_plugins.config.feature.loader import load_features_from_config
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    config_str = """[
        "age",
        {"name": "weight", "options": {"unit": "kg", "precision": 2}},
        {
            "name": "age__scale",
            "mloda_sources": ["age"],
            "options": {"method": "standard"}
        }
    ]"""

    result = load_features_from_config(config_str)

    # Verify we got 3 features
    assert len(result) == 3

    # First feature should be a simple string
    assert isinstance(result[0], str)
    assert result[0] == "age"

    # Second feature should be a regular Feature with options
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "weight"
    assert result[1].options.group.get("unit") == "kg"
    assert result[1].options.group.get("precision") == 2
    # Should NOT have in_features
    assert DefaultOptionKeys.in_features not in result[1].options.context

    # Third feature should be a chained Feature with mloda_sources in context as frozenset
    assert isinstance(result[2], Feature)
    assert result[2].name.name == "age__scale"
    mloda_sources_value = result[2].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset)
    assert mloda_sources_value == frozenset({"age"})
    # Original options should be preserved in group
    assert result[2].options.group.get("method") == "standard"
