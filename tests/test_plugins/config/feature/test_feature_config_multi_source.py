"""
Unit tests for multi-source feature configuration.

This module tests the mloda_sources field which allows features to specify
multiple source features (e.g., distance calculations requiring latitude and longitude).
"""

import pytest
from pydantic import ValidationError

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.config.feature.loader import load_features_from_config
from mloda_plugins.config.feature.models import FeatureConfig
from mloda_plugins.config.feature.parser import parse_json
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


def test_parse_feature_with_multiple_sources() -> None:
    """Test parsing JSON with mloda_sources array field for multi-source features.

    Features that require multiple input sources (e.g., distance calculation
    from latitude and longitude) should use the mloda_sources field with an
    array of source feature names.

    This test verifies that parse_json correctly handles and preserves the
    mloda_sources array field in FeatureConfig objects.

    Example:
        {
            "name": "distance_from_center",
            "mloda_sources": ["latitude", "longitude"]
        }
    """
    config_str = """[
        {
            "name": "distance_from_center",
            "mloda_sources": ["latitude", "longitude"]
        },
        {
            "name": "area_calculation",
            "mloda_sources": ["width", "height"],
            "options": {"unit": "square_meters"}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 2

    # First multi-source feature
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "distance_from_center"
    assert result[0].mloda_sources == ["latitude", "longitude"]
    assert result[0].options == {}

    # Second multi-source feature with options
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "area_calculation"
    assert result[1].mloda_sources == ["width", "height"]
    assert result[1].options == {"unit": "square_meters"}


def test_load_multiple_sources_as_frozenset() -> None:
    """Test that loader converts mloda_sources array to frozenset in options.

    When a feature config includes mloda_sources array (e.g., ["latitude", "longitude"]),
    the loader should:
    1. Convert the array to a frozenset for immutability and performance
    2. Store the frozenset in options.context[DefaultOptionKeys.mloda_source_features]
    3. Preserve any other options in the appropriate location

    The use of frozenset ensures:
    - Immutability (cannot be modified after creation)
    - Hashability (can be used as dict keys or in sets)
    - Efficient membership testing

    Example:
        Input: {"name": "distance", "mloda_sources": ["lat", "lon"]}
        Output: Feature.options.context[mloda_source_features] = frozenset({"lat", "lon"})
    """
    config_str = """[
        {
            "name": "distance_from_center",
            "mloda_sources": ["latitude", "longitude"],
            "options": {"method": "haversine"}
        },
        {
            "name": "area_calculation",
            "mloda_sources": ["width", "height"],
            "group_options": {"unit": "square_meters"},
            "context_options": {"precision": "high"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2

    # First feature: mloda_sources with legacy options
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "distance_from_center"

    # mloda_sources should be converted to frozenset and stored in context
    # Note: Using a string key since DefaultOptionKeys.mloda_source_features doesn't exist yet
    mloda_sources = result[0].options.context.get("mloda_source_features")
    assert isinstance(mloda_sources, frozenset)
    assert mloda_sources == frozenset({"latitude", "longitude"})

    # Regular options should be in group
    assert result[0].options.group.get("method") == "haversine"

    # Second feature: mloda_sources with group_options and context_options
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "area_calculation"

    # mloda_sources should be converted to frozenset and stored in context
    mloda_sources_2 = result[1].options.context.get("mloda_source_features")
    assert isinstance(mloda_sources_2, frozenset)
    assert mloda_sources_2 == frozenset({"width", "height"})

    # Group options should be in group
    assert result[1].options.group.get("unit") == "square_meters"

    # Context options should be merged into context (along with mloda_sources)
    assert result[1].options.context.get("precision") == "high"


def test_validate_source_vs_sources_mutual_exclusion() -> None:
    """Test that mloda_source and mloda_sources are mutually exclusive.

    A feature configuration should use either:
    - mloda_source (single source, for simple transformations)
    - mloda_sources (multiple sources, for multi-input features)

    Using both fields together is invalid and should raise a ValidationError.

    Invalid example that should fail:
        {
            "name": "bad_feature",
            "mloda_source": "age",           # Single source
            "mloda_sources": ["lat", "lon"]  # Multiple sources - CONFLICT!
        }
    """
    # Test case: both mloda_source and mloda_sources should fail validation
    with pytest.raises(ValidationError) as exc_info:
        FeatureConfig(name="bad_feature", mloda_source="age", mloda_sources=["latitude", "longitude"])
    assert "mloda_source" in str(exc_info.value).lower()

    # Verify that using only mloda_source works fine
    config_single = FeatureConfig(name="single_source", mloda_source="age")
    assert config_single.mloda_source == "age"
    assert config_single.mloda_sources is None

    # Verify that using only mloda_sources works fine
    config_multi = FeatureConfig(name="multi_source", mloda_sources=["latitude", "longitude"])
    assert config_multi.mloda_sources == ["latitude", "longitude"]
    assert config_multi.mloda_source is None
