"""
Unit tests for nested feature references in configuration.

This module tests the support for referencing other features as sources
using the `@feature_name` syntax in the mloda_source field.
"""

from mloda_plugins.config.feature.models import FeatureConfig
from mloda_plugins.config.feature.parser import parse_json


def test_parse_feature_with_reference() -> None:
    """Test parsing JSON with @reference syntax in mloda_sources field.

    This test verifies that parse_json correctly handles JSON configurations
    where the mloda_sources field contains a reference to another feature
    using the `@feature_name` syntax (e.g., ["@scaled_age"]).

    The parser should preserve the @ prefix so that the loader can later
    resolve these references to actual Feature objects.
    """
    config_str = """[
        {
            "name": "scaled_age",
            "mloda_sources": ["age"],
            "options": {"method": "standard"}
        },
        {
            "name": "derived_from_scaled",
            "mloda_sources": ["@scaled_age"],
            "options": {"transformation": "log"}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 2

    # First feature: regular chained feature
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "scaled_age"
    assert result[0].mloda_sources == ["age"]
    assert result[0].options == {"method": "standard"}

    # Second feature: feature with reference to another feature
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "derived_from_scaled"
    assert result[1].mloda_sources == ["@scaled_age"]
    assert result[1].options == {"transformation": "log"}
