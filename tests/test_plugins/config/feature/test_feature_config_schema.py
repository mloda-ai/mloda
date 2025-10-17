"""
Tests for feature configuration schema export utility.

This module tests the schema export function that provides JSON Schema
for the FeatureConfig model.
"""

from mloda_plugins.config.feature.models import feature_config_schema


def test_feature_config_schema_structure() -> None:
    """Test that feature_config_schema returns a valid JSON Schema structure."""
    schema = feature_config_schema()

    assert isinstance(schema, dict), "Schema should be a dictionary"

    assert "properties" in schema, "Schema should have 'properties' key"
    assert "type" in schema, "Schema should have 'type' key"
    assert "required" in schema, "Schema should have 'required' key"

    assert "name" in schema["properties"], "'name' should be in schema properties"
    assert "name" in schema["required"], "'name' should be in required fields"
