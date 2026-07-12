"""
Tests for feature configuration schema export utility.

This module tests the schema export function that provides JSON Schema
for the FeatureConfig model.
"""

from mloda.core.api.feature_config.models import feature_config_schema


def test_feature_config_schema_structure() -> None:
    """Test that feature_config_schema returns a valid JSON Schema structure."""
    schema = feature_config_schema()

    assert isinstance(schema, dict), "Schema should be a dictionary"

    assert "properties" in schema, "Schema should have 'properties' key"
    assert "type" in schema, "Schema should have 'type' key"
    assert "required" in schema, "Schema should have 'required' key"

    assert "name" in schema["properties"], "'name' should be in schema properties"
    assert "name" in schema["required"], "'name' should be in required fields"


def test_feature_config_schema_includes_propagate_context_keys() -> None:
    """Test that the schema advertises the propagate_context_keys field.

    The schema should expose propagate_context_keys as an array of strings so
    that the documented configuration field is discoverable.
    """
    schema = feature_config_schema()

    assert "propagate_context_keys" in schema["properties"], "'propagate_context_keys' should be in schema properties"

    prop = schema["properties"]["propagate_context_keys"]
    assert prop["type"] == "array", "'propagate_context_keys' should be an array"
    assert prop["items"] == {"type": "string"}, "'propagate_context_keys' items should be strings"


def test_feature_config_schema_includes_feature_group() -> None:
    """Test that the schema advertises the feature_group resolution scope field (issue #582).

    The config scope is the string form only (JSON cannot carry a class object),
    so the schema exposes it as a plain string property.
    """
    schema = feature_config_schema()

    assert "feature_group" in schema["properties"], "'feature_group' should be in schema properties"
    assert schema["properties"]["feature_group"] == {"type": "string"}, "'feature_group' should be a string"
