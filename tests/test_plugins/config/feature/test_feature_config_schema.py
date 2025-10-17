"""
Tests for feature configuration schema export utility.

This module tests the schema export function that provides JSON Schema
for the FeatureConfig model.
"""

import pytest
from mloda_plugins.config.feature.models import feature_config_schema


def test_feature_config_schema_structure() -> None:
    """Test that feature_config_schema returns a valid JSON Schema structure."""
    # Call the schema export function
    schema = feature_config_schema()

    # Verify it returns a dictionary (JSON Schema)
    assert isinstance(schema, dict), "Schema should be a dictionary"

    # Verify required JSON Schema keys are present
    assert "properties" in schema, "Schema should have 'properties' key"
    assert "type" in schema, "Schema should have 'type' key"
    assert "required" in schema, "Schema should have 'required' key"

    # Verify that 'name' field is in properties
    assert "name" in schema["properties"], "'name' should be in schema properties"

    # Verify that 'name' is in the required list
    assert "name" in schema["required"], "'name' should be in required fields"
