"""
Tests for feature configuration schema export utility.

This module tests the schema export function that provides JSON Schema
for the FeatureConfig model.
"""

import pytest

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


def test_feature_config_schema_forbids_additional_properties() -> None:
    """Test that the schema forbids unknown keys (issue #680).

    FeatureConfig enforces this at the top level and, once the nested in_features
    dict is validated through FeatureConfig too, at every nesting level.
    """
    schema = feature_config_schema()

    assert schema["additionalProperties"] is False, "Unknown config keys must not be allowed"


@pytest.mark.parametrize("field", ["propagate_context_keys", "cascade_context_keys"])
def test_feature_config_schema_includes_context_push_keys(field: str) -> None:
    """The schema advertises each context push key field as an array of strings."""
    schema = feature_config_schema()

    assert field in schema["properties"], f"'{field}' should be in schema properties"

    prop = schema["properties"][field]
    assert prop["type"] == "array", f"'{field}' should be an array"
    assert prop["items"] == {"type": "string"}, f"'{field}' items should be strings"


def test_feature_config_schema_includes_keep_index_and_index() -> None:
    """The schema advertises keep_index as a boolean and index as a non-empty array of non-empty strings."""
    properties = feature_config_schema()["properties"]

    assert properties["keep_index"] == {"type": "boolean"}
    assert properties["index"] == {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
        "uniqueItems": True,
    }


def test_feature_config_schema_index_items_are_unique() -> None:
    """index entries must be unique."""
    assert feature_config_schema()["properties"]["index"]["uniqueItems"] is True
