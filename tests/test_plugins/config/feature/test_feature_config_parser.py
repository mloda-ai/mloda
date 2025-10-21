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


def test_parse_json_with_mloda_source_field() -> None:
    """Test parsing JSON with mloda_sources field for chained features.

    Chained features use the mloda_sources field to specify the source feature
    that will be transformed. This test verifies that parse_json correctly
    handles and preserves the mloda_sources field in FeatureConfig objects.
    """
    config_str = """[
        {
            "name": "scale__age",
            "mloda_sources": ["age"]
        },
        {
            "name": "standard_scaled__mean_imputed__weight",
            "mloda_sources": ["weight"],
            "options": {"method": "standard"}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 2

    # First chained feature
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "scale__age"
    assert result[0].mloda_sources == ["age"]
    assert result[0].options == {}

    # Second chained feature with options
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "standard_scaled__mean_imputed__weight"
    assert result[1].mloda_sources == ["weight"]
    assert result[1].options == {"method": "standard"}


def test_parse_chained_feature_string() -> None:
    """Test parsing JSON with mixed chained and regular features.

    This test verifies that parse_json can handle a configuration that
    includes both regular features (strings and objects) and chained
    features (objects with mloda_sources field) in the same array.
    """
    config_str = """[
        "simple_feature",
        {
            "name": "scale__age",
            "mloda_sources": ["age"]
        },
        {"name": "regular_feature", "options": {"param": "value"}},
        {
            "name": "normalize__scale__height",
            "mloda_sources": ["height"],
            "options": {"method": "minmax"}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 4

    # First: simple string feature
    assert isinstance(result[0], str)
    assert result[0] == "simple_feature"

    # Second: chained feature
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "scale__age"
    assert result[1].mloda_sources == ["age"]

    # Third: regular feature with options
    assert isinstance(result[2], FeatureConfig)
    assert result[2].name == "regular_feature"
    assert result[2].options == {"param": "value"}
    assert result[2].mloda_sources is None

    # Fourth: multi-level chained feature with options
    assert isinstance(result[3], FeatureConfig)
    assert result[3].name == "normalize__scale__height"
    assert result[3].mloda_sources == ["height"]
    assert result[3].options == {"method": "minmax"}


def test_parse_json_with_group_context_options() -> None:
    """Test parsing JSON with group_options and context_options fields.

    This test verifies that parse_json correctly handles JSON configurations
    with the new group_options and context_options fields, which separate
    group-level and context-level configuration from feature-level options.
    """
    config_str = """[
        {
            "name": "feature1",
            "group_options": {"group_param": "group_value"}
        },
        {
            "name": "feature2",
            "context_options": {"context_param": "context_value"}
        },
        {
            "name": "feature3",
            "group_options": {"group_setting": true},
            "context_options": {"context_setting": 42}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 3

    # First: feature with group_options
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "feature1"
    assert result[0].group_options == {"group_param": "group_value"}
    assert result[0].context_options is None
    assert result[0].options == {}

    # Second: feature with context_options
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "feature2"
    assert result[1].group_options is None
    assert result[1].context_options == {"context_param": "context_value"}
    assert result[1].options == {}

    # Third: feature with both group_options and context_options
    assert isinstance(result[2], FeatureConfig)
    assert result[2].name == "feature3"
    assert result[2].group_options == {"group_setting": True}
    assert result[2].context_options == {"context_setting": 42}
    assert result[2].options == {}


def test_parse_json_rejects_mixed_options_formats() -> None:
    """Test that parse_json raises ValidationError when both options formats are mixed.

    This test verifies that the mutual exclusion validation works during parsing,
    ensuring that features cannot use both the legacy 'options' field and the new
    'group_options'/'context_options' fields simultaneously.
    """
    config_str = """[
        {
            "name": "feature1",
            "options": {"param": "value"},
            "group_options": {"group_param": "group_value"}
        }
    ]"""

    with pytest.raises(ValueError, match="Cannot use both 'options' and 'group_options'/'context_options'"):
        parse_json(config_str)


def test_parse_json_with_column_index() -> None:
    """Test parsing JSON with column_index field for multi-column features.

    This test verifies that parse_json correctly handles JSON configurations
    with the column_index field, which specifies which column to access when
    a feature name maps to multiple columns.
    """
    config_str = """[
        {
            "name": "feature1",
            "column_index": 0
        },
        {
            "name": "feature2",
            "column_index": 2,
            "options": {"param": "value"}
        },
        {
            "name": "feature3",
            "column_index": 1,
            "group_options": {"group_param": "group_value"}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 3

    # First: feature with column_index
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "feature1"
    assert result[0].column_index == 0
    assert result[0].options == {}

    # Second: feature with column_index and options
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "feature2"
    assert result[1].column_index == 2
    assert result[1].options == {"param": "value"}

    # Third: feature with column_index and group_options
    assert isinstance(result[2], FeatureConfig)
    assert result[2].name == "feature3"
    assert result[2].column_index == 1
    assert result[2].group_options == {"group_param": "group_value"}


def test_parse_json_handles_tilde_in_name() -> None:
    """Test parsing JSON with tilde (~) in feature names.

    This test verifies that parse_json correctly preserves tilde characters
    in feature names when they are present in the JSON configuration. The
    tilde is used internally for multi-column access disambiguation, but
    if it's already in the JSON, it should be preserved as-is.
    """
    config_str = """[
        "feature~0",
        {
            "name": "complex_feature~1",
            "options": {"param": "value"}
        },
        {
            "name": "chained_feature~2",
            "mloda_sources": ["source_col"],
            "column_index": 2
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 3

    # First: string feature with tilde
    assert isinstance(result[0], str)
    assert result[0] == "feature~0"

    # Second: object feature with tilde and options
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "complex_feature~1"
    assert result[1].options == {"param": "value"}

    # Third: chained feature with tilde and column_index
    assert isinstance(result[2], FeatureConfig)
    assert result[2].name == "chained_feature~2"
    assert result[2].mloda_sources == ["source_col"]
    assert result[2].column_index == 2


def test_parse_json_with_at_reference_in_mloda_source() -> None:
    """Test parsing JSON with @ reference syntax in mloda_sources field.

    This test verifies that parse_json correctly handles @ prefix syntax
    in mloda_sources fields. The @ prefix is used to reference other features
    by their defined name, and should be preserved as-is during parsing.
    """
    config_str = """[
        {
            "name": "scaled_feature",
            "mloda_sources": ["@base_feature"]
        },
        {
            "name": "normalized_feature",
            "mloda_sources": ["@scaled_feature"],
            "options": {"method": "minmax"}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 2

    # First: feature with @ reference in mloda_sources
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "scaled_feature"
    assert result[0].mloda_sources == ["@base_feature"]
    assert result[0].options == {}

    # Second: feature with @ reference and options
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "normalized_feature"
    assert result[1].mloda_sources == ["@scaled_feature"]
    assert result[1].options == {"method": "minmax"}


def test_parse_json_preserves_reference_syntax() -> None:
    """Test that parse_json preserves @ reference syntax exactly as provided.

    This test verifies that the parser treats @ references as plain strings
    and preserves them exactly as they appear in the JSON configuration,
    without any transformation or special handling.
    """
    config_str = """[
        {
            "name": "feature1",
            "mloda_sources": ["@referenced_feature"]
        },
        {
            "name": "feature2",
            "mloda_sources": ["regular_column"]
        },
        {
            "name": "feature3",
            "mloda_sources": ["@another_reference"],
            "group_options": {"group_param": "value"},
            "context_options": {"context_param": 42}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 3

    # First: @ reference is preserved exactly
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "feature1"
    assert result[0].mloda_sources == ["@referenced_feature"]
    assert result[0].mloda_sources[0].startswith("@")

    # Second: regular column name without @
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "feature2"
    assert result[1].mloda_sources == ["regular_column"]
    assert not result[1].mloda_sources[0].startswith("@")

    # Third: @ reference preserved with other fields
    assert isinstance(result[2], FeatureConfig)
    assert result[2].name == "feature3"
    assert result[2].mloda_sources == ["@another_reference"]
    assert result[2].mloda_sources[0].startswith("@")
    assert result[2].group_options == {"group_param": "value"}
    assert result[2].context_options == {"context_param": 42}


def test_parse_json_with_mloda_sources_array() -> None:
    """Test parsing JSON with mloda_sources array field.

    This test verifies that parse_json correctly handles JSON configurations
    with the mloda_sources field, which allows a feature to be derived from
    multiple source features or columns.
    """
    config_str = """[
        {
            "name": "combined_feature",
            "mloda_sources": ["feature1", "feature2"]
        },
        {
            "name": "multi_source_feature",
            "mloda_sources": ["col_a", "col_b", "col_c"],
            "options": {"method": "concat"}
        },
        {
            "name": "ref_multi_feature",
            "mloda_sources": ["@base_feature", "other_col"],
            "group_options": {"group_param": "value"}
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 3

    # First: feature with mloda_sources array
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "combined_feature"
    assert result[0].mloda_sources == ["feature1", "feature2"]
    assert result[0].options == {}

    # Second: feature with mloda_sources and options
    assert isinstance(result[1], FeatureConfig)
    assert result[1].name == "multi_source_feature"
    assert result[1].mloda_sources == ["col_a", "col_b", "col_c"]
    assert result[1].options == {"method": "concat"}

    # Third: feature with mloda_sources including @ reference
    assert isinstance(result[2], FeatureConfig)
    assert result[2].name == "ref_multi_feature"
    assert result[2].mloda_sources == ["@base_feature", "other_col"]
    assert result[2].group_options == {"group_param": "value"}


def test_parse_json_rejects_mloda_source_and_mloda_sources_together() -> None:
    """Test that parse_json no longer accepts mloda_source (singular).

    This test verifies that only mloda_sources (plural, as a list) is accepted,
    and that the old mloda_source field is no longer supported.

    Note: This test is kept for documentation purposes but the validation
    should now reject any use of mloda_source in favor of mloda_sources.
    """
    # This config should be valid now since we only use mloda_sources
    config_str = """[
        {
            "name": "valid_feature",
            "mloda_sources": ["source1", "source2"]
        }
    ]"""

    # This should parse successfully
    result = parse_json(config_str)
    assert len(result) == 1
    assert result[0].mloda_sources == ["source1", "source2"]
