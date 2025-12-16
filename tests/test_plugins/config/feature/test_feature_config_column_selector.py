"""
Unit tests for multi-column access support in feature configuration.

This module tests the `~` syntax for accessing specific columns from
multi-output transformations (e.g., one-hot encoding that produces multiple columns).
"""

from mloda_plugins.config.feature.models import FeatureConfig
from mloda_plugins.config.feature.parser import parse_json
from mloda_plugins.config.feature.loader import load_features_from_config
from mloda import Feature


def test_parse_feature_with_column_index() -> None:
    """Test parsing JSON with column_index field for multi-column access.

    When a transformation produces multiple columns (e.g., one-hot encoding),
    the column_index field specifies which column to select. This test verifies
    that parse_json correctly handles the column_index field.

    Example:
        {
            "name": "onehot_encoded__state",
            "column_index": 0
        }
    """
    config_str = """[
        {
            "name": "onehot_encoded__state",
            "column_index": 0
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "onehot_encoded__state"
    assert result[0].column_index == 0


def test_parse_feature_with_tilde_syntax() -> None:
    """Test that features with column_index get `~{index}` appended to name.

    When a feature configuration includes a column_index field, the loader
    should automatically append `~{index}` to the feature name to indicate
    which column is being accessed from the multi-output transformation.

    This test verifies that the loader modifies the feature name to include
    the tilde syntax (e.g., "onehot_encoded__state" becomes "onehot_encoded__state~0").
    """

    config_str = """[
        {
            "name": "onehot_encoded__state",
            "column_index": 0
        },
        {
            "name": "onehot_encoded__state",
            "column_index": 1
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2
    assert isinstance(result[0], Feature)
    assert isinstance(result[1], Feature)

    # Feature names should have tilde syntax appended
    assert result[0].name.name == "onehot_encoded__state~0"
    assert result[1].name.name == "onehot_encoded__state~1"


def test_load_column_selector_feature() -> None:
    """Test that loader appends `~{index}` to feature name when column_index is present.

    This test verifies the complete loader behavior when processing features
    with column_index fields:
    1. The column_index field is read from the JSON configuration
    2. The loader creates a Feature object
    3. The feature name has `~{column_index}` appended to it
    4. Any options from the config are preserved in the Feature object

    Example:
        Input: {"name": "onehot_encoded__state", "column_index": 2, "options": {"drop_first": true}}
        Output: Feature with name "onehot_encoded__state~2" and options preserved
    """

    config_str = """[
        {
            "name": "onehot_encoded__state",
            "column_index": 2,
            "options": {
                "drop_first": true
            }
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)

    feature = result[0]
    # Feature name should have tilde syntax appended
    assert feature.name.name == "onehot_encoded__state~2"

    # Original options should be preserved
    assert feature.options.get("drop_first") is True
