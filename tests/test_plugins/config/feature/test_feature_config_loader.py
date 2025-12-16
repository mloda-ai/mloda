"""
Unit tests for feature configuration loader.

This module tests the load_features_from_config function from mloda_plugins.config.feature.loader.
"""

import pytest

from mloda import Feature
from mloda_plugins.config.feature.loader import load_features_from_config
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


def test_load_features_string_only() -> None:
    """Test loading config with only string features."""
    config_str = '["feature1", "feature2", "feature3"]'

    result = load_features_from_config(config_str)

    assert len(result) == 3
    assert result[0] == "feature1"
    assert result[1] == "feature2"
    assert result[2] == "feature3"
    assert all(isinstance(item, str) for item in result)


def test_load_features_with_objects() -> None:
    """Test loading config with FeatureConfig objects."""
    config_str = """[
        {"name": "feature1", "options": {"enabled": true}},
        {"name": "feature2", "options": {"threshold": 0.8}}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2
    assert isinstance(result[0], Feature)
    assert isinstance(result[1], Feature)
    assert result[0].name.name == "feature1"
    assert result[0].options.get("enabled") is True
    assert result[1].name.name == "feature2"
    assert result[1].options.get("threshold") == 0.8


def test_load_features_mixed() -> None:
    """Test loading config with both strings and objects."""
    config_str = """[
        "simple_feature",
        {"name": "complex_feature", "options": {"param": "value"}},
        "another_simple"
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], Feature)
    assert isinstance(result[2], str)
    assert result[0] == "simple_feature"
    assert result[1].name.name == "complex_feature"
    assert result[1].options.get("param") == "value"
    assert result[2] == "another_simple"


def test_load_features_unsupported_format() -> None:
    """Test that ValueError is raised for unsupported format."""
    config_str = '["feature1"]'

    with pytest.raises(ValueError, match="Unsupported format: yaml"):
        load_features_from_config(config_str, format="yaml")


def test_load_features_preserves_options() -> None:
    """Test that options dict is correctly passed to Feature objects."""
    config_str = """[
        {"name": "feature_with_options", "options": {"key1": "value1", "key2": 42, "key3": true}}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "feature_with_options"
    assert result[0].options.get("key1") == "value1"
    assert result[0].options.get("key2") == 42
    assert result[0].options.get("key3") is True


def test_load_features_with_mloda_source() -> None:
    """Test that load_features_from_config correctly creates Feature objects with mloda_sources in options.context.

    When a feature config includes mloda_sources field (e.g., for chained features like "scale__age"),
    the loader should:
    1. Create a Feature object with an Options instance
    2. Place mloda_sources in options.context[DefaultOptionKeys.in_features]
    3. Place regular options in options.group
    """
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    config_str = """[
        {
            "name": "scale__age",
            "mloda_sources": ["age"],
            "options": {"method": "standard"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "scale__age"

    # mloda_sources should be in options.context as a frozenset
    mloda_sources_value = result[0].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset)
    assert mloda_sources_value == frozenset({"age"})

    # Regular options should be in options.group
    assert result[0].options.group.get("method") == "standard"


def test_load_features_mixed_chained_and_simple() -> None:
    """Test that the loader can handle a mix of simple, regular, and chained features in one configuration.

    This test validates the loader's ability to process a realistic mixed configuration containing:
    1. Simple string features (e.g., "age")
    2. Regular Feature objects with options only (e.g., {"name": "weight", "options": {...}})
    3. Chained features with mloda_sources (e.g., {"name": "scale__age", "mloda_sources": ["age"], "options": {...}})

    Each type should be processed correctly with appropriate option placement:
    - Simple strings remain as strings
    - Regular features have options in group, no in_features in context
    - Chained features have in_features in context and options in group
    """
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

    config_str = """[
        "age",
        {"name": "weight", "options": {"unit": "kg"}},
        {
            "name": "standard_scaled__mean_imputed__age",
            "mloda_sources": ["age"],
            "options": {"method": "robust"}
        },
        "height"
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 4

    # First feature: simple string
    assert isinstance(result[0], str)
    assert result[0] == "age"

    # Second feature: regular Feature with options only
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "weight"
    assert result[1].options.group.get("unit") == "kg"
    # Should NOT have in_features in context
    assert DefaultOptionKeys.in_features not in result[1].options.context

    # Third feature: chained Feature with mloda_sources
    assert isinstance(result[2], Feature)
    assert result[2].name.name == "standard_scaled__mean_imputed__age"
    mloda_sources_value = result[2].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset)
    assert mloda_sources_value == frozenset({"age"})
    assert result[2].options.group.get("method") == "robust"

    # Fourth feature: simple string
    assert isinstance(result[3], str)
    assert result[3] == "height"


def test_load_features_with_legacy_options() -> None:
    """Test that the loader handles the legacy 'options' field correctly for backward compatibility.

    When a feature config uses the legacy 'options' field (without group_options/context_options),
    the loader should:
    1. Create a Feature object with an Options instance
    2. Place all legacy options in Options.group (for backward compatibility)
    3. Leave Options.context empty
    """
    config_str = """[
        {"name": "legacy_feature", "options": {"param1": "value1", "param2": 42}}
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "legacy_feature"

    # Legacy options should be in group
    assert result[0].options.group.get("param1") == "value1"
    assert result[0].options.group.get("param2") == 42

    # Context should be empty
    assert len(result[0].options.context) == 0


def test_load_features_with_group_context_options() -> None:
    """Test that the loader creates Options objects with proper group/context separation.

    When a feature config uses the new group_options and context_options fields,
    the loader should:
    1. Create a Feature object with an Options instance
    2. Place group_options in Options.group
    3. Place context_options in Options.context
    """

    config_str = """[
        {
            "name": "modern_feature",
            "group_options": {"threshold": 0.5, "method": "advanced"},
            "context_options": {"metadata": "test"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "modern_feature"

    # Group options should be in group
    assert result[0].options.group.get("threshold") == 0.5
    assert result[0].options.group.get("method") == "advanced"

    # Context options should be in context
    assert result[0].options.context.get("metadata") == "test"


def test_load_creates_proper_options_object() -> None:
    """Test that the loader creates the correct Options structure for various scenarios.

    This test verifies proper Options object creation for:
    1. Legacy format (options) - all options go to group
    2. New format (group_options/context_options) - proper separation
    3. Mixed feature configs in one configuration
    """
    config_str = """[
        {"name": "legacy", "options": {"legacy_param": "value"}},
        {
            "name": "modern",
            "group_options": {"group_param": "group_value"},
            "context_options": {"context_param": "context_value"}
        },
        {
            "name": "only_group",
            "group_options": {"only_group_param": "only_group_value"}
        },
        {
            "name": "only_context",
            "context_options": {"only_context_param": "only_context_value"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 4

    # Legacy feature: options in group, context empty
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "legacy"
    assert result[0].options.group.get("legacy_param") == "value"
    assert len(result[0].options.context) == 0

    # Modern feature: proper group/context separation
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "modern"
    assert result[1].options.group.get("group_param") == "group_value"
    assert result[1].options.context.get("context_param") == "context_value"

    # Only group options feature
    assert isinstance(result[2], Feature)
    assert result[2].name.name == "only_group"
    assert result[2].options.group.get("only_group_param") == "only_group_value"
    assert len(result[2].options.context) == 0

    # Only context options feature
    assert isinstance(result[3], Feature)
    assert result[3].name.name == "only_context"
    assert result[3].options.context.get("only_context_param") == "only_context_value"
    assert len(result[3].options.group) == 0


def test_load_features_with_column_index() -> None:
    """Test that loader correctly appends `~{index}` to feature name when column_index is set.

    When a feature configuration includes a column_index field (used for accessing specific
    columns from multi-output transformations like one-hot encoding), the loader should:
    1. Parse the column_index field from the JSON configuration
    2. Append `~{column_index}` to the feature name
    3. Create a Feature object with the modified name
    4. Preserve any options from the original configuration

    Example:
        Input: {"name": "onehot_encoded__state", "column_index": 2}
        Output: Feature with name "onehot_encoded__state~2"
    """
    config_str = """[
        {
            "name": "onehot_encoded__state",
            "column_index": 0
        },
        {
            "name": "onehot_encoded__state",
            "column_index": 1
        },
        {
            "name": "onehot_encoded__state",
            "column_index": 2,
            "options": {"drop_first": true}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 3

    # First feature: column_index 0
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "onehot_encoded__state~0"

    # Second feature: column_index 1
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "onehot_encoded__state~1"

    # Third feature: column_index 2 with options
    assert isinstance(result[2], Feature)
    assert result[2].name.name == "onehot_encoded__state~2"
    assert result[2].options.get("drop_first") is True


def test_load_appends_tilde_syntax_to_name() -> None:
    """Test the tilde syntax appending in various scenarios (with options, with mloda_sources, etc.).

    This test verifies that the loader correctly appends `~{column_index}` to feature names
    across different configuration scenarios:
    1. With legacy options field
    2. With mloda_sources field (chained features)
    3. With new group_options/context_options fields
    4. Without any options (minimal configuration)

    The tilde syntax should always be appended when column_index is present, regardless of
    what other fields are in the configuration.
    """

    config_str = """[
        {
            "name": "onehot__category",
            "column_index": 0,
            "options": {"method": "standard"}
        },
        {
            "name": "scale__mean_imputed__age",
            "column_index": 1,
            "mloda_sources": ["age"],
            "options": {"scaler": "robust"}
        },
        {
            "name": "vectorized__text",
            "column_index": 2,
            "group_options": {"max_features": 100},
            "context_options": {"encoding": "utf-8"}
        },
        {
            "name": "embedded__description",
            "column_index": 3
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 4

    # First: with legacy options
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "onehot__category~0"
    assert result[0].options.get("method") == "standard"

    # Second: with mloda_sources
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "scale__mean_imputed__age~1"
    mloda_sources_value = result[1].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset)
    assert mloda_sources_value == frozenset({"age"})
    assert result[1].options.group.get("scaler") == "robust"

    # Third: with group_options and context_options
    assert isinstance(result[2], Feature)
    assert result[2].name.name == "vectorized__text~2"
    assert result[2].options.group.get("max_features") == 100
    assert result[2].options.context.get("encoding") == "utf-8"

    # Fourth: minimal configuration
    assert isinstance(result[3], Feature)
    assert result[3].name.name == "embedded__description~3"


def test_load_detects_feature_references() -> None:
    """Test that the loader detects when mloda_sources starts with @ prefix.

    When a feature configuration includes mloda_sources field with @ prefix (e.g., ["@base_feature"]),
    the loader should:
    1. Detect the @ prefix in the mloda_sources field
    2. Recognize this as a feature reference (not a string column name)
    3. Process it differently from regular string mloda_sources values

    This test verifies the detection mechanism exists, though full resolution
    will be tested in test_load_resolves_references_to_feature_objects.
    """

    config_str = """[
        {
            "name": "base_feature",
            "options": {"method": "standard"}
        },
        {
            "name": "derived_feature",
            "mloda_sources": ["@base_feature"],
            "options": {"transformation": "log"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2

    # First feature: base feature
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "base_feature"

    # Second feature: should have reference to base_feature
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "derived_feature"

    # The in_features should contain a frozenset with Feature object, not a string
    mloda_sources_value = result[1].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset), "Expected frozenset for mloda_sources"
    # Extract the single feature from the frozenset
    feature_list = list(mloda_sources_value)
    assert len(feature_list) == 1
    assert isinstance(feature_list[0], Feature), "Expected Feature object for @reference, got string"
    assert feature_list[0].name.name == "base_feature"


def test_load_resolves_references_to_feature_objects() -> None:
    """Test that @feature_name references are resolved to actual Feature objects.

    When a feature config includes mloda_sources with @ prefix, the loader should:
    1. Find the referenced feature by name in the feature registry
    2. Replace the string reference with the actual Feature object
    3. Store the Feature object in options.context[in_features]
    4. Handle multiple features referencing the same base feature

    Example:
        Input: {"name": "scaled", "mloda_sources": ["@age"]}
        Output: Feature.options.context[in_features] = frozenset({Feature(name="age")})
    """

    config_str = """[
        "age",
        {
            "name": "imputed_age",
            "mloda_sources": ["@age"],
            "options": {"method": "mean"}
        },
        {
            "name": "scaled_age",
            "mloda_sources": ["@imputed_age"],
            "options": {"scaler": "standard"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 3

    # First feature: simple string (base feature)
    assert result[0] == "age"

    # Second feature: references base feature
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "imputed_age"
    mloda_sources_1 = result[1].options.context.get(DefaultOptionKeys.in_features)
    # Should be a frozenset containing the actual Feature object with name "age", not a string
    assert isinstance(mloda_sources_1, frozenset), "Expected frozenset"
    feature_list_1 = list(mloda_sources_1)
    assert len(feature_list_1) == 1
    assert isinstance(feature_list_1[0], Feature), "Expected Feature object, not string"
    assert feature_list_1[0].name.name == "age"

    # Third feature: references the second feature (chained reference)
    assert isinstance(result[2], Feature)
    assert result[2].name.name == "scaled_age"
    mloda_sources_2 = result[2].options.context.get(DefaultOptionKeys.in_features)
    # Should be a frozenset containing the actual imputed_age Feature object
    assert isinstance(mloda_sources_2, frozenset), "Expected frozenset"
    feature_list_2 = list(mloda_sources_2)
    assert len(feature_list_2) == 1
    assert isinstance(feature_list_2[0], Feature), "Expected Feature object, not string"
    assert feature_list_2[0].name.name == "imputed_age"


def test_load_handles_forward_references() -> None:
    """Test that the loader handles forward references (referencing features defined later).

    The loader should support two-pass resolution:
    1. First pass: Create all Feature objects
    2. Second pass: Resolve @references to Feature objects

    This allows features to reference other features that appear later in the configuration,
    which is important for flexibility in config ordering.

    Example:
        [
            {"name": "derived", "mloda_sources": ["@base"]},  # Forward reference
            {"name": "base", "options": {...}}              # Defined later
        ]
    """

    config_str = """[
        {
            "name": "derived_feature",
            "mloda_sources": ["@base_feature"],
            "options": {"transformation": "log"}
        },
        {
            "name": "base_feature",
            "options": {"method": "standard"}
        },
        {
            "name": "another_derived",
            "mloda_sources": ["@base_feature"],
            "options": {"normalization": "minmax"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 3

    # First feature: forward reference to base_feature
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "derived_feature"
    mloda_sources_1 = result[0].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_1, frozenset), "Forward reference should resolve to frozenset"
    feature_list_1 = list(mloda_sources_1)
    assert len(feature_list_1) == 1
    assert isinstance(feature_list_1[0], Feature), "Forward reference should resolve to Feature object"
    assert feature_list_1[0].name.name == "base_feature"

    # Second feature: base feature (defined after first reference)
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "base_feature"

    # Third feature: also references base_feature
    assert isinstance(result[2], Feature)
    assert result[2].name.name == "another_derived"
    mloda_sources_3 = result[2].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_3, frozenset), "Should also resolve to frozenset"
    feature_list_3 = list(mloda_sources_3)
    assert len(feature_list_3) == 1
    assert isinstance(feature_list_3[0], Feature), "Should also resolve to Feature object"
    assert feature_list_3[0].name.name == "base_feature"

    # Verify both derived features reference the SAME Feature object
    assert feature_list_1[0] is result[1], "Should reference the actual base_feature object"
    assert feature_list_3[0] is result[1], "Should reference the actual base_feature object"


def test_load_features_with_multiple_mloda_sources() -> None:
    """Test that loader handles features with multiple source features via mloda_sources array.

    When a feature configuration includes an mloda_sources field (plural) with an array of
    source feature names, the loader should:
    1. Parse the mloda_sources array from the JSON configuration
    2. Convert the list to a frozenset for immutability and set-like behavior
    3. Store the frozenset in options.context[DefaultOptionKeys.in_features] (singular key)
    4. Preserve any regular options in options.group

    This is useful for features that require multiple source features, such as:
    - Distance calculations (latitude, longitude)
    - Multi-column aggregations (sales, revenue, profit)
    - Complex transformations requiring multiple inputs

    Example:
        Input: {"name": "distance", "mloda_sources": ["latitude", "longitude"]}
        Output: Feature with context[in_features] = frozenset({"latitude", "longitude"})
    """

    config_str = """[
        {
            "name": "distance_feature",
            "mloda_sources": ["latitude", "longitude"],
            "options": {"distance_type": "euclidean"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "distance_feature"

    # mloda_sources should be converted to frozenset in options.context with singular key name
    mloda_sources_value = result[0].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset), "mloda_sources should be converted to frozenset"
    assert mloda_sources_value == frozenset({"latitude", "longitude"})

    # Regular options should still be in options.group
    assert result[0].options.group.get("distance_type") == "euclidean"


def test_load_creates_frozenset_for_mloda_sources() -> None:
    """Test that the loader creates a frozenset when mloda_sources array is provided.

    The loader should convert mloda_sources arrays to frozenset for:
    1. Immutability - prevent accidental modification of source feature sets
    2. Set semantics - eliminate duplicates, support set operations
    3. Hashability - allow features with source sets to be used as dict keys

    This test verifies frozenset creation in various scenarios:
    - Multiple sources (3+ features)
    - Duplicate sources in the array (should be deduplicated)
    - Empty mloda_sources array (edge case)
    """

    config_str = """[
        {
            "name": "multi_source_aggregation",
            "mloda_sources": ["sales", "revenue", "profit"],
            "options": {"aggregation": "sum"}
        },
        {
            "name": "duplicate_sources",
            "mloda_sources": ["feature1", "feature2", "feature1"],
            "options": {"method": "combine"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2

    # First feature: multiple sources
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "multi_source_aggregation"
    mloda_sources_1 = result[0].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_1, frozenset), "Should be frozenset"
    assert mloda_sources_1 == frozenset({"sales", "revenue", "profit"})
    assert result[0].options.group.get("aggregation") == "sum"

    # Second feature: duplicates should be deduplicated by frozenset
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "duplicate_sources"
    mloda_sources_2 = result[1].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_2, frozenset), "Should be frozenset"
    # frozenset automatically handles duplicates - should contain 2 items, not 3
    assert mloda_sources_2 == frozenset({"feature1", "feature2"})
    assert len(mloda_sources_2) == 2
    assert result[1].options.group.get("method") == "combine"


def test_load_adds_mloda_sources_to_in_features_option() -> None:
    """Test that mloda_sources are stored in the correct context option key.

    The loader uses DefaultOptionKeys.in_features (singular) for:
    - mloda_sources (plural): Always stored as a frozenset

    This unified approach allows Options.get_in_features() to handle both
    single-source and multi-source transformations consistently.
    """

    config_str = """[
        {
            "name": "single_source_feature",
            "mloda_sources": ["age"],
            "options": {"method": "standard"}
        },
        {
            "name": "multi_source_feature",
            "mloda_sources": ["latitude", "longitude"],
            "options": {"distance_type": "haversine"}
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 2

    # First feature: single source - stored as frozenset in in_features
    assert isinstance(result[0], Feature)
    assert result[0].name.name == "single_source_feature"
    # Should have in_features (singular) in context
    assert DefaultOptionKeys.in_features in result[0].options.context
    mloda_sources_value_0 = result[0].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value_0, frozenset)
    assert mloda_sources_value_0 == frozenset({"age"})

    # Second feature: multiple sources - stored as frozenset in in_features
    assert isinstance(result[1], Feature)
    assert result[1].name.name == "multi_source_feature"
    # Should have in_features (singular) in context (unified key for both single and multiple)
    assert DefaultOptionKeys.in_features in result[1].options.context
    mloda_sources_value = result[1].options.context.get(DefaultOptionKeys.in_features)
    assert isinstance(mloda_sources_value, frozenset)
    assert mloda_sources_value == frozenset({"latitude", "longitude"})
