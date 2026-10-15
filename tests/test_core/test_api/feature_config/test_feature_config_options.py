"""
Unit tests for group/context options separation in feature configuration.

This module tests the new Options architecture that separates group_options
and context_options for performance optimization.
"""

from typing import Any

import pytest

from mloda.core.api.feature_config.models import FeatureConfig
from mloda.core.api.feature_config.parser import parse_json


def test_parse_group_options() -> None:
    """Test parsing JSON with group_options field instead of simple options.

    The new Options architecture separates group_options and context_options
    for performance optimization. This test verifies that parse_json correctly
    handles the group_options field.

    Example:
        {
            "name": "production_feature",
            "group_options": {
                "data_source": "production"
            }
        }
    """
    config_str = """[
        {
            "name": "production_feature",
            "group_options": {
                "data_source": "production"
            }
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "production_feature"
    assert result[0].group_options == {"data_source": "production"}
    assert result[0].options == {}


def test_parse_context_options() -> None:
    """Test parsing JSON with context_options field.

    The new Options architecture separates group_options and context_options
    for performance optimization. This test verifies that parse_json correctly
    handles the context_options field.

    Example:
        {
            "name": "cached_feature",
            "context_options": {
                "cache_enabled": true
            }
        }
    """
    config_str = """[
        {
            "name": "cached_feature",
            "context_options": {
                "cache_enabled": true
            }
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "cached_feature"
    assert result[0].context_options == {"cache_enabled": True}
    assert result[0].options == {}


def test_parse_group_and_context_together() -> None:
    """Test parsing JSON with both group_options and context_options fields.

    The new Options architecture separates group_options and context_options.
    This test verifies that parse_json correctly handles both fields when
    present in the same feature configuration.

    Example:
        {
            "name": "advanced_feature",
            "group_options": {
                "data_source": "production"
            },
            "context_options": {
                "cache_enabled": true
            }
        }
    """
    config_str = """[
        {
            "name": "advanced_feature",
            "group_options": {
                "data_source": "production"
            },
            "context_options": {
                "cache_enabled": true
            }
        }
    ]"""

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "advanced_feature"
    assert result[0].group_options == {"data_source": "production"}
    assert result[0].context_options == {"cache_enabled": True}
    assert result[0].options == {}


def test_load_creates_options_with_group_context() -> None:
    """Test that the loader creates proper Options objects with group/context separation.

    When a feature config includes both group_options and context_options,
    the loader should create a Feature object with an Options instance that
    properly separates the two types of options.

    The Options object should have:
    - options.group containing the group_options
    - options.context containing the context_options
    """
    from mloda.core.api.feature_config.loader import load_features_from_config
    from mloda.user import Feature

    config_str = """[
        {
            "name": "optimized_feature",
            "group_options": {
                "data_source": "production",
                "threshold": 0.5
            },
            "context_options": {
                "cache_enabled": true,
                "debug_mode": false
            }
        }
    ]"""

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name == "optimized_feature"

    # Verify Options object has correct group options
    assert result[0].options.group == {"data_source": "production", "threshold": 0.5}

    # Verify Options object has correct context options
    assert result[0].options.context == {"cache_enabled": True, "debug_mode": False}


@pytest.mark.parametrize("value", ["session", 5, {"k": 1}, ["a", 1]])
def test_propagate_context_keys_rejects_invalid_types(value: Any) -> None:
    with pytest.raises(ValueError, match="propagate_context_keys"):
        FeatureConfig(
            name="f",
            context_options={"session": "value"},
            propagate_context_keys=value,
        )


@pytest.mark.parametrize("value", [None, [], ["session"]])
def test_propagate_context_keys_accepts_valid_types(value):
    config = FeatureConfig(
        name="f",
        context_options={"session": "value"},
        propagate_context_keys=value,
    )
    assert config.propagate_context_keys == value


def test_validate_options_mutual_exclusion() -> None:
    """Test that options and group_options/context_options are mutually exclusive.

    The Options architecture uses either:
    - The simple `options` field (simple dict), OR
    - The `group_options` and/or `context_options` fields

    Users should NOT mix both formats together. This test verifies that
    FeatureConfig raises a ValueError when both formats are used.

    Invalid examples that should raise ValueError:
        - options + group_options together
        - options + context_options together
        - options + both group_options and context_options
    """
    # Test case 1: options + group_options should fail
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="bad_feature_1", options={"foo": "bar"}, group_options={"baz": "qux"})
    assert "options" in str(exc_info.value).lower()

    # Test case 2: options + context_options should fail
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="bad_feature_2", options={"foo": "bar"}, context_options={"baz": "qux"})
    assert "options" in str(exc_info.value).lower()

    # Test case 3: options + both group_options and context_options should fail
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(
            name="bad_feature_3",
            options={"foo": "bar"},
            group_options={"baz": "qux"},
            context_options={"alpha": "beta"},
        )
    assert "options" in str(exc_info.value).lower()


_KEY_FIELDS = ["propagate_context_keys", "cascade_context_keys"]


@pytest.mark.parametrize("field", _KEY_FIELDS)
def test_parse_context_push_keys(field: str) -> None:
    """parse_json stores the context push key list on the FeatureConfig."""
    config_str = '[{"name": "f", "context_options": {"session_id": "abc"}, "' + field + '": ["session_id"]}]'

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert result[0].name == "f"
    assert result[0].context_options == {"session_id": "abc"}
    assert getattr(result[0], field) == ["session_id"]


@pytest.mark.parametrize("field", _KEY_FIELDS)
def test_parse_context_push_keys_defaults_none(field: str) -> None:
    """The context push key list is None when absent."""
    config_str = '[{"name": "f", "context_options": {"session_id": "abc"}}]'

    result = parse_json(config_str)

    assert len(result) == 1
    assert isinstance(result[0], FeatureConfig)
    assert getattr(result[0], field) is None


@pytest.mark.parametrize("field", _KEY_FIELDS)
def test_load_threads_context_push_keys_into_options(field: str) -> None:
    """The loader threads the context push keys into the Feature's Options as a frozenset."""
    from mloda.core.api.feature_config.loader import load_features_from_config
    from mloda.user import Feature

    config_str = '[{"name": "f", "context_options": {"session_id": "abc"}, "' + field + '": ["session_id"]}]'

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].options.context == {"session_id": "abc"}
    assert getattr(result[0].options, field) == frozenset({"session_id"})


@pytest.mark.parametrize("field", _KEY_FIELDS)
def test_load_context_push_keys_defaults_empty_frozenset(field: str) -> None:
    """Absent context push keys give the default empty frozenset in Options."""
    from mloda.core.api.feature_config.loader import load_features_from_config
    from mloda.user import Feature

    config_str = '[{"name": "f", "context_options": {"session_id": "abc"}}]'

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert getattr(result[0].options, field) == frozenset()


@pytest.mark.parametrize("field", _KEY_FIELDS)
def test_load_documented_context_push_keys_example(field: str) -> None:
    """The documented example yields one Feature pushing the documented key."""
    from mloda.core.api.feature_config.loader import load_features_from_config
    from mloda.user import Feature

    config_str = (
        '[{"name": "my_feature", "context_options": {"session_id": "abc123", "window_function": "sum"}, "'
        + field
        + '": ["session_id"]}]'
    )

    result = load_features_from_config(config_str)

    assert len(result) == 1
    assert isinstance(result[0], Feature)
    assert result[0].name == "my_feature"
    assert result[0].options.context == {"session_id": "abc123", "window_function": "sum"}
    assert getattr(result[0].options, field) == frozenset({"session_id"})


def test_load_both_context_push_keys_arrive_in_options() -> None:
    """Both key fields set on one feature both reach Options."""
    from mloda.core.api.feature_config.loader import load_features_from_config
    from mloda.user import Feature

    config_str = (
        '[{"name": "f", "context_options": {"a": 1, "b": 2},'
        ' "propagate_context_keys": ["a"], "cascade_context_keys": ["b"]}]'
    )

    result = load_features_from_config(config_str)

    assert isinstance(result[0], Feature)
    assert result[0].options.propagate_context_keys == frozenset({"a"})
    assert result[0].options.cascade_context_keys == frozenset({"b"})


@pytest.mark.parametrize("field", _KEY_FIELDS)
def test_context_push_keys_requires_context_options(field: str) -> None:
    """The context push keys require context_options, so the loader never silently drops them."""
    kwargs: dict[str, Any] = {field: ["session_id"]}
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(**kwargs, name="bad_feature")
    assert field in str(exc_info.value).lower()

    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(**kwargs, name="bad_feature_2", options={"foo": "bar"})
    assert field in str(exc_info.value).lower()


@pytest.mark.parametrize("field", _KEY_FIELDS)
def test_load_context_push_key_not_in_context_raises(field: str) -> None:
    """A key absent from context_options surfaces the Options ValueError through the loader."""
    from mloda.core.api.feature_config.loader import load_features_from_config

    config_str = '[{"name": "f", "context_options": {"a": 1}, "' + field + '": ["b"]}]'

    with pytest.raises(ValueError, match=field):
        load_features_from_config(config_str)
