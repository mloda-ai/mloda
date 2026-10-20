"""
Unit tests for FeatureConfig model.

This module tests the validation and behavior of the FeatureConfig model
defined in mloda.core.api.feature_config.models.
"""

import re
from typing import Any

import pytest

from mloda.core.api.feature_config.models import FeatureConfig


def test_valid_feature_config_with_name_only() -> None:
    """Test creating a valid FeatureConfig with only a name."""
    config = FeatureConfig(name="test_feature")

    assert config.name == "test_feature"
    assert config.options == {}


def test_valid_feature_config_with_options() -> None:
    """Test creating a valid FeatureConfig with name and options."""
    options = {"enabled": True, "threshold": 0.75}
    config = FeatureConfig(name="test_feature", options=options)

    assert config.name == "test_feature"
    assert config.options == options


def test_feature_config_options_default_empty() -> None:
    """Test that options defaults to an empty dict when not provided."""
    config = FeatureConfig(name="default_test")

    assert config.options == {}
    assert isinstance(config.options, dict)


def test_feature_config_rejects_empty_name() -> None:
    """Test that an empty name is rejected (shared invariant for the top-level and nested paths)."""
    with pytest.raises(ValueError, match=re.escape("'name' must be a non-empty string, got an empty string")):
        FeatureConfig(name="")


def test_feature_config_rejects_whitespace_only_name() -> None:
    """Test that a whitespace-only name strips to nothing and is rejected."""
    with pytest.raises(ValueError, match=re.escape("'name' must be a non-empty string, got a blank string")):
        FeatureConfig(name="   ")


def test_feature_config_rejects_non_string_name() -> None:
    """Test that a non-string name is rejected: 'name' is a non-empty string."""
    bad_value: Any = 123

    with pytest.raises(ValueError, match=re.escape("'name' must be a non-empty string, got int")):
        FeatureConfig(name=bad_value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("x", "got str", id="str"),
        pytest.param(True, "got bool", id="bool"),
        pytest.param({"api_key": "sk-test-1234567890abcdef"}, "got a dict with keys ['api_key']", id="dict"),  # nosec B105
        pytest.param(-1, "got a negative integer", id="negative"),
    ],
)
def test_column_index_rejects_invalid_values_without_echo(value: Any, expected: str) -> None:
    """A non-integer, bool or negative column_index raises and never echoes the value."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="f", column_index=value)

    message = str(exc_info.value)
    assert f"'column_index' must be a non-negative integer, {expected}" in message
    assert "sk-test-1234567890abcdef" not in message


@pytest.mark.parametrize("value", [None, 0, 3])
def test_column_index_accepts_none_and_non_negative_int(value: int | None) -> None:
    """None and non-negative integers are valid column_index values."""
    assert FeatureConfig(name="f", column_index=value).column_index == value
