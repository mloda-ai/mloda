"""
Unit tests for FeatureConfig model.

This module tests the validation and behavior of the FeatureConfig model
defined in mloda.core.api.feature_config.models.
"""

import re
from typing import Any, cast

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


def test_keep_index_and_index_defaults() -> None:
    """keep_index defaults to False and index to None."""
    config = FeatureConfig(name="f")

    assert config.keep_index is False
    assert config.index is None


def test_keep_index_and_index_accepted() -> None:
    """A real bool keep_index and a list of non-blank strings are valid; index does not require keep_index."""
    config = FeatureConfig(name="f", keep_index=True, index=["id", "k"])
    assert config.keep_index is True
    assert config.index == ["id", "k"]
    assert FeatureConfig(name="f", index=["id"]).index == ["id"]


@pytest.mark.parametrize("value", [1, 0, "true", None, {"a": 1}])
def test_keep_index_rejects_non_bool(value: Any) -> None:
    """keep_index must be a real bool."""
    with pytest.raises(ValueError, match="keep_index"):
        FeatureConfig(name="f", keep_index=value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([], id="empty"),
        pytest.param("id", id="string"),
        pytest.param([""], id="empty_item"),
        pytest.param(["id", "   "], id="blank_item"),
        pytest.param(["id", 1], id="int_item"),
        pytest.param({"k": "id"}, id="dict"),
        pytest.param(["id", "id"], id="duplicate_items"),
    ],
)
def test_index_rejects_invalid_values(value: Any) -> None:
    """index must be a non-empty list of non-blank strings."""
    with pytest.raises(ValueError, match="index"):
        FeatureConfig(name="f", index=value)


def test_index_error_does_not_echo_values() -> None:
    """A rejected index never echoes its offending value."""
    with pytest.raises(ValueError) as exc_info:
        FeatureConfig(name="f", index=cast(Any, [{"api_key": "sk-test-1234567890abcdef"}]))  # nosec B106

    assert "sk-test-1234567890abcdef" not in str(exc_info.value)


def test_index_duplicate_error_does_not_echo_values() -> None:
    """A rejected duplicate index entry is not echoed."""
    secret = "sk-test-1234567890abcdef"  # nosec B105
    with pytest.raises(ValueError, match="index") as exc_info:
        FeatureConfig(name="f", index=[secret, secret])

    assert secret not in str(exc_info.value)
