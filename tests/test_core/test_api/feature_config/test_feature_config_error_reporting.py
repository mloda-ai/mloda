"""
Tests for item-level error reporting in feature config loading.

Every error raised while parsing or building one item of a feature config must say
which 1-based list item failed (and its name, when the name is usable), must point at
the nested field path for nested errors, and must describe rejected values by type
name (dicts by keys) instead of echoing them, so a rejected value can never leak a
secret held in a config payload. See issue #2004.
"""

import json
from typing import Any

import pytest

from mloda.user import Feature
from mloda.core.api.feature_config.loader import load_features_from_config


SECRET = "sk-test-1234567890abcdef"

OK_ITEM_WITH_SECRET: dict[str, Any] = {"name": "ok_with_secret", "options": {"api_key": SECRET}}


def _config(items: list[Any]) -> str:
    return json.dumps(items)


def _load_expect_error(items: list[Any], expected_type: type[BaseException]) -> str:
    """Load the config and return the message of the raised error."""
    with pytest.raises(expected_type) as exc_info:
        load_features_from_config(_config(items))
    return str(exc_info.value)


class TestFeatureConfigItemErrorReporting:
    """Item errors name the failing config item without echoing rejected values (issue #2004)."""

    def test_unknown_field_names_item_and_keeps_type_error(self) -> None:
        """An unknown field keeps raising TypeError and the message names the failing item and the field."""
        items = [OK_ITEM_WITH_SECRET, {"name": "bad_item", "optionz": {}}, "plain_feature"]

        message = _load_expect_error(items, TypeError)

        assert "feature config item 2" in message
        assert '("bad_item")' in message
        assert "optionz" in message
        assert SECRET not in message

    @pytest.mark.parametrize("field_name", ["options", "group_options", "context_options"])
    def test_non_dict_container_names_field_and_type(self, field_name: str) -> None:
        """A non-dict options/group_options/context_options raises ValueError naming the field and the type."""
        items = [OK_ITEM_WITH_SECRET, {"name": "bad_item", field_name: ["x"]}, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 2" in message
        assert '("bad_item")' in message
        assert field_name in message
        assert "got list" in message
        assert SECRET not in message

    def test_non_string_name_reports_type_not_repr(self) -> None:
        """A list name carries the secret in current repr-based wording; the fix must not echo it."""
        items = [OK_ITEM_WITH_SECRET, {"name": [SECRET]}, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 2" in message
        assert 'feature config item 2 ("' not in message
        assert "list" in message
        assert SECRET not in message

    def test_top_level_in_features_string_reports_type_not_repr(self) -> None:
        """A bare string top-level in_features keeps ValueError but must not be echoed via repr."""
        items = [OK_ITEM_WITH_SECRET, {"name": "bad_item", "in_features": SECRET}, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 2" in message
        assert '("bad_item")' in message
        assert "str" in message
        assert SECRET not in message

    def test_non_object_item_reports_type_not_value(self) -> None:
        """A raw list item keeps ValueError and must not echo the item, which can carry secrets."""
        items = [OK_ITEM_WITH_SECRET, [SECRET], "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 2" in message
        assert 'feature config item 2 ("' not in message
        assert "must be a feature name string or an object" in message
        assert "got list" in message
        assert SECRET not in message

    def test_nested_error_carries_field_path_and_item_prefix(self) -> None:
        """A nested in_features error names the field path between the item prefix and the message."""
        nested = {"name": "outer", "options": {"in_features": {"name": [SECRET]}}}
        items = [nested, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 1" in message
        assert '("outer")' in message
        assert "options.in_features" in message
        assert SECRET not in message

    def test_nested_unknown_field_keeps_type_error_and_gets_path(self) -> None:
        """An unknown field in a nested dict keeps the native TypeError and still names item and path."""
        nested = {"name": "outer", "options": {"in_features": {"name": "inner", "bad_field": 1}}}
        items = [nested, "plain_feature"]

        message = _load_expect_error(items, TypeError)

        assert "feature config item 1" in message
        assert '("outer")' in message
        assert "options.in_features" in message
        assert "bad_field" in message

    def test_loader_stage_collision_names_item(self) -> None:
        """An error raised while building features from a parsed item gets the item prefix too."""
        items = [{"name": "collider", "in_features": ["a"], "options": {"in_features": ["b"]}}]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 1" in message
        assert '("collider")' in message

    def test_valid_config_with_secret_option_still_loads(self) -> None:
        """A valid item whose options carry a secret loads fine: reporting scopes to rejected values only."""
        items = [OK_ITEM_WITH_SECRET, "plain_feature"]

        result = load_features_from_config(_config(items))

        assert len(result) == 2
        first = result[0]
        assert isinstance(first, Feature)
        assert first.name == "ok_with_secret"
        assert first.options.get("api_key") == SECRET
        assert result[1] == "plain_feature"
