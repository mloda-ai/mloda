"""Item-level error reporting in feature config loading: errors name the item and never echo rejected values."""

import json
from typing import Any

import pytest

from mloda.user import Feature
from mloda.core.abstract_plugins.components.property_values_validation import PropertyValidationError
from mloda.core.api.feature_config.loader import load_features_from_config
from mloda.core.api.feature_config.models import error_prefix


SECRET = "sk-test-1234567890abcdef"  # nosec B105

OK_ITEM_WITH_SECRET: dict[str, Any] = {"name": "ok_with_secret", "options": {"api_key": SECRET}}


def _config(items: list[Any]) -> str:
    return json.dumps(items)


def _load_expect_error(items: list[Any], expected_type: type[BaseException]) -> str:
    """Load the config and return the message of the raised error."""
    with pytest.raises(expected_type) as exc_info:
        load_features_from_config(_config(items))
    return str(exc_info.value)


class TestFeatureConfigItemErrorReporting:
    """Item errors name the failing config item without echoing rejected values."""

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
        assert "got list" in message
        assert SECRET not in message

    def test_top_level_in_features_string_reports_type_not_repr(self) -> None:
        """A bare string top-level in_features keeps ValueError but must not be echoed via repr."""
        items = [OK_ITEM_WITH_SECRET, {"name": "bad_item", "in_features": SECRET}, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 2" in message
        assert '("bad_item")' in message
        assert "got str" in message
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

    @pytest.mark.parametrize(
        ("container", "path"),
        [
            ("options", "options.in_features:"),
            ("group_options", "group_options.x.in_features:"),
            ("context_options", "context_options.in_features:"),
        ],
    )
    def test_nested_error_carries_field_path_and_item_prefix(self, container: str, path: str) -> None:
        """A nested in_features error names the field path between the item prefix and the message."""
        inner = {"in_features": {"name": [SECRET]}}
        nested = {"name": "outer", container: {"x": inner} if container == "group_options" else inner}
        items = [nested, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 1" in message
        assert '("outer")' in message
        assert path in message
        assert SECRET not in message

    @pytest.mark.parametrize(
        ("container", "path"),
        [
            ("options", "options.in_features:"),
            ("group_options", "group_options.in_features:"),
            ("context_options", "context_options.in_features:"),
        ],
    )
    @pytest.mark.parametrize(
        ("bad", "description"),
        [
            ([SECRET, 7_654_321], "got element at index 1: int"),
            (7_654_321, "got int"),
            ([SECRET, [SECRET]], "got element at index 1: list"),
        ],
        ids=["list_with_int", "scalar_int", "list_with_list"],
    )
    def test_non_string_in_features_value_is_rejected_at_load(
        self, container: str, path: str, bad: Any, description: str
    ) -> None:
        items = [{"name": "outer", container: {"in_features": bad}}, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert "feature config item 1" in message
        assert '("outer")' in message
        assert path in message
        assert description in message
        assert SECRET not in message
        assert "7654321" not in message

    def test_nested_group_non_string_in_features_is_rejected_at_load(self) -> None:
        items = [{"name": "outer", "group_options": {"x": {"in_features": [SECRET, 7_654_321]}}}]

        message = _load_expect_error(items, ValueError)

        assert "group_options.x.in_features:" in message
        assert "got element at index 1: int" in message
        assert SECRET not in message
        assert "7654321" not in message

    @pytest.mark.parametrize("value", ["src", ["a", "b"]], ids=["str", "list_of_names"])
    def test_valid_string_or_name_list_in_features_still_loads(self, value: Any) -> None:
        features = load_features_from_config(_config([{"name": "x", "options": {"in_features": value}}]))

        assert len(features) == 1

    def test_option_key_with_newline_and_quote_is_escaped_in_path(self) -> None:
        """A non-identifier option key renders as a JSON-escaped bracket segment with no raw newline."""
        key = 'a\nb"c'
        items = [{"name": "outer", "options": {key: {"in_features": {"name": [SECRET]}}}}]

        message = _load_expect_error(items, ValueError)

        assert "\n" not in message
        assert f"options[{json.dumps(key)}].in_features:" in message
        assert SECRET not in message

    def test_dotted_option_key_renders_in_brackets(self) -> None:
        """A key containing a dot is not an identifier and renders as options["a.b"]."""
        items = [{"name": "outer", "options": {"a.b": {"in_features": {"name": [SECRET]}}}}]

        message = _load_expect_error(items, ValueError)

        assert 'options["a.b"].in_features:' in message
        assert SECRET not in message

    def test_bad_column_index_names_item_without_value(self) -> None:
        """A rejected column_index names the item and never echoes the value."""
        items = [OK_ITEM_WITH_SECRET, {"name": "bad_item", "column_index": SECRET}, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert 'feature config item 2 ("bad_item")' in message
        assert "'column_index' must be a non-negative integer, got str" in message
        assert SECRET not in message

    @pytest.mark.parametrize(("text", "kind"), [("", "an empty string"), ("  \t", "a blank string")])
    def test_empty_or_blank_string_item_is_rejected(self, text: str, kind: str) -> None:
        """An empty or whitespace-only string item names its position and the kind of offender."""
        items = [OK_ITEM_WITH_SECRET, text, "plain_feature"]

        message = _load_expect_error(items, ValueError)

        assert f"feature config item 2: a feature name string must be non-empty, got {kind}" in message

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

    @pytest.mark.parametrize(
        "item",
        [
            pytest.param({"name": "x", "options": None}, id="top"),
            pytest.param({"name": "x", "options": None, "group_options": {"a": 1}}, id="with_group_options"),
            pytest.param({"name": "x", "options": {"in_features": {"name": "inner", "options": None}}}, id="nested"),
        ],
    )
    def test_null_options_loads_as_absent(self, item: dict[str, Any]) -> None:
        """A null options value is treated as absent."""
        result = load_features_from_config(_config([item]))

        assert len(result) == 1

    def test_top_level_bad_compute_framework_names_item_without_value(self) -> None:
        """A rejected options.compute_framework names the item and never echoes the value."""
        items = [OK_ITEM_WITH_SECRET, {"name": "bad_item", "options": {"compute_framework": SECRET}}]

        message = _load_expect_error(items, ValueError)

        assert 'feature config item 2 ("bad_item")' in message
        assert SECRET not in message

    def test_deep_nested_error_path_appears_once(self) -> None:
        """A two-level nested in_features error carries its dotted path exactly once."""
        deep = {"name": "deep", "options": {"compute_framework": SECRET}}
        mid = {"name": "mid", "in_features": deep}
        items = [{"name": "outer", "options": {"in_features": mid}}]

        message = _load_expect_error(items, ValueError)

        assert message.count("options.in_features.in_features:") == 1
        assert 'feature config item 1 ("outer")' in message
        assert SECRET not in message

    def test_name_with_quote_and_newline_is_escaped(self) -> None:
        """A name holding a quote and a newline renders escaped, with no raw newline in the message."""
        name = 'bad"\nname'
        items = [{"name": name, "optionz": {}}]

        message = _load_expect_error(items, TypeError)

        assert "\n" not in message
        assert 'feature config item 1 ("bad\\"\\nname")' in message

    def test_long_name_is_truncated_in_prefix(self) -> None:
        """A 200 char name is cut to 100 chars plus an ASCII ellipsis in the prefix."""
        name = "n" * 200
        items = [{"name": name, "optionz": {}}]

        message = _load_expect_error(items, TypeError)

        assert f'("{"n" * 100}...")' in message
        assert "n" * 101 not in message


def test_error_prefix_reraises_same_exception_object() -> None:
    """error_prefix prefixes the message but keeps the exception object and its attributes."""
    original = PropertyValidationError("k", "reason")

    with pytest.raises(PropertyValidationError) as exc_info:
        with error_prefix("pfx"):
            raise original

    assert exc_info.value is original
    assert original.key == "k"
    assert original.reason == "reason"
    assert str(original).startswith("pfx: ")
