"""Nested in_features dicts are validated like a FeatureConfig (issue #680).

The nested dict under an "in_features" key is built by hand in the loader: only
'name', 'options', 'in_features' and 'feature_group' are ever read, every other
key is silently dropped. A typo therefore degrades to a feature with no scope
instead of failing, while the very same typo at the TOP level is rejected by
FeatureConfig with a TypeError naming the key.

These tests pin FeatureConfig as the single validation path for both levels:
an unknown key is rejected at any nesting depth, a top-level-only field written
into a nested dict is rejected instead of ignored, 'name' must be a non-empty
string everywhere, and every nested shape that is valid today keeps producing
the same Feature objects.

They also pin the edges the first fix left open: a sibling 'in_features' list
holds source feature names only (a feature dict is supported as the direct value
of 'in_features', never as a list element, and it must not slip through the list
branch unvalidated), the top-level-only ValueError wins over the top-level
invariants of FeatureConfig.__post_init__ while an unknown key stays a TypeError,
and the missing-'name' error lists the keys present rather than dumping an
options payload that may carry credentials.

That last rule holds for every in_features error, not just the missing-'name' one:
a rejected element that is a DICT is named by its KEYS, so no option VALUE ever
reaches the message. Non-dict offenders stay repr'd: informative, payload-free.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from mloda.core.api.feature_config.loader import load_features_from_config, process_nested_features
from mloda.user import Feature


def _feature_named(features: list[Feature | str], name: str) -> Feature:
    """Return the loaded Feature with the given name, failing the test if it is absent."""
    for item in features:
        if isinstance(item, Feature) and item.name == name:
            return item
    raise AssertionError(f"Feature '{name}' not found in the loaded config")


# ---------------------------------------------------------------------------
# Unknown keys in a nested in_features dict
#
# The top level raises "unexpected keyword argument 'feature_grp'" from
# FeatureConfig; the nested dict must raise the same way instead of building a
# scope-less feature out of a typo.
# ---------------------------------------------------------------------------


def test_nested_in_features_rejects_unknown_key() -> None:
    """An unknown key in a nested in_features dict raises and names the key."""
    options: dict[str, Any] = {
        "in_features": {"name": "shared_token", "feature_grp": "ClaimsReader"},
    }

    with pytest.raises(TypeError, match="feature_grp"):
        process_nested_features(options)


def test_loader_nested_in_features_rejects_unknown_key() -> None:
    """Issue repro: the typo'd nested key is rejected through the JSON surface."""
    config_str = """[
        {
            "name": "outer",
            "options": {
                "in_features": {
                    "name": "shared_token",
                    "feature_grp": "ClaimsReader"
                }
            }
        }
    ]"""

    with pytest.raises(TypeError, match="feature_grp"):
        load_features_from_config(config_str, format="json")


def test_nested_in_features_rejects_unknown_key_two_levels_deep() -> None:
    """The recursion rejects an unknown key in a nested feature's own nested feature."""
    options: dict[str, Any] = {
        "in_features": {
            "name": "outer_token",
            "options": {
                "in_features": {"name": "inner_token", "feature_grp": "ClaimsReader"},
            },
        },
    }

    with pytest.raises(TypeError, match="feature_grp"):
        process_nested_features(options)


def test_nested_in_features_rejects_unknown_key_in_sibling_in_features_dict() -> None:
    """The sibling in_features-as-dict path rejects an unknown key too."""
    options: dict[str, Any] = {
        "in_features": {
            "name": "outer_token",
            "in_features": {"name": "inner_token", "feature_grp": "ClaimsReader"},
        },
    }

    with pytest.raises(TypeError, match="feature_grp"):
        process_nested_features(options)


def test_loader_nested_in_features_rejects_unknown_key_two_levels_deep() -> None:
    """The deep rejection surfaces through load_features_from_config as well."""
    config_str = """[
        {
            "name": "outer",
            "options": {
                "in_features": {
                    "name": "middle_token",
                    "options": {
                        "in_features": {
                            "name": "inner_token",
                            "feature_grp": "ClaimsReader"
                        }
                    }
                }
            }
        }
    ]"""

    with pytest.raises(TypeError, match="feature_grp"):
        load_features_from_config(config_str, format="json")


# ---------------------------------------------------------------------------
# Top-level-only fields inside a nested in_features dict
#
# These are keys FeatureConfig knows, but the nested path cannot honor them:
# today they are read by nobody and silently dropped. They must be rejected
# with a ValueError naming the offending key.
# ---------------------------------------------------------------------------

TOP_LEVEL_ONLY_FIELDS: list[tuple[str, Any]] = [
    ("column_index", 1),
    ("group_options", {"threshold": 0.5}),
    ("context_options", {"metadata": "test"}),
    ("propagate_context_keys", ["metadata"]),
]


@pytest.mark.parametrize("key, value", TOP_LEVEL_ONLY_FIELDS)
def test_nested_in_features_rejects_top_level_only_field(key: str, value: Any) -> None:
    """A top-level-only field in a nested dict raises a ValueError naming that key and the top-level rule.

    The 'top level' phrasing is what makes the error actionable: the fix is to move the key up, not to
    add the sibling key that a top-level invariant would ask for.
    """
    options: dict[str, Any] = {
        "in_features": {"name": "shared_token", key: value},
    }

    with pytest.raises(ValueError, match=key) as exc_info:
        process_nested_features(options)

    assert "top level" in str(exc_info.value)


@pytest.mark.parametrize("key, value", TOP_LEVEL_ONLY_FIELDS)
def test_loader_nested_in_features_rejects_top_level_only_field(key: str, value: Any) -> None:
    """The same rejection surfaces through the JSON surface for every top-level-only field."""
    config: dict[str, Any] = {
        "name": "outer",
        "options": {"in_features": {"name": "shared_token", key: value}},
    }

    with pytest.raises(ValueError, match=key) as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    assert "top level" in str(exc_info.value)


# ---------------------------------------------------------------------------
# The top-level-only ValueError wins over the top-level-semantics invariants
#
# FeatureConfig.__post_init__ enforces invariants that only make sense at the
# top level. Firing them for a nested dict gives misleading advice: following it
# (adding 'context_options', dropping 'options') just produces the next error.
# The "this key is only valid at the top level" message must win.
# ---------------------------------------------------------------------------


def test_nested_propagate_context_keys_reports_top_level_not_missing_context_options() -> None:
    """'propagate_context_keys' is top-level-only, so it must not be reported as needing 'context_options'."""
    options: dict[str, Any] = {
        "in_features": {"name": "shared_token", "propagate_context_keys": ["metadata"]},
    }

    with pytest.raises(ValueError, match="propagate_context_keys") as exc_info:
        process_nested_features(options)

    message = str(exc_info.value)
    assert "top level" in message
    assert "requires 'context_options'" not in message


def test_nested_group_options_beside_options_reports_top_level_not_options_conflict() -> None:
    """A nested dict with 'options' and 'group_options' names group_options as top-level-only."""
    options: dict[str, Any] = {
        "in_features": {
            "name": "shared_token",
            "options": {"aggregation_function": "max"},
            "group_options": {"threshold": 0.5},
        },
    }

    with pytest.raises(ValueError, match="group_options") as exc_info:
        process_nested_features(options)

    message = str(exc_info.value)
    assert "top level" in message
    assert "Cannot use both" not in message


MISLEADING_INVARIANT_DICTS: list[Any] = [
    pytest.param(
        {"name": "shared_token", "propagate_context_keys": ["metadata"]},
        "propagate_context_keys",
        id="propagate_context_keys_without_context_options",
    ),
    pytest.param(
        {"name": "shared_token", "options": {"aggregation_function": "max"}, "group_options": {"threshold": 0.5}},
        "group_options",
        id="group_options_beside_options",
    ),
    pytest.param(
        {"name": "shared_token", "options": {"aggregation_function": "max"}, "context_options": {"metadata": "test"}},
        "context_options",
        id="context_options_beside_options",
    ),
]


@pytest.mark.parametrize("nested, key", MISLEADING_INVARIANT_DICTS)
def test_loader_nested_top_level_only_field_wins_over_invariant(nested: dict[str, Any], key: str) -> None:
    """Through the JSON surface too, the top-level-only rule wins over the top-level invariants."""
    config: dict[str, Any] = {"name": "outer", "options": {"in_features": nested}}

    with pytest.raises(ValueError, match=key) as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    assert "top level" in str(exc_info.value)


@pytest.mark.parametrize("key, value", TOP_LEVEL_ONLY_FIELDS)
def test_nested_unknown_key_wins_over_top_level_only_field(key: str, value: Any) -> None:
    """An unknown key stays a TypeError even next to a top-level-only key: a typo is not a misplacement."""
    options: dict[str, Any] = {
        "in_features": {"name": "shared_token", "feature_grp": "ClaimsReader", key: value},
    }

    with pytest.raises(TypeError, match="feature_grp"):
        process_nested_features(options)


def test_loader_nested_unknown_key_wins_over_top_level_only_field() -> None:
    """The unknown-key TypeError beats the top-level-only ValueError through the JSON surface as well."""
    config: dict[str, Any] = {
        "name": "outer",
        "options": {"in_features": {"name": "shared_token", "feature_grp": "ClaimsReader", "column_index": 1}},
    }

    with pytest.raises(TypeError, match="feature_grp"):
        load_features_from_config(json.dumps([config]), format="json")


# ---------------------------------------------------------------------------
# 'name' is a non-empty string at both levels
# ---------------------------------------------------------------------------


def test_nested_in_features_rejects_missing_name() -> None:
    """A nested dict without 'name' raises a ValueError naming the missing field."""
    options: dict[str, Any] = {
        "in_features": {"options": {"scaler_type": "minmax"}},
    }

    with pytest.raises(ValueError, match="name"):
        process_nested_features(options)


def test_nested_in_features_missing_name_error_lists_keys_not_values() -> None:
    """The missing-'name' error names the keys present, never their values: options can carry credentials."""
    sensitive_value = "sk-live-51H8xQzZzZ-abcdef"
    options: dict[str, Any] = {
        "in_features": {
            "options": {"api_key": sensitive_value, "scaler_type": "minmax"},
            "feature_group": "ConfigScopeSourceA",
        },
    }

    with pytest.raises(ValueError, match="name") as exc_info:
        process_nested_features(options)

    message = str(exc_info.value)
    assert sensitive_value not in message
    assert "options" in message
    assert "feature_group" in message


def test_loader_nested_in_features_missing_name_error_lists_keys_not_values() -> None:
    """The same value-free error surfaces through the JSON surface."""
    sensitive_value = "sk-live-51H8xQzZzZ-abcdef"
    config: dict[str, Any] = {
        "name": "outer",
        "options": {"in_features": {"options": {"api_key": sensitive_value}}},
    }

    with pytest.raises(ValueError, match="name") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    message = str(exc_info.value)
    assert sensitive_value not in message
    assert "options" in message


def test_nested_in_features_rejects_empty_name() -> None:
    """An empty nested 'name' is a config mistake, not a feature called ''."""
    options: dict[str, Any] = {
        "in_features": {"name": ""},
    }

    with pytest.raises(ValueError, match="name"):
        process_nested_features(options)


def test_nested_in_features_rejects_whitespace_only_name() -> None:
    """A whitespace-only nested 'name' strips to nothing and must be rejected too."""
    options: dict[str, Any] = {
        "in_features": {"name": "   "},
    }

    with pytest.raises(ValueError, match="name"):
        process_nested_features(options)


def test_loader_nested_in_features_rejects_whitespace_only_name() -> None:
    """The whitespace-only nested name is rejected through the JSON surface as well."""
    config_str = """[
        {
            "name": "outer",
            "options": {
                "in_features": {"name": "   "}
            }
        }
    ]"""

    with pytest.raises(ValueError, match="name"):
        load_features_from_config(config_str, format="json")


# ---------------------------------------------------------------------------
# Regression: every nested shape that is valid today still loads unchanged
# ---------------------------------------------------------------------------


def test_nested_name_and_options_shape_still_loads() -> None:
    """Shape {name, options}: the nested Feature lands in the outer group options."""
    config_str = """[
        {
            "name": "outer",
            "options": {
                "scaler_type": "minmax",
                "in_features": {
                    "name": "shared_token",
                    "options": {"aggregation_function": "max"}
                }
            }
        }
    ]"""

    result = load_features_from_config(config_str, format="json")

    outer = _feature_named(result, "outer")
    assert outer.options.group.get("scaler_type") == "minmax"
    nested = outer.options.group.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "shared_token"
    assert nested.options.group.get("aggregation_function") == "max"
    assert nested.feature_group_scope is None


def test_nested_json_fixture_two_levels_deep_still_loads() -> None:
    """The 2-level deep 'min_max_nested' fixture keeps its Feature chain and its raw string leaf."""
    json_path = Path(__file__).parent / "test_config_features.json"
    config_str = json_path.read_text()

    result = load_features_from_config(config_str, format="json")

    outer = _feature_named(result, "min_max_nested")
    assert outer.options.group.get("scaler_type") == "minmax"

    level_one = outer.options.group.get("in_features")
    assert isinstance(level_one, Feature)
    assert level_one.name == "max_aggregated"
    assert level_one.options.group.get("aggregation_function") == "max"

    level_two = level_one.options.group.get("in_features")
    assert isinstance(level_two, Feature)
    assert level_two.name == "minmaxscaledweght"
    assert level_two.options.group.get("scaler_type") == "minmax"

    # The innermost leaf is a plain source name inside 'options', not a nested dict
    assert level_two.options.group.get("in_features") == "weight"


def test_nested_name_feature_group_and_options_shape_still_loads() -> None:
    """Shape {name, feature_group, options}: the scope survives on the nested Feature."""
    config_str = """[
        {
            "name": "outer",
            "options": {
                "in_features": {
                    "name": "shared_token",
                    "feature_group": "ConfigScopeSourceA",
                    "options": {"in_features": "age"}
                }
            }
        }
    ]"""

    result = load_features_from_config(config_str, format="json")

    nested = _feature_named(result, "outer").options.group.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "shared_token"
    assert nested.feature_group_scope == "ConfigScopeSourceA"
    assert nested.options.group.get("in_features") == "age"


def test_nested_name_and_feature_group_shape_still_loads() -> None:
    """Shape {name, feature_group}: a scoped nested feature with no options at all."""
    options: dict[str, Any] = {
        "in_features": {"name": "shared_token", "feature_group": "ConfigScopeSourceA"},
    }

    processed = process_nested_features(options)

    nested = processed["in_features"]
    assert isinstance(nested, Feature)
    assert nested.name == "shared_token"
    assert nested.feature_group_scope == "ConfigScopeSourceA"
    assert len(nested.options.group) == 0


def test_nested_sibling_in_features_dict_still_loads() -> None:
    """Shape {name, in_features}: a dict sibling becomes a Feature under the plain 'in_features' key."""
    options: dict[str, Any] = {
        "in_features": {
            "name": "outer_token",
            "in_features": {"name": "inner_token", "options": {"scaler_type": "minmax"}},
        },
    }

    processed = process_nested_features(options)

    outer = processed["in_features"]
    assert isinstance(outer, Feature)
    assert outer.name == "outer_token"
    inner = outer.options.group.get("in_features")
    assert isinstance(inner, Feature)
    assert inner.name == "inner_token"
    assert inner.options.group.get("scaler_type") == "minmax"


def test_nested_sibling_in_features_single_element_list_collapses_to_scalar() -> None:
    """Shape {name, in_features}: a 1-element list collapses to the bare element."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": ["age"]},
    }

    processed = process_nested_features(options)

    outer = processed["in_features"]
    assert isinstance(outer, Feature)
    assert outer.options.group.get("in_features") == "age"


def test_nested_sibling_in_features_multi_element_list_stays_a_list() -> None:
    """Shape {name, in_features}: a list with more than one entry is kept as a list."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": ["age", "weight"]},
    }

    processed = process_nested_features(options)

    outer = processed["in_features"]
    assert isinstance(outer, Feature)
    assert outer.options.group.get("in_features") == ["age", "weight"]


def test_nested_sibling_in_features_scalar_string_is_kept() -> None:
    """Shape {name, in_features}: a scalar source name is stored as-is."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": "age"},
    }

    processed = process_nested_features(options)

    outer = processed["in_features"]
    assert isinstance(outer, Feature)
    assert outer.options.group.get("in_features") == "age"


def test_loader_nested_sibling_in_features_list_still_loads() -> None:
    """The sibling in_features list survives the full JSON surface too."""
    config_str = """[
        {
            "name": "outer",
            "options": {
                "in_features": {
                    "name": "outer_token",
                    "in_features": ["age", "weight"]
                }
            }
        }
    ]"""

    result = load_features_from_config(config_str, format="json")

    nested = _feature_named(result, "outer").options.group.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "outer_token"
    assert nested.options.group.get("in_features") == ["age", "weight"]


# ---------------------------------------------------------------------------
# A sibling in_features LIST carries source feature names only
#
# The list branch of build_nested_feature stores its elements without looking at
# them, so a feature dict inside the list is neither validated nor converted: it
# stays a raw dict in the options. That is the #680 failure mode via the list
# path. A nested feature dict is supported as the direct VALUE of in_features
# only; a list holds non-empty source feature name strings.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_features",
    [
        pytest.param([{"name": "inner_token", "feature_grp": "ClaimsReader"}], id="single_element_list"),
        pytest.param(["age", {"name": "inner_token"}], id="multi_element_list"),
    ],
)
def test_nested_sibling_in_features_list_rejects_feature_dict_element(in_features: list[Any]) -> None:
    """A feature dict inside an in_features list is rejected instead of being stored as a raw dict."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": in_features},
    }

    with pytest.raises(ValueError, match="in_features") as exc_info:
        process_nested_features(options)

    message = str(exc_info.value)
    assert "dict" in message
    assert "list" in message


@pytest.mark.parametrize(
    "in_features",
    [
        pytest.param([1, 2], id="ints"),
        pytest.param(["age", 7], id="mixed_int"),
        pytest.param([None], id="null"),
        pytest.param([["age"]], id="inner_list"),
    ],
)
def test_nested_sibling_in_features_list_rejects_non_string_element(in_features: list[Any]) -> None:
    """A list element that is not a string is not a source feature name."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": in_features},
    }

    with pytest.raises(ValueError, match="in_features"):
        process_nested_features(options)


@pytest.mark.parametrize(
    "in_features",
    [
        pytest.param([""], id="empty_string"),
        pytest.param(["age", "   "], id="whitespace_only_string"),
    ],
)
def test_nested_sibling_in_features_list_rejects_empty_string_element(in_features: list[Any]) -> None:
    """An empty or whitespace-only list element names no feature and must be rejected."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": in_features},
    }

    with pytest.raises(ValueError, match="in_features"):
        process_nested_features(options)


@pytest.mark.parametrize(
    "in_features",
    [
        pytest.param(123, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_nested_sibling_in_features_rejects_non_string_scalar(in_features: Any) -> None:
    """A bare non-string scalar is stored as a 'source feature name' today; it is a config mistake."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": in_features},
    }

    with pytest.raises(ValueError, match="in_features"):
        process_nested_features(options)


@pytest.mark.parametrize(
    "in_features",
    [
        pytest.param("", id="empty_string"),
        pytest.param("   ", id="whitespace_only_string"),
        pytest.param([], id="empty_list"),
    ],
)
def test_nested_sibling_in_features_rejects_empty_value(in_features: Any) -> None:
    """An empty in_features is silently dropped today; an empty source declaration is a mistake."""
    options: dict[str, Any] = {
        "in_features": {"name": "outer_token", "in_features": in_features},
    }

    with pytest.raises(ValueError, match="in_features"):
        process_nested_features(options)


def test_loader_nested_sibling_in_features_list_rejects_feature_dict_element() -> None:
    """Issue repro through the list path: the typo'd feature dict inside the list is rejected."""
    config_str = """[
        {
            "name": "outer",
            "options": {
                "in_features": {
                    "name": "mid",
                    "in_features": [{"name": "a", "feature_grp": "Typo"}]
                }
            }
        }
    ]"""

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(config_str, format="json")

    message = str(exc_info.value)
    assert "dict" in message
    assert "list" in message


BAD_NESTED_SIBLING_IN_FEATURES: list[Any] = [
    pytest.param([{"name": "inner_token"}], id="list_with_feature_dict"),
    pytest.param(["age", {"name": "inner_token"}], id="multi_element_list_with_feature_dict"),
    pytest.param([1, 2], id="list_of_ints"),
    pytest.param([""], id="list_with_empty_string"),
    pytest.param([None], id="list_with_null"),
    pytest.param(123, id="bare_int"),
    pytest.param("", id="empty_string"),
    pytest.param([], id="empty_list"),
]


@pytest.mark.parametrize("in_features", BAD_NESTED_SIBLING_IN_FEATURES)
def test_loader_nested_sibling_in_features_rejects_invalid_value(in_features: Any) -> None:
    """Every invalid sibling in_features shape is rejected through the JSON surface."""
    config: dict[str, Any] = {
        "name": "outer",
        "options": {"in_features": {"name": "outer_token", "in_features": in_features}},
    }

    with pytest.raises(ValueError, match="in_features"):
        load_features_from_config(json.dumps([config]), format="json")


# ---------------------------------------------------------------------------
# An in_features error names an offending DICT by its keys, never by its repr
#
# A rejected feature dict is reported with {element!r} today, which dumps its whole
# 'options' payload into the error: a credential written there lands in logs and in
# an agent's context. The missing-'name' error already sets the norm by listing the
# keys present. Every in_features error follows it: a dict offender is named by its
# KEYS, a non-dict offender (string, int, None) is still repr'd, which is
# informative and carries no payload.
# ---------------------------------------------------------------------------

SENSITIVE_OPTION_VALUE = "hunter2-do-not-log"

# The offending element: a feature dict whose options carry a credential.
SENSITIVE_ELEMENT: dict[str, Any] = {
    "name": "inner_token",
    "feature_group": "ConfigScopeSourceA",
    "options": {"api_key": SENSITIVE_OPTION_VALUE},
}

PAYLOAD_CONTAINERS = ["options", "group_options", "context_options"]


def _assert_names_keys_not_payload(message: str) -> None:
    """The message names in_features and the offending element's keys, and leaks no option value."""
    assert SENSITIVE_OPTION_VALUE not in message
    assert "in_features" in message
    assert "feature_group" in message
    assert "options" in message


@pytest.mark.parametrize("container", PAYLOAD_CONTAINERS)
def test_nested_in_features_list_element_error_names_keys_not_payload(container: str) -> None:
    """The rejected list element is named by its keys: its options payload stays out of the error."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {"in_features": {"name": "outer_token", "in_features": [SENSITIVE_ELEMENT]}},
    }

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    _assert_names_keys_not_payload(str(exc_info.value))


def test_nested_in_features_list_element_error_keeps_repr_for_non_dict_offender() -> None:
    """A non-dict offender carries no payload, so it is still reported by its repr."""
    config: dict[str, Any] = {
        "name": "outer",
        "options": {"in_features": {"name": "outer_token", "in_features": ["age", 7]}},
    }

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    assert "7" in str(exc_info.value)


def test_top_level_in_features_array_element_error_names_keys_not_payload() -> None:
    """A feature dict inside the top-level in_features array is named by its keys too."""
    config: dict[str, Any] = {"name": "outer", "in_features": [SENSITIVE_ELEMENT]}

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    _assert_names_keys_not_payload(str(exc_info.value))


def test_top_level_in_features_dict_value_error_names_keys_not_payload() -> None:
    """A feature dict written as the whole top-level in_features value leaks the same payload today."""
    config: dict[str, Any] = {"name": "outer", "in_features": SENSITIVE_ELEMENT}

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    _assert_names_keys_not_payload(str(exc_info.value))
