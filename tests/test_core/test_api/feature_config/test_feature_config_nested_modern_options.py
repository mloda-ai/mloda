"""Nested in_features dicts work inside group_options/context_options (issue #681).

The loader has three construction branches. The 'options' branch and the
'in_features' branch both run process_nested_features first, so a nested feature
dict under an 'in_features' key becomes a Feature. The group_options/context_options
branch builds Options straight from the raw parsed dicts and never processes them,
so the very same nested dict stays an inert plain dict: it is neither converted nor
validated.

These tests pin the modern option style at parity with the legacy one: a nested
in_features dict becomes a Feature at any depth in group_options and in
context_options, gets the same validation there (unknown key, top-level-only key,
empty name, feature dict as a list element), and lands in Options.group or
Options.context according to the container it came from.

They also pin the collision the parity opens up: the top-level 'in_features' field
is injected into context by the loader, so an 'in_features' key written inside
group_options (cryptic duplicate-key error today) or inside context_options
(silently overwritten today) must raise one clear error naming the key and the
container. And the loader must leave the parsed config dicts unmutated.
"""

import json
from typing import Any

import pytest

from mloda.core.api.feature_config import loader as loader_module
from mloda.core.api.feature_config.loader import load_features_from_config
from mloda.core.api.feature_config.models import FeatureConfig, FeatureConfigItem
from mloda.core.api.feature_config.parser import parse_json
from mloda.provider import DefaultOptionKeys
from mloda.user import Feature


CONTAINERS = ["group_options", "context_options"]


def _single_feature(config: dict[str, Any]) -> Feature:
    """Load a one-item config and return its Feature."""
    result = load_features_from_config(json.dumps([config]), format="json")

    assert len(result) == 1
    feature = result[0]
    assert isinstance(feature, Feature)
    return feature


def _container_options(feature: Feature, container: str) -> dict[str, Any]:
    """Return the Options dict a container writes into: group_options -> group, context_options -> context."""
    return feature.options.group if container == "group_options" else feature.options.context


# ---------------------------------------------------------------------------
# Parity: a nested in_features dict becomes a Feature in both modern containers
# ---------------------------------------------------------------------------


def test_issue_repro_group_options_nested_in_features_becomes_feature() -> None:
    """Issue #681 repro: the nested dict under group_options becomes a Feature, not a raw dict."""
    config_str = json.dumps(
        [{"name": "outer", "group_options": {"in_features": {"name": "inner", "feature_group": "InnerFG"}}}]
    )

    result = load_features_from_config(config_str, format="json")

    assert len(result) == 1
    outer = result[0]
    assert isinstance(outer, Feature)

    nested = outer.options.group.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "inner"
    assert nested.feature_group_scope == "InnerFG"


def test_context_options_nested_in_features_becomes_feature() -> None:
    """The same nested dict under context_options becomes a Feature and stays in context."""
    config: dict[str, Any] = {
        "name": "outer",
        "context_options": {"in_features": {"name": "inner", "feature_group": "InnerFG"}},
    }

    outer = _single_feature(config)

    nested = outer.options.context.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "inner"
    assert nested.feature_group_scope == "InnerFG"
    assert "in_features" not in outer.options.group


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_feature_keeps_its_own_options(container: str) -> None:
    """The nested Feature carries its own options, exactly as under the legacy 'options' key."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {
            "threshold": 0.5,
            "in_features": {"name": "inner", "options": {"aggregation_function": "max"}},
        },
    }

    outer = _single_feature(config)

    options = _container_options(outer, container)
    assert options.get("threshold") == 0.5
    nested = options.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "inner"
    assert nested.options.group.get("aggregation_function") == "max"


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_feature_two_levels_deep(container: str) -> None:
    """The recursion builds the whole nested Feature chain, not just the first level."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {
            "in_features": {
                "name": "middle",
                "options": {"in_features": {"name": "inner", "options": {"scaler_type": "minmax"}}},
            }
        },
    }

    outer = _single_feature(config)

    middle = _container_options(outer, container).get("in_features")
    assert isinstance(middle, Feature)
    assert middle.name == "middle"

    inner = middle.options.group.get("in_features")
    assert isinstance(inner, Feature)
    assert inner.name == "inner"
    assert inner.options.group.get("scaler_type") == "minmax"


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_feature_below_a_plain_key(container: str) -> None:
    """An in_features dict nested below a plain option key is processed too: the walk is recursive."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {"reader_cfg": {"in_features": {"name": "inner"}}},
    }

    outer = _single_feature(config)

    reader_cfg = _container_options(outer, container).get("reader_cfg")
    assert isinstance(reader_cfg, dict)
    nested = reader_cfg.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "inner"


def test_group_and_context_options_both_process_nested_features() -> None:
    """Both containers are processed in the same load, each keeping its own nested Feature."""
    config: dict[str, Any] = {
        "name": "outer",
        "group_options": {"in_features": {"name": "group_inner"}},
        "context_options": {"reader_cfg": {"in_features": {"name": "context_inner"}}},
    }

    outer = _single_feature(config)

    group_nested = outer.options.group.get("in_features")
    assert isinstance(group_nested, Feature)
    assert group_nested.name == "group_inner"

    reader_cfg = outer.options.context.get("reader_cfg")
    assert isinstance(reader_cfg, dict)
    context_nested = reader_cfg.get("in_features")
    assert isinstance(context_nested, Feature)
    assert context_nested.name == "context_inner"


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_feature_is_reachable_via_get_in_features(container: str) -> None:
    """Options.get_in_features resolves the nested Feature from group as well as from context."""
    config: dict[str, Any] = {"name": "outer", container: {"in_features": {"name": "inner"}}}

    outer = _single_feature(config)

    in_features = outer.options.get_in_features()
    assert {feature.name for feature in in_features} == {"inner"}


# ---------------------------------------------------------------------------
# Parity: the nested dict gets the same validation as under 'options'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_rejects_unknown_key(container: str) -> None:
    """An unknown key in the nested dict raises a TypeError naming the key, as under 'options'."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {"in_features": {"name": "inner", "feature_grp": "ClaimsReader"}},
    }

    with pytest.raises(TypeError, match="feature_grp"):
        load_features_from_config(json.dumps([config]), format="json")


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_rejects_unknown_key_two_levels_deep(container: str) -> None:
    """The unknown-key rejection reaches a nested feature's own nested feature."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {
            "in_features": {
                "name": "middle",
                "options": {"in_features": {"name": "inner", "feature_grp": "ClaimsReader"}},
            }
        },
    }

    with pytest.raises(TypeError, match="feature_grp"):
        load_features_from_config(json.dumps([config]), format="json")


TOP_LEVEL_ONLY_FIELDS: list[tuple[str, Any]] = [
    ("column_index", 1),
    ("group_options", {"threshold": 0.5}),
    ("context_options", {"metadata": "test"}),
    ("propagate_context_keys", ["metadata"]),
]


@pytest.mark.parametrize("container", CONTAINERS)
@pytest.mark.parametrize("key, value", TOP_LEVEL_ONLY_FIELDS)
def test_modern_options_nested_rejects_top_level_only_field(container: str, key: str, value: Any) -> None:
    """A top-level-only field inside the nested dict is rejected instead of silently dropped."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {"in_features": {"name": "inner", key: value}},
    }

    with pytest.raises(ValueError, match=key) as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    assert "top level" in str(exc_info.value)


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_rejects_missing_name(container: str) -> None:
    """A nested dict without 'name' is rejected instead of becoming an unusable raw dict."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {"in_features": {"options": {"scaler_type": "minmax"}}},
    }

    with pytest.raises(ValueError, match="name"):
        load_features_from_config(json.dumps([config]), format="json")


@pytest.mark.parametrize("container", CONTAINERS)
@pytest.mark.parametrize("name", ["", "   "])
def test_modern_options_nested_rejects_empty_name(container: str, name: str) -> None:
    """An empty or whitespace-only nested 'name' names no feature and is rejected."""
    config: dict[str, Any] = {"name": "outer", container: {"in_features": {"name": name}}}

    with pytest.raises(ValueError, match="name"):
        load_features_from_config(json.dumps([config]), format="json")


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_rejects_feature_dict_as_list_element(container: str) -> None:
    """A feature dict inside a nested sibling in_features list is rejected, as under 'options'."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {"in_features": {"name": "middle", "in_features": [{"name": "inner"}]}},
    }

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    message = str(exc_info.value)
    assert "dict" in message
    assert "list" in message


@pytest.mark.parametrize("container", CONTAINERS)
def test_modern_options_nested_sibling_in_features_list_still_loads(container: str) -> None:
    """The valid sibling in_features list survives the modern containers unchanged."""
    config: dict[str, Any] = {
        "name": "outer",
        container: {"in_features": {"name": "middle", "in_features": ["age", "weight"]}},
    }

    outer = _single_feature(config)

    middle = _container_options(outer, container).get("in_features")
    assert isinstance(middle, Feature)
    assert middle.options.group.get("in_features") == ["age", "weight"]


# ---------------------------------------------------------------------------
# Collision: the top-level 'in_features' field vs an 'in_features' container key
#
# The loader injects the top-level field into context[DefaultOptionKeys.in_features].
# Combining it with an 'in_features' key inside group_options gives a cryptic
# duplicate-key error from OptionsValidator today; inside context_options the
# user's value is silently overwritten. One clear error must name the key and the
# container it was written into.
# ---------------------------------------------------------------------------


COLLIDING_VALUES: list[Any] = [
    pytest.param({"name": "inner"}, id="nested_feature_dict"),
    pytest.param(["weight"], id="list_of_names"),
    pytest.param("weight", id="single_name"),
]


@pytest.mark.parametrize("container", CONTAINERS)
@pytest.mark.parametrize("value", COLLIDING_VALUES)
def test_top_level_in_features_collides_with_container_in_features(container: str, value: Any) -> None:
    """The top-level in_features field must not be combined with an in_features key in a container."""
    config: dict[str, Any] = {
        "name": "outer",
        "in_features": ["age"],
        container: {"in_features": value},
    }

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    # Naming the container is what makes the error actionable: the cryptic
    # "Keys cannot exist in both group and context" does not say where to look.
    assert container in str(exc_info.value)


def test_top_level_in_features_collision_replaces_the_cryptic_duplicate_key_error() -> None:
    """The collision is reported by the loader, not by the group/context duplicate-key check."""
    config: dict[str, Any] = {
        "name": "outer",
        "in_features": ["age"],
        "group_options": {"in_features": {"name": "inner"}},
    }

    with pytest.raises(ValueError, match="in_features") as exc_info:
        load_features_from_config(json.dumps([config]), format="json")

    message = str(exc_info.value)
    assert "group_options" in message
    assert "Keys cannot exist in both group and context" not in message


# ---------------------------------------------------------------------------
# The loader leaves the parsed config dicts alone
# ---------------------------------------------------------------------------


def _spy_on_parse_json(monkeypatch: pytest.MonkeyPatch) -> list[FeatureConfigItem]:
    """Capture the FeatureConfig items the loader receives, to inspect them after the load."""
    captured: list[FeatureConfigItem] = []

    def spy(config_str: str) -> list[FeatureConfigItem]:
        items = parse_json(config_str)
        captured.extend(items)
        return items

    monkeypatch.setattr(loader_module, "parse_json", spy)
    return captured


def test_loader_does_not_write_in_features_into_parsed_context_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-level in_features field lands in Options.context without being written into the parsed dict."""
    captured = _spy_on_parse_json(monkeypatch)

    config: dict[str, Any] = {
        "name": "outer",
        "in_features": ["age"],
        "context_options": {"metadata": "test"},
    }
    outer = _single_feature(config)

    assert outer.options.context.get(DefaultOptionKeys.in_features) == frozenset({"age"})

    parsed = captured[0]
    assert isinstance(parsed, FeatureConfig)
    assert parsed.context_options == {"metadata": "test"}


@pytest.mark.parametrize("container", CONTAINERS)
def test_loader_does_not_mutate_parsed_options_when_building_nested_features(
    container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processing produces new dicts: the parsed config keeps its raw nested dict."""
    captured = _spy_on_parse_json(monkeypatch)

    config: dict[str, Any] = {"name": "outer", container: {"in_features": {"name": "inner"}}}
    outer = _single_feature(config)

    assert isinstance(_container_options(outer, container).get("in_features"), Feature)

    parsed = captured[0]
    assert isinstance(parsed, FeatureConfig)
    raw = parsed.group_options if container == "group_options" else parsed.context_options
    assert raw == {"in_features": {"name": "inner"}}


# ---------------------------------------------------------------------------
# Regression: every branch keeps its current behavior
# ---------------------------------------------------------------------------


def test_plain_group_and_context_options_land_verbatim() -> None:
    """No in_features key anywhere: group and context options are stored exactly as given."""
    config: dict[str, Any] = {
        "name": "outer",
        "group_options": {"threshold": 0.5, "method": "advanced"},
        "context_options": {"metadata": "test", "debug_mode": False},
    }

    outer = _single_feature(config)

    assert outer.options.group == {"threshold": 0.5, "method": "advanced"}
    assert outer.options.context == {"metadata": "test", "debug_mode": False}


@pytest.mark.parametrize("container", CONTAINERS)
def test_plain_nested_dict_without_in_features_key_stays_a_dict(container: str) -> None:
    """A nested dict that carries no in_features key is plain option data and stays a dict."""
    config: dict[str, Any] = {"name": "outer", container: {"reader_cfg": {"path": "/data", "sep": ","}}}

    outer = _single_feature(config)

    assert _container_options(outer, container) == {"reader_cfg": {"path": "/data", "sep": ","}}


@pytest.mark.parametrize("container", CONTAINERS)
def test_container_in_features_list_of_names_stays_a_list(container: str) -> None:
    """Without a top-level in_features field, an in_features list inside a container is kept verbatim."""
    config: dict[str, Any] = {"name": "outer", container: {"in_features": ["age", "weight"]}}

    outer = _single_feature(config)

    assert _container_options(outer, container).get("in_features") == ["age", "weight"]


def test_propagate_context_keys_still_threaded_into_options() -> None:
    """The group/context branch keeps threading propagate_context_keys into Options."""
    config: dict[str, Any] = {
        "name": "outer",
        "context_options": {"session_id": "abc"},
        "propagate_context_keys": ["session_id"],
    }

    outer = _single_feature(config)

    assert outer.options.context == {"session_id": "abc"}
    assert outer.options.propagate_context_keys == frozenset({"session_id"})


def test_top_level_in_features_with_group_options_still_lands_in_context() -> None:
    """The top-level in_features field keeps landing in context as a frozenset next to group options."""
    config: dict[str, Any] = {
        "name": "scale__age",
        "in_features": ["age"],
        "group_options": {"threshold": 0.5},
        "context_options": {"metadata": "test"},
    }

    outer = _single_feature(config)

    assert outer.options.group == {"threshold": 0.5}
    assert outer.options.context.get("metadata") == "test"
    assert outer.options.context.get(DefaultOptionKeys.in_features) == frozenset({"age"})


def test_column_index_suffix_still_applied_with_group_options() -> None:
    """column_index still suffixes the name on the group/context branch."""
    config: dict[str, Any] = {
        "name": "vectorized__text",
        "column_index": 2,
        "group_options": {"max_features": 100},
    }

    outer = _single_feature(config)

    assert outer.name == "vectorized__text~2"
    assert outer.options.group.get("max_features") == 100


def test_options_branch_nested_feature_still_loads() -> None:
    """The legacy 'options' branch keeps building nested Features."""
    config: dict[str, Any] = {
        "name": "outer",
        "options": {"scaler_type": "minmax", "in_features": {"name": "inner", "feature_group": "InnerFG"}},
    }

    outer = _single_feature(config)

    assert outer.options.group.get("scaler_type") == "minmax"
    nested = outer.options.group.get("in_features")
    assert isinstance(nested, Feature)
    assert nested.name == "inner"
    assert nested.feature_group_scope == "InnerFG"


def test_in_features_branch_with_options_still_loads() -> None:
    """The top-level in_features branch keeps putting options in group and the sources in context."""
    config: dict[str, Any] = {
        "name": "scale__age",
        "in_features": ["age"],
        "options": {"method": "standard"},
    }

    outer = _single_feature(config)

    assert outer.options.group == {"method": "standard"}
    assert outer.options.context.get(DefaultOptionKeys.in_features) == frozenset({"age"})


def test_options_branch_collision_with_top_level_in_features_still_raises() -> None:
    """An in_features key in 'options' beside the top-level field stays an error on the legacy branch too."""
    config: dict[str, Any] = {
        "name": "outer",
        "in_features": ["age"],
        "options": {"in_features": {"name": "inner"}},
    }

    with pytest.raises(ValueError, match="in_features"):
        load_features_from_config(json.dumps([config]), format="json")
