"""All Options rebuild paths share one provenance-preserving factory."""

from copy import copy, deepcopy
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.provider import FeatureGroup, PropertySpec
from mloda.user import Options


def test_rebuild_preserves_provenance_and_replaces_values() -> None:
    original = Options(group={"old": 1}, context={"session": "old"}, propagate_context_keys=frozenset({"session"}))
    original.inherited_group_keys = frozenset({"earlier", "latest"})
    original.inherited_context_keys = frozenset({"session"})
    original.last_forwarded_group_keys = frozenset({"latest"})
    group = {"new": [2]}
    context = {"session": "new"}

    rebuilt = original.rebuild(group=group, context=context)

    assert rebuilt is not original
    assert rebuilt.group == group
    assert rebuilt.context == context
    assert rebuilt.group["new"] is group["new"]
    assert rebuilt.inherited_group_keys == frozenset({"earlier", "latest"})
    assert rebuilt.inherited_context_keys == frozenset({"session"})
    assert rebuilt.last_forwarded_group_keys == frozenset({"latest"})
    assert rebuilt.propagate_context_keys == frozenset({"session"})
    assert original.group == {"old": 1}
    assert original.context == {"session": "old"}


@pytest.mark.parametrize(
    ("group", "context", "message"),
    [
        ({"session": "duplicate"}, {"session": "value"}, "both"),
        ({}, {}, "propagate_context_keys"),
    ],
)
def test_rebuild_retains_constructor_validation(group: dict[str, Any], context: dict[str, Any], message: str) -> None:
    original = Options(context={"session": "value"}, propagate_context_keys=frozenset({"session"}))

    with pytest.raises(ValueError, match=message):
        original.rebuild(group=group, context=context)


@pytest.mark.parametrize("deep", [False, True], ids=["shallow", "deep"])
def test_copying_rebuilt_options_preserves_copy_semantics(deep: bool) -> None:
    original = Options(context={"session": "value"}, propagate_context_keys=frozenset({"session"}))
    original.inherited_group_keys = frozenset({"items"})
    original.inherited_context_keys = frozenset({"session"})
    original.last_forwarded_group_keys = frozenset({"items"})
    rebuilt = original.rebuild(group={"items": [1]}, context=dict(original.context))

    copied = deepcopy(rebuilt) if deep else copy(rebuilt)

    assert copied.group is not rebuilt.group
    assert copied.context is not rebuilt.context
    assert (copied.group["items"] is rebuilt.group["items"]) is not deep
    assert copied.inherited_group_keys == rebuilt.inherited_group_keys
    assert copied.inherited_context_keys == rebuilt.inherited_context_keys
    assert copied.last_forwarded_group_keys == rebuilt.last_forwarded_group_keys
    assert copied.propagate_context_keys == rebuilt.propagate_context_keys
    copied.group["extra"] = 2
    copied.context["session"] = "changed"
    assert "extra" not in rebuilt.group
    assert rebuilt.context["session"] == "value"


@pytest.mark.parametrize("path", ["defaults", "bindings"])
def test_materialization_delegates_to_public_rebuild(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    original = Options(group={"keep": 1}, context={"session": "value"})
    result = Options(group={"factory_result": True})
    calls: list[tuple[Options, dict[str, Any], dict[str, Any]]] = []

    def rebuild(self: Options, group: dict[str, Any], context: dict[str, Any]) -> Options:
        calls.append((self, group, context))
        return result

    monkeypatch.setattr(Options, "rebuild", rebuild, raising=False)
    mapping = {
        "group_default": PropertySpec("Group option", context=False, default="group_value"),
        "context_default": PropertySpec("Context option", default="context_value"),
    }
    if path == "defaults":
        monkeypatch.setattr(FeatureGroup, "PROPERTY_MAPPING", mapping)
        effective = FeatureGroup.options_with_defaults(original)
    else:
        effective = FeatureChainParser._merge_bindings(
            original, {"group_default": "group_value", "context_default": "context_value"}, mapping
        )

    assert effective is result
    assert len(calls) == 1
    source, group, context = calls[0]
    assert source is original
    assert group == {"keep": 1, "group_default": "group_value"}
    assert context == {"session": "value", "context_default": "context_value"}
    assert original.group == {"keep": 1}
    assert original.context == {"session": "value"}
