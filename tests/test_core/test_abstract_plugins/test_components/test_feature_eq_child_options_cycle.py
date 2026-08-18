"""Regression tests for #608: cyclic child_options must not recurse forever in Feature.__eq__/__hash__."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


def _make_cyclic_feature(name: str) -> Feature:
    """Build a feature whose child_options in_features frozenset holds a value-equal copy, closing the cycle."""
    child = Feature(name)
    nested = Feature(name)
    child.child_options = Options(group={DefaultOptionKeys.in_features: frozenset({nested})})
    nested.child_options = Options(group={DefaultOptionKeys.in_features: frozenset({child})})
    return child


def test_feature_eq_with_cyclic_child_options_terminates() -> None:
    """Directly comparing two value-equal cyclic features must terminate, not recurse forever."""
    child = _make_cyclic_feature("src")
    other_child = _make_cyclic_feature("src")

    assert (child == other_child) is True


def test_feature_eq_cyclic_child_options_detects_inequality() -> None:
    """Cyclic features with different names must compare unequal without recursing forever."""
    child = _make_cyclic_feature("src")
    different = _make_cyclic_feature("other")

    assert (child == different) is False


def test_feature_hash_with_cyclic_child_options_terminates() -> None:
    """Hashing a cyclic feature must terminate and stay consistent with equality."""
    child = _make_cyclic_feature("src")
    other_child = _make_cyclic_feature("src")

    assert hash(child) == hash(other_child)


def test_cyclic_features_usable_as_set_and_dict_members() -> None:
    """Value-equal cyclic features must collapse in a set and be usable as dict keys."""
    child = _make_cyclic_feature("src")
    other_child = _make_cyclic_feature("src")

    collapsed = {child, other_child}
    assert len(collapsed) == 1

    mapping = {child: "value"}
    assert mapping[other_child] == "value"


def test_feature_eq_dict_nested_feature_cycle_terminates() -> None:
    """A Feature hidden inside a nested dict in child_options must not cause RecursionError (GAP A)."""
    child_a = Feature("child")
    nested_a = Feature("child")
    child_a.child_options = Options(group={"wrapper": {"src": nested_a}})
    nested_a.child_options = Options(group={"wrapper": {"src": child_a}})

    child_b = Feature("child")
    nested_b = Feature("child")
    child_b.child_options = Options(group={"wrapper": {"src": nested_b}})
    nested_b.child_options = Options(group={"wrapper": {"src": child_b}})

    assert child_a == child_b
    assert hash(child_a) == hash(child_b)
    assert len({child_a, child_b}) == 1


def test_feature_eq_options_nested_feature_cycle_terminates() -> None:
    """A Feature hidden inside a nested Options in child_options must not cause RecursionError (GAP A)."""
    child_a = Feature("child")
    nested_a = Feature("child")
    child_a.child_options = Options(
        group={"nested": Options(group={DefaultOptionKeys.in_features: frozenset({nested_a})})}
    )
    nested_a.child_options = Options(
        group={"nested": Options(group={DefaultOptionKeys.in_features: frozenset({child_a})})}
    )

    child_b = Feature("child")
    nested_b = Feature("child")
    child_b.child_options = Options(
        group={"nested": Options(group={DefaultOptionKeys.in_features: frozenset({nested_b})})}
    )
    nested_b.child_options = Options(
        group={"nested": Options(group={DefaultOptionKeys.in_features: frozenset({child_b})})}
    )

    assert child_a == child_b
    assert hash(child_a) == hash(child_b)
    assert len({child_a, child_b}) == 1


def test_feature_eq_nested_in_features_options_distinguished() -> None:
    """Same-named in_features children with different options must not collapse (GAP B)."""
    child_a = Feature("src", options={"variant": "A"})
    child_b = Feature("src", options={"variant": "B"})

    parent_a = Feature("consumer")
    parent_b = Feature("consumer")
    parent_a.child_options = Options(group={DefaultOptionKeys.in_features: frozenset({child_a})})
    parent_b.child_options = Options(group={DefaultOptionKeys.in_features: frozenset({child_b})})

    assert child_a != child_b
    assert parent_a != parent_b, "same-named in_features children with different options must not collapse"
    assert len({parent_a, parent_b}) == 2


def test_child_options_key_with_mixed_type_dict_keys_does_not_raise() -> None:
    """Feature._reduce's dict sort must tolerate mixed-type keys like _deep_hashable already does."""
    f = Feature("root")
    f.child_options = Options(group={"a": 1, 2: "b"})  # type: ignore[dict-item]  # mixed key types are the point

    result = f._child_options_key()

    assert result == f._child_options_key(), "repeated calls must be deterministic"


def test_child_options_key_with_nested_mixed_type_dict_keys_does_not_raise() -> None:
    """The mixed-type-key sort fallback must thread through recursion, not just the top level."""
    f = Feature("root")
    f.child_options = Options(group={"outer": {"a": 1, 2: "b"}})

    result = f._child_options_key()

    assert result == f._child_options_key(), "repeated calls must be deterministic"

    f2 = Feature("root")
    f2.child_options = Options(group={"outer": {2: "b", "a": 1}})

    assert result == f2._child_options_key(), "nested dict order must not affect the canonical key"


def test_child_options_key_with_mixed_type_dict_keys_matches_across_separate_dict_objects() -> None:
    f1 = Feature("root")
    f1.child_options = Options(group={"a": 1, 2: "b"})  # type: ignore[dict-item]  # mixed key types are the point

    f2 = Feature("root")
    f2.child_options = Options(group={2: "b", "a": 1})  # type: ignore[dict-item]  # mixed key types are the point

    assert f1._child_options_key() == f2._child_options_key()
    assert hash(f1) == hash(f2)


def test_child_options_key_with_int_bool_equal_keys_matches_across_separate_dict_objects() -> None:
    """1 and True are == and hash-equal; Feature._reduce's fallback sort must not split them apart."""
    f1 = Feature("root")
    f1.child_options = Options(group={1: "a", 2.5: "b", "s": "c"})  # type: ignore[dict-item]

    f2 = Feature("root")
    f2.child_options = Options(group={True: "a", 2.5: "b", "s": "c"})  # type: ignore[dict-item]

    assert f1._child_options_key() == f2._child_options_key()
    assert hash(f1) == hash(f2)


def test_feature_options_field_with_int_bool_equal_keys_hashes_consistently_with_equality() -> None:
    """Same bug reached through Feature.options (hashable_dict.py's fallback), not just child_options."""
    left = Feature("root", options={1: "a", 2.5: "b", "s": "c"})  # type: ignore[dict-item]
    right = Feature("root", options={True: "a", 2.5: "b", "s": "c"})  # type: ignore[dict-item]

    assert left == right
    assert hash(left) == hash(right)


def test_nested_feature_cycle_through_plain_container_still_recursionerrors() -> None:
    """Deliberately out-of-scope: a Feature reached only via a plain dict/list is hashed as a leaf with a fresh
    `seen`, so the cycle guard never fires here; pinned to catch accidental behavior changes during normalizer
    unification."""
    inner: dict[str, Feature] = {}
    o = Options(group={"n": inner})
    inner["back"] = Feature(name="z", options=o)

    with pytest.raises(RecursionError):
        hash(o)


def test_feature_eq_with_self_as_raw_child_options_key_does_not_raise() -> None:
    """A bare Feature used directly as a raw dict key must still compare via pure `==`, with no `hash(key)`
    call: a single-item dict never needs a comparison to sort, so `_reduce_dict_items` takes the native-order
    path and never touches the key's `__hash__`, which is not cycle-safe when re-entered outside `_reduce`'s
    own `seen` tracking. Unlike the sibling plain-container case above (Feature reached only through a
    container, the guard never applies there, RecursionError is accepted), this one must NOT raise."""
    f = Feature("root")
    f.child_options = Options(group={f: "v"})  # type: ignore[dict-item]  # a bare Feature used as a raw dict key

    assert f == f
