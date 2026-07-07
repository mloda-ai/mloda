"""Regression tests for #608: cyclic child_options must not recurse forever in Feature.__eq__/__hash__."""

from __future__ import annotations

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
