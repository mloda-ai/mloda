"""Cycle guards must thread through nested Options/HashableDict values instead of restarting there."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.hashable_dict import HashableDict
from mloda.core.abstract_plugins.components.options import Options


def _options_cycle(marker: Any = None) -> Options:
    inner: dict[str, Any] = {}
    options = Options(group={"n": inner})
    # Back-reference first: the walk must survive it before it ever reaches the marker.
    inner["back"] = options
    inner["marker"] = marker
    return options


def _hashable_dict_cycle(marker: Any = None) -> HashableDict:
    inner: dict[str, Any] = {}
    hashable = HashableDict({"n": inner})
    inner["back"] = hashable
    inner["marker"] = marker
    return hashable


def _cycle_through_nested_hashable_dict(marker: Any = None) -> Options:
    return Options(group={"k": _hashable_dict_cycle(marker)})


def _hashable_dict_cycle_via_options(marker: Any = None) -> HashableDict:
    inner: dict[str, Any] = {}
    hashable = HashableDict({"n": inner})
    inner["back"] = Options(group={"o": hashable})
    inner["marker"] = marker
    return hashable


def _feature_with_child_option(value: Any) -> Feature:
    feature = Feature(name="y", options={})
    feature.child_options = Options(group={"k": value})
    return feature


class InheritingOptions(Options):
    """Subclass adding nothing: both dunders stay Options'."""


class InheritingHashableDict(HashableDict):
    """Subclass adding nothing: both dunders stay HashableDict's."""


class DisagreeingOptions(Options):
    """Subclass whose own __eq__ refuses every comparison."""

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return Options.__hash__(self)


class DisagreeingHashableDict(HashableDict):
    """Subclass whose own __eq__ refuses every comparison."""

    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return HashableDict.__hash__(self)


def _acyclic_unrolled(marker: Any = None) -> Options:
    """Same keys and lengths as _options_cycle, but the back-reference terminates."""
    leaf = Options(group={"n": {"back": None, "marker": marker}})
    return Options(group={"n": {"back": leaf, "marker": marker}})


class TestHashingSurvivesCyclesThroughNestedNodes:
    def test_cycle_through_a_nested_options_hashes(self) -> None:
        assert isinstance(hash(_options_cycle()), int)

    def test_cycle_through_a_nested_hashable_dict_hashes(self) -> None:
        assert isinstance(hash(_cycle_through_nested_hashable_dict()), int)

    def test_hashable_dict_cycle_hashes(self) -> None:
        assert isinstance(hash(_hashable_dict_cycle()), int)

    def test_hashable_dict_cycle_routed_through_options_hashes(self) -> None:
        assert isinstance(hash(_hashable_dict_cycle_via_options()), int)

    def test_hash_is_stable_across_repeated_calls(self) -> None:
        options = _options_cycle()
        assert hash(options) == hash(options)

    def test_hashable_dict_hash_is_stable_across_repeated_calls(self) -> None:
        hashable = _hashable_dict_cycle()
        assert hash(hashable) == hash(hashable)

    def test_structurally_identical_option_cycles_hash_equal(self) -> None:
        assert hash(_options_cycle()) == hash(_options_cycle())

    def test_structurally_identical_hashable_dict_cycles_hash_equal(self) -> None:
        assert hash(_hashable_dict_cycle()) == hash(_hashable_dict_cycle())

    def test_structurally_identical_mixed_cycles_hash_equal(self) -> None:
        assert hash(_cycle_through_nested_hashable_dict()) == hash(_cycle_through_nested_hashable_dict())


class TestEqualityAcrossNestedNodeCycles:
    def test_independently_built_option_cycles_compare_equal(self) -> None:
        assert _options_cycle() == _options_cycle()

    def test_independently_built_hashable_dict_cycles_compare_equal(self) -> None:
        assert _hashable_dict_cycle() == _hashable_dict_cycle()

    def test_independently_built_mixed_cycles_compare_equal(self) -> None:
        assert _cycle_through_nested_hashable_dict() == _cycle_through_nested_hashable_dict()

    def test_equal_option_cycles_collapse_in_a_set(self) -> None:
        assert len({_options_cycle(), _options_cycle()}) == 1

    def test_equal_hashable_dict_cycles_collapse_in_a_set(self) -> None:
        assert len({_hashable_dict_cycle(), _hashable_dict_cycle()}) == 1

    def test_differing_markers_in_option_cycles_compare_unequal(self) -> None:
        assert _options_cycle(1) != _options_cycle(2)

    def test_differing_markers_in_hashable_dict_cycles_compare_unequal(self) -> None:
        assert _hashable_dict_cycle(1) != _hashable_dict_cycle(2)

    def test_a_cycle_does_not_equal_the_acyclic_unrolling(self) -> None:
        assert _options_cycle() != _acyclic_unrolled()
        assert _acyclic_unrolled() != _options_cycle()


class TestNestedNodeTypeIdentityPreserved:
    def test_a_nested_options_is_not_equal_to_a_plain_dict(self) -> None:
        assert Options(group={"k": Options(group={"a": 1})}) != Options(group={"k": {"a": 1}})
        assert Options(group={"k": {"a": 1}}) != Options(group={"k": Options(group={"a": 1})})

    def test_a_nested_options_is_not_equal_to_a_hashable_dict(self) -> None:
        assert Options(group={"k": Options(group={"a": 1})}) != Options(group={"k": HashableDict({"a": 1})})
        assert Options(group={"k": HashableDict({"a": 1})}) != Options(group={"k": Options(group={"a": 1})})

    def test_a_nested_hashable_dict_is_not_equal_to_a_plain_dict(self) -> None:
        assert Options(group={"k": HashableDict({"a": 1})}) != Options(group={"k": {"a": 1}})
        assert HashableDict({"k": HashableDict({"a": 1})}) != HashableDict({"k": {"a": 1}})


class TestAcyclicNestedNodesUnchanged:
    def test_equal_nested_options_compare_equal_and_hash_alike(self) -> None:
        left = Options(group={"k": Options(group={"a": [1, {"b": 2}]})})
        right = Options(group={"k": Options(group={"a": [1, {"b": 2}]})})

        assert left == right
        assert hash(left) == hash(right)

    def test_differing_nested_options_compare_unequal(self) -> None:
        assert Options(group={"k": Options(group={"a": 1})}) != Options(group={"k": Options(group={"a": 2})})

    def test_equal_nested_hashable_dicts_compare_equal_and_hash_alike(self) -> None:
        left = HashableDict({"k": HashableDict({"a": [1, {"b": 2}]})})
        right = HashableDict({"k": HashableDict({"a": [1, {"b": 2}]})})

        assert left == right
        assert hash(left) == hash(right)

    def test_differing_nested_hashable_dicts_compare_unequal(self) -> None:
        assert HashableDict({"k": HashableDict({"a": 1})}) != HashableDict({"k": HashableDict({"a": 2})})

    def test_nested_options_context_is_still_ignored(self) -> None:
        left = Options(group={"k": Options(group={"a": 1}, context={"c": "left"})})
        right = Options(group={"k": Options(group={"a": 1}, context={"c": "right"})})

        assert left == right
        assert hash(left) == hash(right)


class TestChildOptionsKeepsNestedNodeTypeIdentity:
    def test_a_hashable_dict_child_option_is_not_equal_to_a_plain_dict(self) -> None:
        assert _feature_with_child_option(HashableDict({"a": 1})) != _feature_with_child_option({"a": 1})
        assert _feature_with_child_option({"a": 1}) != _feature_with_child_option(HashableDict({"a": 1}))

    def test_a_hashable_dict_and_a_plain_dict_child_option_do_not_collapse_in_a_set(self) -> None:
        features = {_feature_with_child_option(HashableDict({"a": 1})), _feature_with_child_option({"a": 1})}
        assert len(features) == 2

    def test_an_options_child_option_is_not_equal_to_a_plain_dict(self) -> None:
        assert _feature_with_child_option(Options(group={"a": 1})) != _feature_with_child_option({"a": 1})
        assert _feature_with_child_option({"a": 1}) != _feature_with_child_option(Options(group={"a": 1}))

    def test_an_options_child_option_is_not_equal_to_a_hashable_dict(self) -> None:
        assert _feature_with_child_option(Options(group={"a": 1})) != _feature_with_child_option(HashableDict({"a": 1}))
        assert _feature_with_child_option(HashableDict({"a": 1})) != _feature_with_child_option(Options(group={"a": 1}))

    def test_an_options_and_a_hashable_dict_child_option_do_not_collapse_in_a_set(self) -> None:
        features = {
            _feature_with_child_option(Options(group={"a": 1})),
            _feature_with_child_option(HashableDict({"a": 1})),
        }
        assert len(features) == 2

    def test_equal_hashable_dict_child_options_compare_equal_and_hash_alike(self) -> None:
        left = _feature_with_child_option(HashableDict({"a": [1, {"b": 2}]}))
        right = _feature_with_child_option(HashableDict({"a": [1, {"b": 2}]}))

        assert left == right
        assert hash(left) == hash(right)

    def test_differing_hashable_dict_child_options_compare_unequal(self) -> None:
        assert _feature_with_child_option(HashableDict({"a": 1})) != _feature_with_child_option(HashableDict({"a": 2}))

    def test_a_cycle_in_a_hashable_dict_child_option_does_not_raise(self) -> None:
        feature = _feature_with_child_option(_hashable_dict_cycle())
        assert isinstance(hash(feature), int)
        assert feature == feature

    def test_a_cycle_in_an_options_child_option_does_not_raise(self) -> None:
        feature = _feature_with_child_option(_options_cycle())
        assert isinstance(hash(feature), int)
        assert feature == feature


class TestNodeSubclassesFollowTheirOwnDunders:
    def test_an_inheriting_options_subclass_is_walked_as_options(self) -> None:
        left = Options(group={"k": InheritingOptions(group={"a": 1})})
        right = Options(group={"k": Options(group={"a": 1})})

        assert left == right
        assert right == left
        assert hash(left) == hash(right)

    def test_an_inheriting_hashable_dict_subclass_is_walked_as_hashable_dict(self) -> None:
        left = Options(group={"k": InheritingHashableDict({"a": 1})})
        right = Options(group={"k": HashableDict({"a": 1})})

        assert left == right
        assert right == left
        assert hash(left) == hash(right)

    def test_an_options_subclass_overriding_eq_decides_for_itself(self) -> None:
        left = Options(group={"k": DisagreeingOptions(group={"a": 1})})
        right = Options(group={"k": DisagreeingOptions(group={"a": 1})})

        assert left != right

    def test_a_hashable_dict_subclass_overriding_eq_decides_for_itself(self) -> None:
        left = Options(group={"k": DisagreeingHashableDict({"a": 1})})
        right = Options(group={"k": DisagreeingHashableDict({"a": 1})})

        assert left != right
