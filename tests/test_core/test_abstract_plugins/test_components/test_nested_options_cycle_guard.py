"""Cycle guards must thread through nested Options/HashableDict values instead of restarting there."""

from __future__ import annotations

from typing import Any

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
