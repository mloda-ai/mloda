"""Cycle-safe equality for Options and HashableDict, with acyclic semantics unchanged."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.hashable_dict import HashableDict
from mloda.core.abstract_plugins.components.options import Options


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return 0


class _NeverEqual:
    def __eq__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 0


def _self_referential_list() -> list[Any]:
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    return cyclic


def _self_referential_dict() -> dict[str, Any]:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    return cyclic


class TestCyclicValuesCompareWithoutRecursing:
    def test_options_with_two_cyclic_lists_usable_as_dict_keys(self) -> None:
        left = Options(group={"k": _self_referential_list()})
        right = Options(group={"k": _self_referential_list()})

        mapping: dict[Options, int] = {left: 1}
        mapping[right] = 2

        assert left == right
        assert len(mapping) == 1

    def test_hashable_dict_with_two_cyclic_lists_usable_as_dict_keys(self) -> None:
        left = HashableDict({"k": _self_referential_list()})
        right = HashableDict({"k": _self_referential_list()})

        mapping: dict[HashableDict, int] = {left: 1}
        mapping[right] = 2

        assert left == right
        assert len(mapping) == 1

    def test_options_with_two_cyclic_dicts_compare_equal(self) -> None:
        assert Options(group={"k": _self_referential_dict()}) == Options(group={"k": _self_referential_dict()})

    def test_hashable_dict_with_two_cyclic_dicts_compare_equal(self) -> None:
        assert HashableDict({"k": _self_referential_dict()}) == HashableDict({"k": _self_referential_dict()})

    def test_mutually_referential_containers_compare_equal(self) -> None:
        def build() -> list[Any]:
            outer: list[Any] = []
            inner: dict[str, Any] = {"back": outer}
            outer.append(inner)
            return outer

        assert Options(group={"k": build()}) == Options(group={"k": build()})

    def test_cycle_nested_under_a_tuple_compares_equal(self) -> None:
        assert Options(group={"k": (_self_referential_list(),)}) == Options(group={"k": (_self_referential_list(),)})

    def test_differing_cycles_still_compare_unequal(self) -> None:
        left: list[Any] = [1]
        left.append(left)
        right: list[Any] = [2]
        right.append(right)

        assert Options(group={"k": left}) != Options(group={"k": right})

    def test_options_with_cyclic_content_usable_in_a_set(self) -> None:
        collapsed = {Options(group={"k": _self_referential_list()}), Options(group={"k": _self_referential_list()})}
        assert len(collapsed) == 1

    def test_hashable_dict_with_cyclic_content_usable_in_a_set(self) -> None:
        collapsed = {
            HashableDict({"k": _self_referential_list()}),
            HashableDict({"k": _self_referential_list()}),
        }
        assert len(collapsed) == 1


class TestAcyclicSemanticsUnchanged:
    def test_list_and_tuple_of_the_same_items_differ(self) -> None:
        assert Options(group={"k": [1, 2]}) != Options(group={"k": (1, 2)})

    def test_hashable_dict_list_and_tuple_of_the_same_items_differ(self) -> None:
        assert HashableDict({"k": [1, 2]}) != HashableDict({"k": (1, 2)})

    def test_equal_nested_structures_compare_equal(self) -> None:
        def build() -> dict[str, Any]:
            return {"a": [1, {"b": (2, [3])}], "c": {"d": ["e"]}}

        assert Options(group={"k": build()}) == Options(group={"k": build()})
        assert HashableDict(build()) == HashableDict(build())

    def test_differing_leaf_value_compares_unequal(self) -> None:
        assert Options(group={"k": {"a": [1, 2]}}) != Options(group={"k": {"a": [1, 3]}})

    def test_differing_length_compares_unequal(self) -> None:
        assert Options(group={"k": [1, 2]}) != Options(group={"k": [1, 2, 3]})
        assert Options(group={"k": {"a": 1}}) != Options(group={"k": {"a": 1, "b": 2}})

    def test_differing_keys_compare_unequal(self) -> None:
        assert Options(group={"k": {"a": 1}}) != Options(group={"k": {"b": 1}})

    def test_a_custom_eq_leaf_that_accepts_anything_decides(self) -> None:
        assert Options(group={"k": [_AlwaysEqual()]}) == Options(group={"k": ["anything"]})

    def test_a_custom_eq_leaf_that_rejects_everything_decides(self) -> None:
        assert Options(group={"k": [_NeverEqual()]}) != Options(group={"k": [_NeverEqual()]})

    def test_an_identical_leaf_short_circuits_before_its_custom_eq(self) -> None:
        """Element identity wins, as it does in CPython's own list and dict compare."""
        leaf = _NeverEqual()
        assert Options(group={"k": [leaf]}) == Options(group={"k": [leaf]})
        assert Options(group={"k": {"n": leaf}}) == Options(group={"k": {"n": leaf}})

    def test_sets_still_compare_by_plain_equality(self) -> None:
        assert Options(group={"k": {1, 2}}) == Options(group={"k": {2, 1}})
        assert Options(group={"k": {1, 2}}) != Options(group={"k": {1, 3}})

    def test_a_dict_subclass_is_not_walked_structurally(self) -> None:
        class _Subclass(dict[str, Any]):
            def __eq__(self, other: object) -> bool:
                return False

        assert Options(group={"k": _Subclass()}) != Options(group={"k": _Subclass()})

    def test_options_against_a_non_options_is_false(self) -> None:
        assert Options(group={"k": 1}) != {"k": 1}
        assert Options(group={"k": 1}).__eq__("not options") is False

    def test_hashable_dict_against_a_non_hashable_dict_is_false(self) -> None:
        assert HashableDict({"k": 1}) != {"k": 1}
        assert HashableDict({"k": 1}).__eq__("not a hashable dict") is False

    def test_options_equality_still_ignores_context(self) -> None:
        left = Options(group={"k": 1}, context={"c": "left"})
        right = Options(group={"k": 1}, context={"c": "right"})

        assert left == right

    def test_options_equality_still_ignores_context_for_cyclic_values(self) -> None:
        left = Options(group={"k": _self_referential_list()}, context={"c": "left"})
        right = Options(group={"k": _self_referential_list()}, context={"c": "right"})

        assert left == right

    @pytest.mark.parametrize("value", [1, "text", None, 3.5, True, b"bytes", frozenset({1})])
    def test_leaf_values_compare_as_before(self, value: Any) -> None:
        assert Options(group={"k": value}) == Options(group={"k": value})
