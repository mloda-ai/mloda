"""Cycle-safe equality for Options, HashableDict and Feature, with acyclic semantics unchanged."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
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


def _two_node_cycle() -> list[Any]:
    outer: list[Any] = []
    inner: list[Any] = [outer]
    outer.append(inner)
    return outer


def _mutually_referential() -> list[Any]:
    outer: list[Any] = []
    inner: dict[str, Any] = {"back": outer}
    outer.append(inner)
    return outer


def _acyclic_nested() -> dict[str, Any]:
    return {"a": [1, {"b": (2, [3])}], "c": {"d": ["e"]}}


class _PlainDictSubclass(dict[str, Any]):
    """A dict subclass that inherits dict equality."""


class _PlainListSubclass(list[Any]):
    """A list subclass that inherits list equality."""


def _self_referential_dict_subclass() -> _PlainDictSubclass:
    cyclic = _PlainDictSubclass()
    cyclic["self"] = cyclic
    return cyclic


def _self_referential_list_subclass() -> _PlainListSubclass:
    cyclic = _PlainListSubclass()
    cyclic.append(cyclic)
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


class TestFeatureEqualityWithCyclicContext:
    """Feature.__eq__ compares context, which __hash__ excludes, so the probe must survive cycles."""

    @staticmethod
    def _feature_with_cyclic_context() -> Feature:
        return Feature(name="x", options=Options(group={"g": 1}, context={"c": _self_referential_list()}))

    def test_features_with_separately_built_cyclic_contexts_collapse_in_a_set(self) -> None:
        collapsed = {self._feature_with_cyclic_context(), self._feature_with_cyclic_context()}
        assert len(collapsed) == 1

    def test_features_with_separately_built_cyclic_contexts_compare_equal(self) -> None:
        assert self._feature_with_cyclic_context() == self._feature_with_cyclic_context()

    def test_features_with_separately_built_cyclic_group_and_context_compare_equal(self) -> None:
        def build() -> Feature:
            return Feature(
                name="x",
                options=Options(group={"g": _self_referential_list()}, context={"c": _self_referential_dict()}),
            )

        assert build() == build()
        assert len({build(), build()}) == 1

    def test_features_with_differing_cyclic_contexts_compare_unequal(self) -> None:
        def build(marker: int) -> Feature:
            cyclic: list[Any] = [marker]
            cyclic.append(cyclic)
            return Feature(name="x", options=Options(group={"g": 1}, context={"c": cyclic}))

        assert build(1) != build(2)

    def test_features_with_differing_acyclic_contexts_compare_unequal(self) -> None:
        left = Feature(name="x", options=Options(group={"g": 1}, context={"c": [1, 2]}))
        right = Feature(name="x", options=Options(group={"g": 1}, context={"c": [1, 3]}))

        assert left != right

    def test_features_with_differing_context_keys_compare_unequal(self) -> None:
        left = Feature(name="x", options=Options(group={"g": 1}, context={"a": _self_referential_list()}))
        right = Feature(name="x", options=Options(group={"g": 1}, context={"b": _self_referential_list()}))

        assert left != right


class TestCyclicContainerSubclasses:
    """dict/list subclasses without a custom __eq__ normalize for hashing, so equality must walk them too."""

    def test_options_with_two_cyclic_dict_subclasses_collapse_in_a_set(self) -> None:
        left = Options(group={"k": _self_referential_dict_subclass()})
        right = Options(group={"k": _self_referential_dict_subclass()})

        assert len({left, right}) == 1
        assert left == right

    def test_options_with_two_cyclic_list_subclasses_collapse_in_a_set(self) -> None:
        left = Options(group={"k": _self_referential_list_subclass()})
        right = Options(group={"k": _self_referential_list_subclass()})

        assert len({left, right}) == 1
        assert left == right

    def test_hashable_dict_with_two_cyclic_dict_subclasses_collapse_in_a_set(self) -> None:
        left = HashableDict({"k": _self_referential_dict_subclass()})
        right = HashableDict({"k": _self_referential_dict_subclass()})

        assert len({left, right}) == 1
        assert left == right

    def test_hashable_dict_with_two_cyclic_list_subclasses_collapse_in_a_set(self) -> None:
        left = HashableDict({"k": _self_referential_list_subclass()})
        right = HashableDict({"k": _self_referential_list_subclass()})

        assert len({left, right}) == 1
        assert left == right

    def test_differing_cyclic_list_subclasses_compare_unequal(self) -> None:
        def build(marker: int) -> _PlainListSubclass:
            cyclic = _PlainListSubclass([marker])
            cyclic.append(cyclic)
            return cyclic

        assert Options(group={"k": build(1)}) != Options(group={"k": build(2)})

    def test_a_list_subclass_with_a_custom_eq_still_decides(self) -> None:
        class _Subclass(list[Any]):
            def __eq__(self, other: object) -> bool:
                return False

        assert Options(group={"k": _Subclass()}) != Options(group={"k": _Subclass()})

    def test_a_dict_subclass_and_a_plain_dict_still_compare_by_value(self) -> None:
        assert Options(group={"k": _PlainDictSubclass({"a": 1})}) == Options(group={"k": {"a": 1}})


_HASH_CONTRACT_PAIRS: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
    ("one_node_vs_two_node_cycle", _self_referential_list, _two_node_cycle),
    ("two_node_vs_one_node_cycle", _two_node_cycle, _self_referential_list),
    ("same_shape_self_cycles", _self_referential_list, _self_referential_list),
    ("same_shape_two_node_cycles", _two_node_cycle, _two_node_cycle),
    ("same_shape_cyclic_dicts", _self_referential_dict, _self_referential_dict),
    ("cyclic_dict_vs_cyclic_list", _self_referential_dict, _self_referential_list),
    ("mutually_referential", _mutually_referential, _mutually_referential),
    ("mutually_referential_vs_self_cycle", _mutually_referential, _self_referential_list),
    ("cyclic_dict_subclasses", _self_referential_dict_subclass, _self_referential_dict_subclass),
    ("cyclic_list_subclasses", _self_referential_list_subclass, _self_referential_list_subclass),
    ("acyclic_nested", _acyclic_nested, _acyclic_nested),
    ("acyclic_vs_cyclic", _acyclic_nested, _self_referential_dict),
]


class TestEqualityAgreesWithHashing:
    """Equality follows the shape-based hasher: a back-reference matches only another back-reference."""

    @pytest.mark.parametrize(
        ("left_builder", "right_builder"),
        [pytest.param(left, right, id=name) for name, left, right in _HASH_CONTRACT_PAIRS],
    )
    def test_equal_options_hash_alike(self, left_builder: Callable[[], Any], right_builder: Callable[[], Any]) -> None:
        left = Options(group={"k": left_builder()})
        right = Options(group={"k": right_builder()})

        if left == right:
            assert hash(left) == hash(right)

    @pytest.mark.parametrize(
        ("left_builder", "right_builder"),
        [pytest.param(left, right, id=name) for name, left, right in _HASH_CONTRACT_PAIRS],
    )
    def test_equal_hashable_dicts_hash_alike(
        self, left_builder: Callable[[], Any], right_builder: Callable[[], Any]
    ) -> None:
        left = HashableDict({"k": left_builder()})
        right = HashableDict({"k": right_builder()})

        if left == right:
            assert hash(left) == hash(right)

    def test_one_node_and_two_node_cycles_compare_unequal(self) -> None:
        one_node = _self_referential_list()
        two_node = _two_node_cycle()

        assert Options(group={"k": one_node}) != Options(group={"k": two_node})
        assert Options(group={"k": two_node}) != Options(group={"k": one_node})

    def test_one_node_and_two_node_cycles_compare_unequal_for_hashable_dict(self) -> None:
        one_node = _self_referential_list()
        two_node = _two_node_cycle()

        assert HashableDict({"k": one_node}) != HashableDict({"k": two_node})
        assert HashableDict({"k": two_node}) != HashableDict({"k": one_node})


class TestAcyclicParityWithPlainEquality:
    """Cases where the walk must reproduce plain container == exactly."""

    def test_equal_but_not_identical_string_keys_match(self) -> None:
        left = {"".join(["ab", "cd"]): 1}
        right = {"".join(["abc", "d"]): 1}

        assert Options(group={"k": left}) == Options(group={"k": right})

    def test_equal_but_not_identical_tuple_keys_match(self) -> None:
        left = {(1, "".join(["a", "b"])): [1]}
        right = {(1, "".join(["ab"])): [1]}

        assert Options(group={"k": left}) == Options(group={"k": right})

    def test_same_items_in_different_insertion_orders_compare_equal(self) -> None:
        assert Options(group={"k": {"a": 1, "b": 2}}) == Options(group={"k": {"b": 2, "a": 1}})
        assert HashableDict({"a": 1, "b": 2}) == HashableDict({"b": 2, "a": 1})

    def test_true_and_one_keys_collide_as_in_plain_dicts(self) -> None:
        assert Options(group={"k": {True: "x"}}) == Options(group={"k": {1: "x"}})
        assert Options(group={"k": {True: "x"}}) != Options(group={"k": {1: "y"}})

    def test_distinct_nan_leaves_compare_unequal(self) -> None:
        assert Options(group={"k": [float("nan")]}) != Options(group={"k": [float("nan")]})
        assert Options(group={"k": {"n": float("nan")}}) != Options(group={"k": {"n": float("nan")}})

    def test_the_same_nan_object_compares_equal(self) -> None:
        nan = float("nan")
        assert Options(group={"k": [nan]}) == Options(group={"k": [nan]})
        assert Options(group={"k": {"n": nan}}) == Options(group={"k": {"n": nan}})

    def test_a_non_bool_eq_leaf_raises_as_plain_equality_did(self) -> None:
        np = pytest.importorskip("numpy")

        with pytest.raises(ValueError):
            _ = Options(group={"k": [np.array([1, 2])]}) == Options(group={"k": [np.array([1, 2])]})

    def test_a_top_level_non_bool_eq_value_raises_as_plain_equality_did(self) -> None:
        np = pytest.importorskip("numpy")

        with pytest.raises(ValueError):
            _ = Options(group={"k": np.array([1, 2])}) == Options(group={"k": np.array([1, 2])})
