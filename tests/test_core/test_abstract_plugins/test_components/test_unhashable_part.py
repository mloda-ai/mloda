"""Tests for the shared `unhashable_part` probe and its two call sites: filters reject, HashableDict coerces."""

from collections.abc import Hashable
from typing import Any

import pytest

from mloda.core.abstract_plugins.components import feature as feature_module
from mloda.core.abstract_plugins.components import hashable_dict as hashable_dict_module
from mloda.core.abstract_plugins.components import options as options_module
from mloda.core.abstract_plugins.components.hashable_dict import HashableDict, _deep_hashable
from mloda.core.abstract_plugins.components.utils import unhashable_part
from mloda.core.filter import filter_parameter as filter_parameter_module
from mloda.core.filter.filter_parameter import FilterParameterImpl, _normalize_collections


class _RaisingHash:
    """A leaf that advertises __hash__ but raises TypeError when actually hashed."""

    def __init__(self, token: str) -> None:
        self.token = token

    def __hash__(self) -> int:
        raise TypeError(f"{self.token} refuses to hash")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RaisingHash) and self.token == other.token

    def __repr__(self) -> str:
        return f"_RaisingHash({self.token})"


class TestUnhashablePartOnHashableValues:
    """A value that hashes end to end reports no offender."""

    @pytest.mark.parametrize(
        "value",
        [1, "text", b"bytes", None, 3.5, True, (1, "a"), (), ((1, 2), (3, 4)), frozenset({1, 2}), (frozenset({1}),)],
    )
    def test_hashable_value_returns_none(self, value: Any) -> None:
        assert unhashable_part(value) is None


class TestUnhashablePartNamesTheOffender:
    """The first non-hashing part is named by type, recursing into tuples."""

    def test_dict_is_named(self) -> None:
        assert unhashable_part({"a": 1}) == "dict"

    def test_list_is_named(self) -> None:
        assert unhashable_part([1, 2]) == "list"

    def test_set_is_named(self) -> None:
        assert unhashable_part({1, 2}) == "set"

    def test_tuple_carrying_a_dict_names_the_nested_dict(self) -> None:
        assert unhashable_part((1, {"a": 1})) == "dict"

    def test_deeply_nested_offender_is_named(self) -> None:
        assert unhashable_part((1, (2, (3, [4])))) == "list"

    def test_first_offender_wins(self) -> None:
        assert unhashable_part(({"a": 1}, [2])) == "dict"

    def test_class_whose_hash_raises_is_named(self) -> None:
        assert unhashable_part(_RaisingHash("leaf")) == "_RaisingHash"

    def test_tuple_carrying_a_raising_hash_names_the_leaf(self) -> None:
        assert unhashable_part((1, _RaisingHash("leaf"))) == "_RaisingHash"


class TestUnhashablePartProbesTheRealHash:
    """isinstance(value, Hashable) lies in exactly the two cases the probe must catch."""

    def test_tuple_with_dict_advertises_hashable_but_is_not(self) -> None:
        value = (1, {"a": 1})
        assert isinstance(value, Hashable)
        assert unhashable_part(value) == "dict"

    def test_raising_hash_advertises_hashable_but_is_not(self) -> None:
        value = _RaisingHash("leaf")
        assert isinstance(value, Hashable)
        assert unhashable_part(value) == "_RaisingHash"


class TestFilterParameterRejects:
    """Filter policy: normalize collections shallowly, then REJECT anything that still does not hash."""

    def test_nested_dict_raises_value_error_naming_dict_and_key(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            FilterParameterImpl.from_dict({"values": [{"a": 1}]})
        message = str(excinfo.value)
        assert "dict" in message
        assert "values" in message

    def test_raising_hash_value_is_rejected_by_name(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            FilterParameterImpl.from_dict({"value": _RaisingHash("leaf")})
        assert "_RaisingHash" in str(excinfo.value)

    def test_hashable_parameters_still_build(self) -> None:
        parameter = FilterParameterImpl.from_dict({"values": ["a", "b"]})
        assert parameter.values == ["a", "b"]

    def test_call_site_uses_the_shared_probe(self) -> None:
        assert getattr(filter_parameter_module, "unhashable_part") is unhashable_part

    def test_the_module_private_probe_is_gone(self) -> None:
        assert not hasattr(filter_parameter_module, "_unhashable_type")

    def test_the_old_helper_name_is_gone(self) -> None:
        assert not hasattr(filter_parameter_module, "_make_hashable")


class TestNormalizeCollectionsStaysShallow:
    """`_normalize_collections` only reshapes the top level; it never recurses and never coerces."""

    def test_str_stays_scalar(self) -> None:
        assert _normalize_collections("abc") == "abc"

    def test_bytes_stay_scalar(self) -> None:
        assert _normalize_collections(b"abc") == b"abc"

    def test_list_becomes_a_tuple(self) -> None:
        assert _normalize_collections([1, 2]) == (1, 2)

    def test_set_becomes_a_tuple_sorted_by_repr(self) -> None:
        assert _normalize_collections({"b", "a"}) == ("a", "b")

    def test_frozenset_becomes_a_tuple_sorted_by_repr(self) -> None:
        assert _normalize_collections(frozenset({"b", "a"})) == ("a", "b")

    def test_dict_is_left_unchanged_so_the_probe_can_reject_it(self) -> None:
        assert _normalize_collections({"a": 1}) == {"a": 1}

    def test_nested_list_is_not_recursed_into(self) -> None:
        assert _normalize_collections([[1], [2]]) == ([1], [2])


class TestHashableDictCoerces:
    """HashableDict policy: name the offending leaf, then degrade it to repr so grouping never crashes."""

    def test_hashing_a_raising_leaf_does_not_raise(self) -> None:
        assert isinstance(hash(HashableDict({"leaf": _RaisingHash("acme")})), int)

    def test_equal_repr_leaves_hash_alike(self) -> None:
        left = HashableDict({"leaf": _RaisingHash("acme")})
        right = HashableDict({"leaf": _RaisingHash("acme")})
        assert hash(left) == hash(right)

    def test_a_nested_raising_leaf_is_also_coerced(self) -> None:
        left = HashableDict({"leaf": [{"inner": _RaisingHash("acme")}]})
        right = HashableDict({"leaf": [{"inner": _RaisingHash("acme")}]})
        assert hash(left) == hash(right)

    def test_deep_hashable_coerces_the_leaf_to_its_repr(self) -> None:
        assert _deep_hashable(_RaisingHash("acme")) == repr(_RaisingHash("acme"))

    def test_deep_hashable_recurses_into_containers(self) -> None:
        assert _deep_hashable({"a": [1, 2]}) == (("a", (1, 2)),)

    def test_call_site_uses_the_shared_probe(self) -> None:
        assert getattr(hashable_dict_module, "unhashable_part") is unhashable_part

    def test_the_old_helper_name_is_gone(self) -> None:
        assert not hasattr(hashable_dict_module, "_make_hashable")


class TestCrossModuleImportsFollowTheRename:
    """options and feature import the deep policy under its new name."""

    def test_options_imports_deep_hashable(self) -> None:
        assert getattr(options_module, "_deep_hashable") is _deep_hashable

    def test_feature_imports_deep_hashable(self) -> None:
        assert getattr(feature_module, "_deep_hashable") is _deep_hashable

    @pytest.mark.parametrize("module", [options_module, feature_module])
    def test_the_old_name_is_no_longer_imported(self, module: Any) -> None:
        assert not hasattr(module, "_make_hashable")
