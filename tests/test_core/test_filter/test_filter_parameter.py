"""Tests for FilterParameter Protocol and FilterParameterImpl."""

import re
from decimal import Decimal

import pytest
from typing import Any
from mloda.core.filter.filter_parameter import FilterParameter, FilterParameterImpl


class AlwaysRaisesOnHash:
    """Defines __hash__, so it reports as hashable through its type, but raises when hashed."""

    def __hash__(self) -> int:
        raise TypeError("this object refuses to be hashed")


# --- Creation tests ---


def test_from_dict_with_single_value() -> None:
    """Test creating FilterParameterImpl with single value."""
    params = {"value": 25}
    filter_param = FilterParameterImpl.from_dict(params)

    assert isinstance(filter_param, FilterParameterImpl)
    assert filter_param._raw == (("value", 25),)


def test_from_dict_with_range_params() -> None:
    """Test creating FilterParameterImpl with min and max for range filter."""
    params = {"min": 25, "max": 50}
    filter_param = FilterParameterImpl.from_dict(params)

    assert isinstance(filter_param, FilterParameterImpl)
    assert filter_param._raw == (("max", 50), ("min", 25))


def test_from_dict_with_categorical_values() -> None:
    """Test creating FilterParameterImpl with multiple values.

    Internal storage normalizes the list to a tuple so the frozen dataclass stays hashable.
    """
    params = {"values": ["A", "B", "C"]}
    filter_param = FilterParameterImpl.from_dict(params)

    assert isinstance(filter_param, FilterParameterImpl)
    assert filter_param._raw == (("values", ("A", "B", "C")),)


def test_from_dict_with_max_exclusive() -> None:
    """Test creating FilterParameterImpl with max_exclusive flag."""
    params = {"min": 0, "max": 100, "max_exclusive": True}
    filter_param = FilterParameterImpl.from_dict(params)

    expected_raw = (("max", 100), ("max_exclusive", True), ("min", 0))
    assert filter_param._raw == expected_raw


def test_from_dict_with_empty_dict() -> None:
    """Test creating FilterParameterImpl with empty dict."""
    params: dict[str, Any] = {}
    filter_param = FilterParameterImpl.from_dict(params)

    assert isinstance(filter_param, FilterParameterImpl)
    assert filter_param._raw == ()


# --- Value property tests ---


def test_value_property_returns_value_when_present() -> None:
    """Test value property returns the value when present."""
    filter_param = FilterParameterImpl.from_dict({"value": 42})
    assert filter_param.value == 42


def test_value_property_returns_none_when_not_present() -> None:
    """Test value property returns None when not in parameters."""
    filter_param = FilterParameterImpl.from_dict({"min": 10, "max": 20})
    assert filter_param.value is None


def test_value_property_with_string_value() -> None:
    """Test value property works with string values."""
    filter_param = FilterParameterImpl.from_dict({"value": "test_pattern"})
    assert filter_param.value == "test_pattern"


# --- Values property tests ---


def test_values_property_returns_list_for_categorical() -> None:
    """Test values property returns list for categorical_inclusion filter.

    The public accessor must honour its declared `Optional[list[Any]]` type even though the
    internal storage keeps a tuple. Filter engines rely on this: PySpark's `Column.isin` only
    unwraps list/set arguments, so leaking a tuple silently breaks the Spark categorical filter.
    """
    filter_param = FilterParameterImpl.from_dict({"values": ["A", "B", "C"]})
    assert isinstance(filter_param.values, list)
    assert filter_param.values == ["A", "B", "C"]


def test_values_property_returns_none_when_not_present() -> None:
    """Test values property returns None when not in parameters."""
    filter_param = FilterParameterImpl.from_dict({"value": 10})
    assert filter_param.values is None


def test_values_property_with_empty_list() -> None:
    """Test values property handles empty list."""
    params: dict[str, Any] = {"values": []}
    filter_param = FilterParameterImpl.from_dict(params)
    assert isinstance(filter_param.values, list)
    assert filter_param.values == []


# --- Range property tests ---


def test_min_value_property_for_range_filter() -> None:
    """Test min_value property returns minimum value."""
    filter_param = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    assert filter_param.min_value == 25


def test_max_value_property_for_range_filter() -> None:
    """Test max_value property returns maximum value."""
    filter_param = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    assert filter_param.max_value == 50


def test_min_value_returns_none_when_not_present() -> None:
    """Test min_value returns None when not in parameters."""
    filter_param = FilterParameterImpl.from_dict({"value": 10})
    assert filter_param.min_value is None


def test_max_value_returns_none_when_not_present() -> None:
    """Test max_value returns None when not in parameters."""
    filter_param = FilterParameterImpl.from_dict({"value": 10})
    assert filter_param.max_value is None


# --- Max exclusive property tests ---


def test_max_exclusive_returns_true_when_set() -> None:
    """Test max_exclusive property returns True when set."""
    filter_param = FilterParameterImpl.from_dict({"min": 0, "max": 100, "max_exclusive": True})
    assert filter_param.max_exclusive is True


def test_max_exclusive_returns_false_as_default() -> None:
    """Test max_exclusive property returns False when not present."""
    filter_param = FilterParameterImpl.from_dict({"min": 0, "max": 100})
    assert filter_param.max_exclusive is False


def test_max_exclusive_returns_false_when_explicitly_false() -> None:
    """Test max_exclusive property returns False when explicitly False."""
    filter_param = FilterParameterImpl.from_dict({"min": 0, "max": 100, "max_exclusive": False})
    assert filter_param.max_exclusive is False


# --- Hashability tests ---


def test_filter_parameter_is_hashable() -> None:
    """Test FilterParameterImpl instances are hashable."""
    filter_param = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    hash_value = hash(filter_param)
    assert isinstance(hash_value, int)


def test_equal_parameters_have_equal_hashes() -> None:
    """Test equal FilterParameterImpl have equal hashes."""
    filter_param1 = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    filter_param2 = FilterParameterImpl.from_dict({"min": 25, "max": 50})

    assert hash(filter_param1) == hash(filter_param2)
    assert filter_param1 == filter_param2


def test_filter_parameter_can_be_used_in_set() -> None:
    """Test FilterParameterImpl can be used in a set."""
    filter_param1 = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    filter_param2 = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    filter_param3 = FilterParameterImpl.from_dict({"value": 100})

    param_set = {filter_param1, filter_param2, filter_param3}
    assert len(param_set) == 2


def test_filter_parameter_can_be_used_as_dict_key() -> None:
    """Test FilterParameterImpl can be used as dictionary key."""
    filter_param = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    test_dict = {filter_param: "test_value"}
    assert test_dict[filter_param] == "test_value"


def test_different_parameters_have_different_hashes() -> None:
    """Test different FilterParameterImpl have different hashes."""
    filter_param1 = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    filter_param2 = FilterParameterImpl.from_dict({"min": 30, "max": 50})

    assert filter_param1 != filter_param2
    assert hash(filter_param1) != hash(filter_param2)


# --- Immutability tests ---


def test_filter_parameter_is_immutable() -> None:
    """Test FilterParameterImpl cannot be modified after creation."""
    filter_param = FilterParameterImpl.from_dict({"value": 25})

    with pytest.raises((AttributeError, Exception)):
        filter_param._raw = (("value", 50),)  # type: ignore


# --- Protocol compliance tests ---


def test_filter_parameter_impl_satisfies_protocol() -> None:
    """Test FilterParameterImpl implements FilterParameter protocol."""
    filter_param = FilterParameterImpl.from_dict({"min": 25, "max": 50})
    assert isinstance(filter_param, FilterParameter)


def test_protocol_has_required_properties() -> None:
    """Test FilterParameter protocol defines all required properties."""
    expected_properties = ["value", "values", "min_value", "max_value", "max_exclusive"]

    for prop in expected_properties:
        assert hasattr(FilterParameter, prop), f"FilterParameter should define '{prop}'"


# --- Edge case tests ---


def test_parameter_with_none_value() -> None:
    """Test FilterParameterImpl handles None as a value."""
    filter_param = FilterParameterImpl.from_dict({"value": None})
    assert filter_param.value is None


def test_parameter_with_zero_value() -> None:
    """Test FilterParameterImpl correctly handles zero value."""
    filter_param = FilterParameterImpl.from_dict({"value": 0})
    assert filter_param.value == 0
    assert filter_param.value is not None


def test_parameter_sorting_is_consistent() -> None:
    """Test parameter sorting is consistent regardless of input order."""
    filter_param1 = FilterParameterImpl.from_dict({"max": 50, "min": 25})
    filter_param2 = FilterParameterImpl.from_dict({"min": 25, "max": 50})

    assert filter_param1._raw == filter_param2._raw
    assert filter_param1 == filter_param2


# --- Collection value normalization tests (see issue #664) ---
#
# Contract: collection values are stored hashable (tuple) in `_raw`, but the public `values`
# property returns a `list`, matching its declared type. Scalars must never be exploded.


def test_from_dict_with_list_values_normalizes_raw_to_tuple() -> None:
    """Test a list value is stored as a tuple internally so the frozen dataclass stays hashable."""
    filter_param = FilterParameterImpl.from_dict({"values": ["EU", "NA"]})

    assert filter_param._raw == (("values", ("EU", "NA")),)
    assert isinstance(hash(filter_param), int)


def test_from_dict_with_set_values_is_hashable() -> None:
    """Test a set value is accepted and does not break hashing."""
    params: dict[str, Any] = {"values": {"EU", "NA"}}
    filter_param = FilterParameterImpl.from_dict(params)

    assert isinstance(hash(filter_param), int)


def test_values_property_returns_list_for_set_input() -> None:
    """Test a set value comes back out of the public accessor as a list."""
    params: dict[str, Any] = {"values": {"EU", "NA"}}
    filter_param = FilterParameterImpl.from_dict(params)

    assert isinstance(filter_param.values, list)
    assert sorted(filter_param.values) == ["EU", "NA"]


def test_cross_type_equal_set_values_normalize_equal() -> None:
    """Test set elements normalize by value equality, not repr, so 1 and True land in the same order."""
    from_int = FilterParameterImpl.from_dict({"values": {1, 2}})
    from_bool = FilterParameterImpl.from_dict({"values": {True, 2}})

    assert from_int == from_bool
    assert hash(from_int) == hash(from_bool)


def test_cross_type_equal_frozenset_values_normalize_equal() -> None:
    """Test the same cross-type-equal normalization holds for a frozenset input."""
    from_int = FilterParameterImpl.from_dict({"values": frozenset({1, 2})})
    from_bool = FilterParameterImpl.from_dict({"values": frozenset({True, 2})})

    assert from_int == from_bool
    assert hash(from_int) == hash(from_bool)


def test_homogeneous_set_values_still_deduplicate_regardless_of_insertion_order() -> None:
    """Test the pre-existing homogeneous, natively-orderable behavior (dedup, hashability) survives."""
    first = FilterParameterImpl.from_dict({"values": {"NA", "EU"}})
    second = FilterParameterImpl.from_dict({"values": {"EU", "NA"}})

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_values_property_returns_list_for_tuple_input() -> None:
    """Test a tuple value comes back out of the public accessor as a list."""
    filter_param = FilterParameterImpl.from_dict({"values": ("EU", "NA")})

    assert isinstance(filter_param.values, list)
    assert filter_param.values == ["EU", "NA"]


def test_values_property_does_not_explode_string_value() -> None:
    """Test a plain string value is not treated as a sequence of characters.

    Normalizing collection values must not reach into scalars: a string is iterable, so a naive
    conversion would turn "EU" into ["E", "U"]. `values` is typed as a list, so the scalar is read
    back through `Any`.
    """
    filter_param = FilterParameterImpl.from_dict({"values": "EU"})
    values: Any = filter_param.values

    assert values == "EU"
    assert values != ["E", "U"]


def test_value_property_does_not_explode_string_value() -> None:
    """Test the scalar `value` accessor keeps a string intact."""
    filter_param = FilterParameterImpl.from_dict({"value": "EU"})

    assert filter_param.value == "EU"


def test_list_and_tuple_values_are_equal_and_hash_equal() -> None:
    """Test list and tuple inputs normalize to the same parameter so they deduplicate."""
    from_list = FilterParameterImpl.from_dict({"values": ["EU"]})
    from_tuple = FilterParameterImpl.from_dict({"values": ("EU",)})

    assert from_list == from_tuple
    assert hash(from_list) == hash(from_tuple)
    assert len({from_list, from_tuple}) == 1


# --- Unhashable value rejection tests (see issue #925) ---


def test_from_dict_rejects_dict_value() -> None:
    """Test a dict value is rejected, since normalization leaves it raw and unhashable."""
    params: dict[str, Any] = {"value": {"a": 1}}

    with pytest.raises(ValueError, match=r"'value'"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_rejects_dict_nested_in_list_value() -> None:
    """Test a list value holding a dict is rejected, since the shallow tuple conversion misses it."""
    params: dict[str, Any] = {"values": [{"a": 1}]}

    with pytest.raises(ValueError, match="values"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_rejects_list_nested_in_list_value() -> None:
    """Test a list value holding a list is rejected."""
    params: dict[str, Any] = {"values": [[1, 2]]}

    with pytest.raises(ValueError, match="values"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_rejects_dict_nested_in_tuple_value() -> None:
    """Test a tuple value holding a dict is rejected, even though the tuple itself is a hashable type."""
    params: dict[str, Any] = {"values": ({"a": 1},)}

    with pytest.raises(ValueError, match="values"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_rejects_unhashable_leaf_value() -> None:
    """Test an unhashable scalar such as bytearray is rejected."""
    params: dict[str, Any] = {"value": bytearray(b"abc")}

    with pytest.raises(ValueError, match=r"'value'"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_rejects_unhashable_range_bound() -> None:
    """Test the rejection is not limited to the value/values keys."""
    params: dict[str, Any] = {"min": {"a": 1}, "max": 50}

    with pytest.raises(ValueError, match="min"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_error_names_the_offending_key() -> None:
    """Test the message names the key that carries the unhashable value, not a neighbouring key."""
    params: dict[str, Any] = {"min": 1, "payload": {"a": 1}}

    with pytest.raises(ValueError, match="payload"):
        FilterParameterImpl.from_dict(params)


@pytest.mark.parametrize(
    "value",
    [Decimal("sNaN"), memoryview(bytearray(b"ab")), AlwaysRaisesOnHash()],
    ids=["signaling-nan", "writable-memoryview", "raising-hash"],
)
def test_from_dict_rejects_value_whose_hash_raises(value: Any) -> None:
    """Test a value that defines __hash__ but raises is rejected, not deferred to a later hash call."""
    with pytest.raises(ValueError, match=r"'value'"):
        FilterParameterImpl.from_dict({"value": value})


def test_from_dict_rejects_value_whose_hash_raises_inside_a_collection() -> None:
    """Test the raising value is caught through a list too, not only as a bare scalar."""
    params: dict[str, Any] = {"values": [1, AlwaysRaisesOnHash()]}

    with pytest.raises(ValueError, match=r"'values'"):
        FilterParameterImpl.from_dict(params)


@pytest.mark.parametrize(
    "params, key, culprit",
    [
        ({"value": {"a": 1}}, "value", "dict"),
        ({"values": [{"a": 1}]}, "values", "dict"),
        ({"values": [[1, 2]]}, "values", "list"),
    ],
    ids=["dict-value", "dict-in-list", "list-in-list"],
)
def test_from_dict_error_names_the_offending_type(params: dict[str, Any], key: str, culprit: str) -> None:
    """Test the message names the inner type that cannot be hashed, since the key alone can mislead."""
    with pytest.raises(ValueError) as excinfo:
        FilterParameterImpl.from_dict(params)

    message = str(excinfo.value)
    assert f"'{key}'" in message, message
    assert re.search(rf"\b{culprit}\b", message), message


def test_from_dict_accepts_deeply_nested_tuple() -> None:
    """Test a tuple nested past the Python recursion limit stays accepted, since hash() copes with it."""
    deep: Any = ()
    for _ in range(1000):
        deep = (deep,)
    hash(deep)

    filter_param = FilterParameterImpl.from_dict({"value": deep})

    assert filter_param.value is deep
    hash(filter_param)


# --- Non-string key rejection tests (see issue #959) ---


def test_from_dict_rejects_non_string_key() -> None:
    """Test a non-string key raises ValueError instead of the raw TypeError out of sorted()."""
    params: dict[Any, Any] = {1: "a", "b": 2}

    with pytest.raises(ValueError, match=r"key 1 is not a string"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_non_string_key_error_names_the_key() -> None:
    """Test the message names the offending key rather than a neighbouring one."""
    params: dict[Any, Any] = {"value": 25, ("tuple", "key"): 2}

    with pytest.raises(ValueError) as excinfo:
        FilterParameterImpl.from_dict(params)

    message = str(excinfo.value)
    assert "tuple" in message, message


def test_from_dict_rejects_non_string_key_even_without_mixed_types() -> None:
    """Test a key set that sorts fine is still rejected when the keys are not strings."""
    params: dict[Any, Any] = {1: "a", 2: "b"}

    with pytest.raises(ValueError, match=r"key 1 is not a string"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_rejects_none_key() -> None:
    """Test None as a key is rejected with a ValueError."""
    params: dict[Any, Any] = {None: "a"}

    with pytest.raises(ValueError, match="None"):
        FilterParameterImpl.from_dict(params)


def test_from_dict_rejects_non_string_key_before_unhashable_value_check() -> None:
    """Test the key check fires even when another key carries an unhashable value."""
    params: dict[Any, Any] = {1: "a", "payload": {"a": 1}}

    with pytest.raises(ValueError) as excinfo:
        FilterParameterImpl.from_dict(params)

    assert "1" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize(
    "params",
    [
        {"value": 25},
        {"value": None},
        {"value": "EU"},
        {"values": "EU"},
        {"values": ["A", "B"]},
        {"values": {"A", "B"}},
        {"values": ("A", "B")},
        {"values": [(1, 2), (3, 4)]},
        {"min": 0, "max": 100, "max_exclusive": True},
    ],
    ids=["int", "none", "str", "str-values", "list", "set", "tuple", "list-of-tuples", "range"],
)
def test_from_dict_accepts_every_hashable_value_shape(params: dict[str, Any]) -> None:
    """Test the rejection leaves the supported value shapes untouched."""
    hash(FilterParameterImpl.from_dict(params))
