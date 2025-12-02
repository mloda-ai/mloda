"""Tests for FilterParameter Protocol and FilterParameterImpl."""

import pytest
from typing import Any, Dict
from mloda_core.filter.filter_parameter import FilterParameter, FilterParameterImpl


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
    """Test creating FilterParameterImpl with multiple values."""
    params = {"values": ["A", "B", "C"]}
    filter_param = FilterParameterImpl.from_dict(params)

    assert isinstance(filter_param, FilterParameterImpl)
    assert filter_param._raw == (("values", ["A", "B", "C"]),)


def test_from_dict_with_max_exclusive() -> None:
    """Test creating FilterParameterImpl with max_exclusive flag."""
    params = {"min": 0, "max": 100, "max_exclusive": True}
    filter_param = FilterParameterImpl.from_dict(params)

    expected_raw = (("max", 100), ("max_exclusive", True), ("min", 0))
    assert filter_param._raw == expected_raw


def test_from_dict_with_empty_dict() -> None:
    """Test creating FilterParameterImpl with empty dict."""
    params: Dict[str, Any] = {}
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
    """Test values property returns list for categorical_inclusion filter."""
    filter_param = FilterParameterImpl.from_dict({"values": ["A", "B", "C"]})
    assert filter_param.values == ["A", "B", "C"]


def test_values_property_returns_none_when_not_present() -> None:
    """Test values property returns None when not in parameters."""
    filter_param = FilterParameterImpl.from_dict({"value": 10})
    assert filter_param.values is None


def test_values_property_with_empty_list() -> None:
    """Test values property handles empty list."""
    params: Dict[str, Any] = {"values": []}
    filter_param = FilterParameterImpl.from_dict(params)
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
