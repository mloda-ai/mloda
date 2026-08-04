"""Tests for SingleFilter integration with FilterParameterImpl."""

from typing import Any

import pytest

from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda.core.filter.filter_parameter import FilterParameter, FilterParameterImpl


# --- Parameter type tests ---


def test_parameter_is_filter_parameter_impl() -> None:
    """Test SingleFilter.parameter is a FilterParameterImpl instance."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert isinstance(single_filter.parameter, FilterParameterImpl)


def test_parameter_satisfies_filter_parameter_protocol() -> None:
    """Test SingleFilter.parameter satisfies FilterParameter protocol."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert isinstance(single_filter.parameter, FilterParameter)


# --- Property accessor tests ---


def test_value_property_for_min_filter() -> None:
    """Test accessing parameter.value for a min filter."""
    single_filter = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterType.MIN,
        parameter={"value": 0},
    )
    assert single_filter.parameter.value == 0


def test_value_property_for_max_filter() -> None:
    """Test accessing parameter.value for a max filter."""
    single_filter = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterType.MAX,
        parameter={"value": 100},
    )
    assert single_filter.parameter.value == 100


def test_value_property_for_equal_filter() -> None:
    """Test accessing parameter.value for an equal filter."""
    single_filter = SingleFilter(
        filter_feature="status",
        filter_type=FilterType.EQUAL,
        parameter={"value": "active"},
    )
    assert single_filter.parameter.value == "active"


def test_values_property_for_categorical_inclusion() -> None:
    """Test accessing parameter.values for categorical_inclusion filter.

    Filter engines consume this accessor directly, so it must stay a list regardless of how the
    parameter is stored internally.
    """
    single_filter = SingleFilter(
        filter_feature="category",
        filter_type=FilterType.CATEGORICAL_INCLUSION,
        parameter={"values": ["A", "B", "C"]},
    )
    assert isinstance(single_filter.parameter.values, list)
    assert single_filter.parameter.values == ["A", "B", "C"]


def test_range_filter_min_value_property() -> None:
    """Test accessing parameter.min_value for range filter."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.min_value == 25


def test_range_filter_max_value_property() -> None:
    """Test accessing parameter.max_value for range filter."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.max_value == 50


def test_range_filter_max_exclusive_property() -> None:
    """Test accessing parameter.max_exclusive for range filter."""
    single_filter = SingleFilter(
        filter_feature="score",
        filter_type=FilterType.RANGE,
        parameter={"min": 0, "max": 100, "max_exclusive": True},
    )
    assert single_filter.parameter.max_exclusive is True


def test_range_filter_max_exclusive_default_false() -> None:
    """Test parameter.max_exclusive defaults to False."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.max_exclusive is False


def test_value_property_returns_none_when_not_present() -> None:
    """Test parameter.value returns None when not present."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.value is None


def test_values_property_returns_none_when_not_present() -> None:
    """Test parameter.values returns None when not present."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.values is None


# --- Hashability preservation tests ---


def test_single_filter_is_hashable() -> None:
    """Test SingleFilter instances are hashable."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    hash_value = hash(single_filter)
    assert isinstance(hash_value, int)


def test_equal_filters_have_equal_hashes() -> None:
    """Test equal SingleFilters have the same hash."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert hash(single_filter1) == hash(single_filter2)


def test_single_filter_can_be_used_in_set() -> None:
    """Test SingleFilter can be added to a set."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    single_filter3 = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterType.MIN,
        parameter={"value": 0},
    )

    filter_set = {single_filter1, single_filter2, single_filter3}
    assert len(filter_set) == 2


def test_single_filter_can_be_dict_key() -> None:
    """Test SingleFilter can be used as dictionary key."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    filter_dict = {single_filter: "test_value"}
    assert filter_dict[single_filter] == "test_value"


# --- Equality preservation tests ---


def test_filters_with_same_parameters_are_equal() -> None:
    """Test SingleFilters with identical parameters are equal."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter1 == single_filter2


def test_filters_with_different_parameters_are_not_equal() -> None:
    """Test SingleFilters with different parameters are not equal."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 30, "max": 60},
    )
    assert single_filter1 != single_filter2


def test_filters_with_different_features_are_not_equal() -> None:
    """Test SingleFilters with different features are not equal."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter1 != single_filter2


def test_filters_with_different_types_are_not_equal() -> None:
    """Test SingleFilters with different filter types are not equal."""
    single_filter1 = SingleFilter(
        filter_feature="value",
        filter_type=FilterType.MIN,
        parameter={"value": 25},
    )
    single_filter2 = SingleFilter(
        filter_feature="value",
        filter_type=FilterType.MAX,
        parameter={"value": 25},
    )
    assert single_filter1 != single_filter2


def test_filter_equality_with_unordered_parameters() -> None:
    """Test parameter order doesn't affect equality."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"min": 25, "max": 50, "max_exclusive": True},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterType.RANGE,
        parameter={"max": 50, "max_exclusive": True, "min": 25},
    )
    assert single_filter1 == single_filter2


# --- Unhashable parameter rejection tests (see issue #925) ---
#
# SingleFilter is hashed as soon as it enters a set, so an unhashable parameter value has to be
# refused while constructing the filter, where the caller can still see which key is at fault.


def test_single_filter_rejects_dict_parameter_value() -> None:
    """Test constructing a filter with a dict parameter value raises ValueError instead of building it."""
    parameter: dict[str, Any] = {"value": {"a": 1}}

    with pytest.raises(ValueError, match="value"):
        SingleFilter(filter_feature="config", filter_type=FilterType.EQUAL, parameter=parameter)


def test_single_filter_rejects_nested_unhashable_parameter_value() -> None:
    """Test a dict nested in a list parameter value is refused at construction."""
    parameter: dict[str, Any] = {"values": [{"a": 1}]}

    with pytest.raises(ValueError, match="values"):
        SingleFilter(filter_feature="category", filter_type=FilterType.CATEGORICAL_INCLUSION, parameter=parameter)


def test_single_filter_rejection_names_the_offending_key() -> None:
    """Test the raised message names the key holding the unhashable value."""
    parameter: dict[str, Any] = {"min": 25, "payload": {"a": 1}}

    with pytest.raises(ValueError, match="payload"):
        SingleFilter(filter_feature="age", filter_type=FilterType.RANGE, parameter=parameter)


def test_single_filter_still_accepts_collection_parameter_values() -> None:
    """Test hashable collection parameters keep constructing and hashing after the rejection."""
    parameter: dict[str, Any] = {"values": {"A", "B"}}
    single_filter = SingleFilter(
        filter_feature="category",
        filter_type=FilterType.CATEGORICAL_INCLUSION,
        parameter=parameter,
    )

    assert isinstance(hash(single_filter), int)
    assert sorted(single_filter.parameter.values or []) == ["A", "B"]
