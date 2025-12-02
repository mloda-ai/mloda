"""Tests for SingleFilter integration with FilterParameterImpl."""

from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_core.filter.filter_parameter import FilterParameter, FilterParameterImpl


# --- Parameter type tests ---


def test_parameter_is_filter_parameter_impl() -> None:
    """Test SingleFilter.parameter is a FilterParameterImpl instance."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert isinstance(single_filter.parameter, FilterParameterImpl)


def test_parameter_satisfies_filter_parameter_protocol() -> None:
    """Test SingleFilter.parameter satisfies FilterParameter protocol."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert isinstance(single_filter.parameter, FilterParameter)


# --- Property accessor tests ---


def test_value_property_for_min_filter() -> None:
    """Test accessing parameter.value for a min filter."""
    single_filter = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterTypeEnum.min,
        parameter={"value": 0},
    )
    assert single_filter.parameter.value == 0


def test_value_property_for_max_filter() -> None:
    """Test accessing parameter.value for a max filter."""
    single_filter = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterTypeEnum.max,
        parameter={"value": 100},
    )
    assert single_filter.parameter.value == 100


def test_value_property_for_equal_filter() -> None:
    """Test accessing parameter.value for an equal filter."""
    single_filter = SingleFilter(
        filter_feature="status",
        filter_type=FilterTypeEnum.equal,
        parameter={"value": "active"},
    )
    assert single_filter.parameter.value == "active"


def test_values_property_for_categorical_inclusion() -> None:
    """Test accessing parameter.values for categorical_inclusion filter."""
    single_filter = SingleFilter(
        filter_feature="category",
        filter_type=FilterTypeEnum.categorical_inclusion,
        parameter={"values": ["A", "B", "C"]},
    )
    assert single_filter.parameter.values == ["A", "B", "C"]


def test_range_filter_min_value_property() -> None:
    """Test accessing parameter.min_value for range filter."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.min_value == 25


def test_range_filter_max_value_property() -> None:
    """Test accessing parameter.max_value for range filter."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.max_value == 50


def test_range_filter_max_exclusive_property() -> None:
    """Test accessing parameter.max_exclusive for range filter."""
    single_filter = SingleFilter(
        filter_feature="score",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 0, "max": 100, "max_exclusive": True},
    )
    assert single_filter.parameter.max_exclusive is True


def test_range_filter_max_exclusive_default_false() -> None:
    """Test parameter.max_exclusive defaults to False."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.max_exclusive is False


def test_value_property_returns_none_when_not_present() -> None:
    """Test parameter.value returns None when not present."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.value is None


def test_values_property_returns_none_when_not_present() -> None:
    """Test parameter.values returns None when not present."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter.parameter.values is None


# --- Hashability preservation tests ---


def test_single_filter_is_hashable() -> None:
    """Test SingleFilter instances are hashable."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    hash_value = hash(single_filter)
    assert isinstance(hash_value, int)


def test_equal_filters_have_equal_hashes() -> None:
    """Test equal SingleFilters have the same hash."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert hash(single_filter1) == hash(single_filter2)


def test_single_filter_can_be_used_in_set() -> None:
    """Test SingleFilter can be added to a set."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    single_filter3 = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterTypeEnum.min,
        parameter={"value": 0},
    )

    filter_set = {single_filter1, single_filter2, single_filter3}
    assert len(filter_set) == 2


def test_single_filter_can_be_dict_key() -> None:
    """Test SingleFilter can be used as dictionary key."""
    single_filter = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    filter_dict = {single_filter: "test_value"}
    assert filter_dict[single_filter] == "test_value"


# --- Equality preservation tests ---


def test_filters_with_same_parameters_are_equal() -> None:
    """Test SingleFilters with identical parameters are equal."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter1 == single_filter2


def test_filters_with_different_parameters_are_not_equal() -> None:
    """Test SingleFilters with different parameters are not equal."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 30, "max": 60},
    )
    assert single_filter1 != single_filter2


def test_filters_with_different_features_are_not_equal() -> None:
    """Test SingleFilters with different features are not equal."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    single_filter2 = SingleFilter(
        filter_feature="temperature",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50},
    )
    assert single_filter1 != single_filter2


def test_filters_with_different_types_are_not_equal() -> None:
    """Test SingleFilters with different filter types are not equal."""
    single_filter1 = SingleFilter(
        filter_feature="value",
        filter_type=FilterTypeEnum.min,
        parameter={"value": 25},
    )
    single_filter2 = SingleFilter(
        filter_feature="value",
        filter_type=FilterTypeEnum.max,
        parameter={"value": 25},
    )
    assert single_filter1 != single_filter2


def test_filter_equality_with_unordered_parameters() -> None:
    """Test parameter order doesn't affect equality."""
    single_filter1 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"min": 25, "max": 50, "max_exclusive": True},
    )
    single_filter2 = SingleFilter(
        filter_feature="age",
        filter_type=FilterTypeEnum.range,
        parameter={"max": 50, "max_exclusive": True, "min": 25},
    )
    assert single_filter1 == single_filter2
