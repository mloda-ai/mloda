import gc

import pytest

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.filter.filter_parameter import FilterParameterImpl


class TestSingleFilter:
    def setup_method(self) -> None:
        """Set up test variables."""
        self.feature = Feature("age")
        self.filter_type = FilterType.RANGE
        self.parameter = {"min": 25, "max": 50}

    def test_single_filter_initialization(self) -> None:
        """Test that SingleFilter initializes correctly."""
        single_filter = SingleFilter(self.feature, self.filter_type, self.parameter)

        assert single_filter.filter_feature == self.feature
        assert single_filter.filter_type == "range"
        assert isinstance(single_filter.parameter, FilterParameterImpl)
        assert single_filter.parameter.min_value == 25
        assert single_filter.parameter.max_value == 50

    def test_invalid_filter_type(self) -> None:
        """Test that invalid filter type raises ValueError."""
        with pytest.raises(ValueError):
            SingleFilter(self.feature, 123, self.parameter)  # type: ignore

    def test_invalid_filter_feature(self) -> None:
        """Test that invalid filter feature raises ValueError."""
        with pytest.raises(ValueError):
            SingleFilter(123, self.filter_type, self.parameter)

    def test_invalid_parameter(self) -> None:
        """Test that invalid parameter raises ValueError."""
        with pytest.raises(ValueError):
            SingleFilter(self.feature, self.filter_type, "not_a_dict")  # type: ignore

    def test_empty_parameter(self) -> None:
        """Test that an empty parameter raises ValueError."""
        with pytest.raises(ValueError):
            SingleFilter(self.feature, self.filter_type, {})  # empty parameter dict

    def test_filter_equality(self) -> None:
        """Test that two identical SingleFilters are considered equal."""
        single_filter1 = SingleFilter(self.feature, self.filter_type, self.parameter)
        single_filter2 = SingleFilter(self.feature, self.filter_type, self.parameter)
        assert single_filter1 == single_filter2

    def test_filter_hash(self) -> None:
        """Test that SingleFilter objects can be used in a set (requires __hash__)."""
        single_filter1 = SingleFilter(self.feature, self.filter_type, self.parameter)
        single_filter2 = SingleFilter(self.feature, self.filter_type, self.parameter)

        filter_set = {single_filter1, single_filter2}
        assert len(filter_set) == 1  # Since they are equal, only one should be in the set


SCOPE_A_ID = "ScopeIdA728x"
SCOPE_B_ID = "ScopeIdB728x"


def _scoped_filter(scope: str | type[FeatureGroup] | None) -> SingleFilter:
    """An otherwise identical filter declaration carrying the given feature_group scope."""
    return SingleFilter(Feature("age", feature_group=scope), FilterType.RANGE, {"min": 25, "max": 50})


def _drive_class_scope_identity() -> tuple[bool, bool, int]:
    """Plain-data readout (cross-scope equal, same-scope equal, set size) over two throwaway class scopes."""
    gc.collect()

    class ScopeIdClassAFG728x(FeatureGroup):
        pass

    class ScopeIdClassBFG728x(FeatureGroup):
        pass

    first_a = _scoped_filter(ScopeIdClassAFG728x)
    second_a = _scoped_filter(ScopeIdClassAFG728x)
    only_b = _scoped_filter(ScopeIdClassBFG728x)
    try:
        return first_a == only_b, first_a == second_a, len({first_a, second_a, only_b})
    finally:
        del ScopeIdClassAFG728x, ScopeIdClassBFG728x, first_a, second_a, only_b
        gc.collect()


class TestSingleFilterScopeIdentity:
    """feature_group_scope is part of SingleFilter identity, so scope-distinct duplicates never collapse."""

    def test_different_string_scopes_are_unequal_and_stay_two_in_a_set(self) -> None:
        scoped_a = _scoped_filter(SCOPE_A_ID)
        scoped_b = _scoped_filter(SCOPE_B_ID)

        assert scoped_a != scoped_b, "two declarations differing only in scope must be unequal"
        assert len({scoped_a, scoped_b}) == 2, "two string scopes must stay two set elements"

    def test_a_scoped_and_an_unscoped_filter_are_unequal_and_stay_two_in_a_set(self) -> None:
        scoped = _scoped_filter(SCOPE_A_ID)
        unscoped = _scoped_filter(None)

        assert scoped != unscoped, "a scoped and an unscoped declaration must be unequal"
        assert len({scoped, unscoped}) == 2, "scoped and unscoped must stay two set elements"

    def test_the_same_string_scope_still_merges_to_one(self) -> None:
        """Identity must not overshoot: an identical scope keeps the duplicates one filter."""
        first = _scoped_filter(SCOPE_A_ID)
        second = _scoped_filter(SCOPE_A_ID)

        assert first == second, "the same scope must keep the declarations equal"
        assert hash(first) == hash(second), "equal same-scope filters must hash equal"
        assert len({first, second}) == 1, "the same scope must merge to one set element"

    def test_unscoped_duplicates_still_merge_to_one(self) -> None:
        """Regression control: both scopes are None, so the duplicates keep merging."""
        first = _scoped_filter(None)
        second = _scoped_filter(None)

        assert first == second, "unscoped duplicates must stay equal"
        assert hash(first) == hash(second), "unscoped duplicates must hash equal"
        assert len({first, second}) == 1, "unscoped duplicates must keep merging to one set element"

    def test_class_object_scopes_split_on_the_class_and_merge_on_the_same_class(self) -> None:
        cross_scope_equal, same_scope_equal, set_size = _drive_class_scope_identity()

        assert cross_scope_equal is False, "two different class scopes must make the filters unequal"
        assert same_scope_equal is True, "the same class scope twice must keep the filters equal"
        assert set_size == 2, "three filters over two class scopes must land as two set elements"
