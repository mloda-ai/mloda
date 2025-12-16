import pytest

from mloda import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda.core.filter.filter_parameter import FilterParameterImpl


class TestSingleFilter:
    def setup_method(self) -> None:
        """Set up test variables."""
        self.feature = Feature("age")
        self.filter_type = FilterType.range
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
