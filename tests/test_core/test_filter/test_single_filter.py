import unittest
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum


class TestSingleFilter(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test variables."""
        self.feature = Feature("age")
        self.filter_type = FilterTypeEnum.range
        self.parameter = {"min": 25, "max": 50}

    def test_single_filter_initialization(self) -> None:
        """Test that SingleFilter initializes correctly."""
        single_filter = SingleFilter(self.feature, self.filter_type, self.parameter)

        self.assertEqual(single_filter.filter_feature, self.feature)
        self.assertEqual(single_filter.filter_type, "range")
        self.assertEqual(single_filter.parameter, (("max", 50), ("min", 25)))  # parameter is tuple

    def test_invalid_filter_type(self) -> None:
        """Test that invalid filter type raises ValueError."""
        with self.assertRaises(ValueError):
            SingleFilter(self.feature, 123, self.parameter)  # type: ignore

    def test_invalid_filter_feature(self) -> None:
        """Test that invalid filter feature raises ValueError."""
        with self.assertRaises(ValueError):
            SingleFilter(123, self.filter_type, self.parameter)

    def test_invalid_parameter(self) -> None:
        """Test that invalid parameter raises ValueError."""
        with self.assertRaises(ValueError):
            SingleFilter(self.feature, self.filter_type, "not_a_dict")  # type: ignore

    def test_empty_parameter(self) -> None:
        """Test that an empty parameter raises ValueError."""
        with self.assertRaises(ValueError):
            SingleFilter(self.feature, self.filter_type, {})  # empty parameter dict

    def test_filter_equality(self) -> None:
        """Test that two identical SingleFilters are considered equal."""
        single_filter1 = SingleFilter(self.feature, self.filter_type, self.parameter)
        single_filter2 = SingleFilter(self.feature, self.filter_type, self.parameter)
        self.assertEqual(single_filter1, single_filter2)

    def test_filter_hash(self) -> None:
        """Test that SingleFilter objects can be used in a set (requires __hash__)."""
        single_filter1 = SingleFilter(self.feature, self.filter_type, self.parameter)
        single_filter2 = SingleFilter(self.feature, self.filter_type, self.parameter)

        filter_set = {single_filter1, single_filter2}
        self.assertEqual(len(filter_set), 1)  # Since they are equal, only one should be in the set
