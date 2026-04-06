"""
Tests for FeatureSet filter helper methods.

These tests define the expected behavior of three new methods:
- get_filters_by_column: returns all filters matching a given column name
- get_filter: returns the first filter matching a column name
- get_filter_value: returns the parameter value of the first matching filter
"""

from typing import Any

import pytest

from mloda.user import SingleFilter
from mloda.provider import FeatureSet


class TestGetFiltersByColumn:
    """Tests for get_filters_by_column method."""

    def test_returns_empty_list_when_filters_is_none(self) -> None:
        """FeatureSet with no filters should return an empty list."""
        fs = FeatureSet()

        result = fs.get_filters_by_column("status")

        assert result == []

    def test_returns_empty_list_when_no_match(self) -> None:
        """Querying a column that has no filters should return an empty list."""
        fs = FeatureSet()
        sf = SingleFilter("age", "min", {"value": 18})
        fs.add_filters({sf})

        result = fs.get_filters_by_column("status")

        assert result == []

    def test_returns_matching_filter(self) -> None:
        """Should return a list containing the matching filter."""
        fs = FeatureSet()
        sf = SingleFilter("status", "equal", {"value": "active"})
        fs.add_filters({sf})

        result = fs.get_filters_by_column("status")

        assert len(result) == 1
        assert result[0] is sf

    def test_returns_multiple_filters_for_same_column(self) -> None:
        """Should return all filters that match the given column."""
        fs = FeatureSet()
        sf_min = SingleFilter("age", "min", {"value": 18})
        sf_max = SingleFilter("age", "max", {"value": 65})
        fs.add_filters({sf_min, sf_max})

        result = fs.get_filters_by_column("age")

        assert len(result) == 2
        assert sf_min in result
        assert sf_max in result

    def test_does_not_return_filters_for_other_columns(self) -> None:
        """Querying one column should not return filters for a different column."""
        fs = FeatureSet()
        sf_age = SingleFilter("age", "min", {"value": 18})
        sf_status = SingleFilter("status", "equal", {"value": "active"})
        fs.add_filters({sf_age, sf_status})

        result = fs.get_filters_by_column("age")

        assert len(result) == 1
        assert result[0] is sf_age


class TestGetFilter:
    """Tests for get_filter method."""

    def test_returns_none_when_filters_is_none(self) -> None:
        """FeatureSet with no filters should return None."""
        fs = FeatureSet()

        result = fs.get_filter("status")

        assert result is None

    def test_returns_none_when_no_match(self) -> None:
        """Querying a column with no matching filter should return None."""
        fs = FeatureSet()
        sf = SingleFilter("age", "min", {"value": 18})
        fs.add_filters({sf})

        result = fs.get_filter("status")

        assert result is None

    def test_returns_filter_for_matching_column(self) -> None:
        """Should return the SingleFilter that matches the column name."""
        fs = FeatureSet()
        sf = SingleFilter("status", "equal", {"value": "active"})
        fs.add_filters({sf})

        result = fs.get_filter("status")

        assert isinstance(result, SingleFilter)
        assert result is sf

    def test_returns_a_filter_when_multiple_exist(self) -> None:
        """When multiple filters exist for a column, should return one of them."""
        fs = FeatureSet()
        sf_min = SingleFilter("age", "min", {"value": 18})
        sf_max = SingleFilter("age", "max", {"value": 65})
        fs.add_filters({sf_min, sf_max})

        result = fs.get_filter("age")

        assert result is not None
        assert isinstance(result, SingleFilter)
        assert result.name == "age"


class TestGetFilterValue:
    """Tests for get_filter_value method."""

    def test_returns_none_when_filters_is_none(self) -> None:
        """FeatureSet with no filters should return None."""
        fs = FeatureSet()

        result = fs.get_filter_value("status")

        assert result is None

    def test_returns_none_when_no_match(self) -> None:
        """Querying a column with no matching filter should return None."""
        fs = FeatureSet()
        sf = SingleFilter("age", "min", {"value": 18})
        fs.add_filters({sf})

        result = fs.get_filter_value("status")

        assert result is None

    def test_returns_value_for_equal_filter(self) -> None:
        """Should return the parameter value for a string-valued filter."""
        fs = FeatureSet()
        sf = SingleFilter("status", "equal", {"value": "active"})
        fs.add_filters({sf})

        result = fs.get_filter_value("status")

        assert result == "active"

    def test_returns_value_for_numeric_filter(self) -> None:
        """Should return the parameter value for a numeric-valued filter."""
        fs = FeatureSet()
        sf = SingleFilter("score", "equal", {"value": 42})
        fs.add_filters({sf})

        result = fs.get_filter_value("score")

        assert result == 42

    def test_returns_none_when_filter_has_no_value_key(self) -> None:
        """A range filter with no 'value' key should return None."""
        fs = FeatureSet()
        sf_range = SingleFilter("age", "range", {"min": 0, "max": 100})
        fs.add_filters({sf_range})

        result = fs.get_filter_value("age")

        assert result is None
