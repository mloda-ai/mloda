from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from mloda.user import Feature
from mloda.user import GlobalFilter
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda.core.filter.filter_parameter import FilterParameterImpl


class TestGlobalFilter:
    def setup_method(self) -> None:
        """Set up test variables."""
        self.global_filter = GlobalFilter()
        self.feature = Feature("age")
        self.filter_type = FilterType.RANGE
        self.parameter = {"min": 25, "max": 50}

    def test_add_single_filter(self) -> None:
        """Test adding a filter to the GlobalFilter config."""
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)

        # Assert that the filter has been added
        assert len(self.global_filter.filters) == 1
        added_filter = next(iter(self.global_filter.filters))  # Get the added SingleFilter
        assert isinstance(added_filter, SingleFilter)
        assert added_filter.filter_feature == self.feature
        assert added_filter.filter_type == "range"
        assert isinstance(added_filter.parameter, FilterParameterImpl)
        assert added_filter.parameter.min_value == 25
        assert added_filter.parameter.max_value == 50

    def test_adding_identical_filters(self) -> None:
        """Test that adding identical filters doesn't duplicate them in the config set."""
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)

        # Assert that only one filter is added due to set behavior
        assert len(self.global_filter.filters) == 1

    def test_adding_different_filters(self) -> None:
        """Test that adding different filters works correctly."""
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)
        self.global_filter.add_filter(Feature("salary"), FilterType.EQUAL, {"value": 50000})

        # Assert that two distinct filters have been added
        assert len(self.global_filter.filters) == 2

    def test_filter_config_empty(self) -> None:
        """Test that the GlobalFilter starts with an empty config."""
        assert len(self.global_filter.filters) == 0


class TestGlobalFilterTimeTravel:
    def setup_method(self) -> None:
        """Set up test variables."""
        self.global_filter = GlobalFilter()
        self.feature = Feature("age")
        self.filter_type = FilterType.RANGE
        self.parameter = {"min": 25, "max": 50}

    def test_add_single_filter(self) -> None:
        """Test adding a filter to the GlobalFilter config."""
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)

        # Assert that the filter has been added
        assert len(self.global_filter.filters) == 1
        added_filter = next(iter(self.global_filter.filters))  # Get the added SingleFilter
        assert isinstance(added_filter, SingleFilter)
        assert added_filter.filter_feature == self.feature
        assert added_filter.filter_type == "range"
        assert isinstance(added_filter.parameter, FilterParameterImpl)
        assert added_filter.parameter.min_value == 25
        assert added_filter.parameter.max_value == 50

    def test_adding_identical_filters(self) -> None:
        """Test that adding identical filters doesn't duplicate them in the config set."""
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)

        # Assert that only one filter is added due to set behavior
        assert len(self.global_filter.filters) == 1

    def test_adding_different_filters(self) -> None:
        """Test that adding different filters works correctly."""
        self.global_filter.add_filter(self.feature, self.filter_type, self.parameter)
        self.global_filter.add_filter(Feature("salary"), FilterType.EQUAL, {"value": 50000})

        # Assert that two distinct filters have been added
        assert len(self.global_filter.filters) == 2

    def test_filter_config_empty(self) -> None:
        """Test that the GlobalFilter starts with an empty config."""
        assert len(self.global_filter.filters) == 0

    def test_add_time_and_time_travel_filters(self) -> None:
        """Test adding time and time-travel filters."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
        valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)

        self.global_filter.add_time_and_time_travel_filters(event_from, event_to, valid_from, valid_to)

        # Assert that two filters have been added
        assert len(self.global_filter.filters) == 2

    def test_add_time_and_time_travel_filters_with_custom_features(self) -> None:
        """Test adding time and time-travel filters with custom feature names."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
        valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)
        event_time_column = "custom_time_filter"
        validity_time_column = "custom_time_travel_filter"

        self.global_filter.add_time_and_time_travel_filters(
            event_from,
            event_to,
            valid_from,
            valid_to,
            event_time_column=event_time_column,
            validity_time_column=validity_time_column,
        )

        # Assert that two filters have been added
        assert len(self.global_filter.filters) == 2

        # Check that the filters have the correct feature names
        filter_features = {f.filter_feature.name for f in self.global_filter.filters}

        assert event_time_column in filter_features
        assert validity_time_column in filter_features

    def test_add_time_filters_with_custom_feature(self) -> None:
        """Test adding time filters with a custom feature name."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        event_time_column = "custom_time_filter"

        self.global_filter.add_time_and_time_travel_filters(event_from, event_to, event_time_column=event_time_column)

        # Assert that one filter has been added
        assert len(self.global_filter.filters) == 1

        # Check that the filter has the correct feature name
        added_filter = next(iter(self.global_filter.filters)).filter_feature
        assert added_filter.name == event_time_column

    def test_add_time_travel_filters_with_custom_feature(self) -> None:
        """Test adding time-travel filters with a custom feature name."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
        valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)
        validity_time_column = "custom_time_travel_filter"

        self.global_filter.add_time_and_time_travel_filters(
            event_from, event_to, valid_from, valid_to, validity_time_column=validity_time_column
        )

        # Assert that two filters have been added
        assert len(self.global_filter.filters) == 2

        # Check that the time-travel filter has the correct feature name
        filter_features = {f.filter_feature.name for f in self.global_filter.filters}
        assert validity_time_column in filter_features

    def test_add_time_filters_without_validity(self) -> None:
        """Test adding time filters without validity period."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)

        self.global_filter.add_time_and_time_travel_filters(event_from, event_to)

        # Assert that one filter has been added
        assert len(self.global_filter.filters) == 1

    def test_add_time_filters_with_invalid_validity(self) -> None:
        """Test adding time filters with invalid validity period."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError):
            self.global_filter.add_time_and_time_travel_filters(event_from, event_to, valid_from)

    def test_normalize_to_utc(self) -> None:
        """Test the _normalize_to_utc method returns a UTC-normalized datetime (not a string)."""
        # Identity case: already-UTC input passes through unchanged.
        time_with_tz = datetime(2023, 1, 1, tzinfo=timezone.utc)
        converted_time = self.global_filter._normalize_to_utc(time_with_tz)
        assert converted_time == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert converted_time.tzinfo == timezone.utc

        # Conversion case: a non-UTC tz must shift to UTC.
        time_offset = datetime(2023, 1, 1, 12, tzinfo=timezone(timedelta(hours=5)))
        converted_offset = self.global_filter._normalize_to_utc(time_offset)
        assert converted_offset == datetime(2023, 1, 1, 7, tzinfo=timezone.utc)
        assert converted_offset.tzinfo == timezone.utc

    def test_normalize_to_utc_without_tz(self) -> None:
        """Test the _normalize_to_utc method with missing timezone info."""
        time_without_tz = datetime(2023, 1, 1)

        with pytest.raises(ValueError):
            self.global_filter._normalize_to_utc(time_without_tz)

    def test_add_time_filters_with_event_time_column(self) -> None:
        """Test adding time filters with the new event_time_column parameter name."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        event_time_column = "custom_event_time"

        self.global_filter.add_time_and_time_travel_filters(event_from, event_to, event_time_column=event_time_column)

        # Assert that one filter has been added
        assert len(self.global_filter.filters) == 1

        # Check that the filter has the correct feature name
        added_filter = next(iter(self.global_filter.filters)).filter_feature
        assert added_filter.name == event_time_column

    def test_add_time_filters_with_validity_time_column(self) -> None:
        """Test adding time-travel filters with the new validity_time_column parameter name."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
        valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)
        validity_time_column = "custom_validity_time"

        self.global_filter.add_time_and_time_travel_filters(
            event_from, event_to, valid_from, valid_to, validity_time_column=validity_time_column
        )

        # Assert that two filters have been added
        assert len(self.global_filter.filters) == 2

        # Check that the time-travel filter has the correct feature name
        filter_features = {f.filter_feature.name for f in self.global_filter.filters}
        assert validity_time_column in filter_features

    def test_add_time_filters_with_both_new_parameter_names(self) -> None:
        """Test adding filters with both event_time_column and validity_time_column parameter names."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
        valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)
        event_time_column = "custom_event_time"
        validity_time_column = "custom_validity_time"

        self.global_filter.add_time_and_time_travel_filters(
            event_from,
            event_to,
            valid_from,
            valid_to,
            event_time_column=event_time_column,
            validity_time_column=validity_time_column,
        )

        # Assert that two filters have been added
        assert len(self.global_filter.filters) == 2

        # Check that both filters have the correct feature names
        filter_features = {f.filter_feature.name for f in self.global_filter.filters}
        assert event_time_column in filter_features
        assert validity_time_column in filter_features


class TestGlobalFilterCategoricalInclusion:
    """Collection filter values must survive `GlobalFilter.add_filter` (issue #664).

    `add_filter` puts the `SingleFilter` into a `set`, which hashes the parameter. Collection
    values therefore need a hashable internal representation, but the public `values` accessor
    must still hand back a `list` for the filter engines that consume it.
    """

    def setup_method(self) -> None:
        self.global_filter = GlobalFilter()

    def _only_filter(self) -> SingleFilter:
        assert len(self.global_filter.filters) == 1
        return next(iter(self.global_filter.filters))

    def test_add_filter_with_list_value(self) -> None:
        """Test add_filter accepts a list value without raising TypeError: unhashable type."""
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, {"values": ["EU", "NA"]})

        added_filter = self._only_filter()
        assert added_filter.filter_type == "categorical_inclusion"
        assert isinstance(added_filter.parameter, FilterParameterImpl)

    def test_add_filter_with_list_value_exposes_values_as_list(self) -> None:
        """Test the value round-trips through add_filter as a list, not a tuple."""
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, {"values": ["EU", "NA"]})

        values = self._only_filter().parameter.values
        assert isinstance(values, list)
        assert values == ["EU", "NA"]

    def test_add_filter_with_set_value(self) -> None:
        """Test add_filter accepts a set value and exposes it as a list."""
        parameter: dict[str, Any] = {"values": {"EU", "NA"}}
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, parameter)

        values = self._only_filter().parameter.values
        assert isinstance(values, list)
        assert sorted(values) == ["EU", "NA"]

    def test_add_filter_with_string_value_is_not_exploded(self) -> None:
        """Test a scalar string value is not split into its characters."""
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, {"values": "EU"})

        values: Any = self._only_filter().parameter.values
        assert values == "EU"
        assert values != ["E", "U"]

    def test_add_filter_deduplicates_list_and_tuple_values(self) -> None:
        """Test list and tuple values normalize to the same filter and deduplicate in the set."""
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, {"values": ["EU"]})
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, {"values": ("EU",)})

        assert len(self.global_filter.filters) == 1

    def test_add_filter_keeps_distinct_collection_values_apart(self) -> None:
        """Test different collection values still produce distinct filters."""
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, {"values": ["EU"]})
        self.global_filter.add_filter("region", FilterType.CATEGORICAL_INCLUSION, {"values": ["NA"]})

        assert len(self.global_filter.filters) == 2
