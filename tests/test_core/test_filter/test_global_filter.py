from datetime import datetime, timezone

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.global_filter import GlobalFilter
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_core.filter.filter_parameter import FilterParameterImpl


class TestGlobalFilter:
    def setup_method(self) -> None:
        """Set up test variables."""
        self.global_filter = GlobalFilter()
        self.feature = Feature("age")
        self.filter_type = FilterTypeEnum.range
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
        self.global_filter.add_filter(Feature("salary"), FilterTypeEnum.equal, {"value": 50000})

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
        self.filter_type = FilterTypeEnum.range
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
        self.global_filter.add_filter(Feature("salary"), FilterTypeEnum.equal, {"value": 50000})

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
        time_filter_feature = "custom_time_filter"
        time_travel_filter_feature = "custom_time_travel_filter"

        self.global_filter.add_time_and_time_travel_filters(
            event_from,
            event_to,
            valid_from,
            valid_to,
            time_filter_feature=time_filter_feature,
            time_travel_filter_feature=time_travel_filter_feature,
        )

        # Assert that two filters have been added
        assert len(self.global_filter.filters) == 2

        # Check that the filters have the correct feature names
        filter_features = {f.filter_feature.name for f in self.global_filter.filters}

        assert time_filter_feature in filter_features
        assert time_travel_filter_feature in filter_features

    def test_add_time_filters_with_custom_feature(self) -> None:
        """Test adding time filters with a custom feature name."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        time_filter_feature = "custom_time_filter"

        self.global_filter.add_time_and_time_travel_filters(
            event_from, event_to, time_filter_feature=time_filter_feature
        )

        # Assert that one filter has been added
        assert len(self.global_filter.filters) == 1

        # Check that the filter has the correct feature name
        added_filter = next(iter(self.global_filter.filters)).filter_feature
        assert added_filter.name.name == time_filter_feature

    def test_add_time_travel_filters_with_custom_feature(self) -> None:
        """Test adding time-travel filters with a custom feature name."""
        event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
        event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
        valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
        valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)
        time_travel_filter_feature = "custom_time_travel_filter"

        self.global_filter.add_time_and_time_travel_filters(
            event_from, event_to, valid_from, valid_to, time_travel_filter_feature=time_travel_filter_feature
        )

        # Assert that two filters have been added
        assert len(self.global_filter.filters) == 2

        # Check that the time-travel filter has the correct feature name
        filter_features = {f.filter_feature.name for f in self.global_filter.filters}
        assert time_travel_filter_feature in filter_features

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

    def test_check_and_convert_time_info(self) -> None:
        """Test the _check_and_convert_time_info method."""
        time_with_tz = datetime(2023, 1, 1, tzinfo=timezone.utc)
        converted_time = self.global_filter._check_and_convert_time_info(time_with_tz)

        # Assert that the time is correctly converted to ISO 8601 format in UTC
        assert converted_time == "2023-01-01T00:00:00+00:00"

    def test_check_and_convert_time_info_without_tz(self) -> None:
        """Test the _check_and_convert_time_info method with missing timezone info."""
        time_without_tz = datetime(2023, 1, 1)

        with pytest.raises(ValueError):
            self.global_filter._check_and_convert_time_info(time_without_tz)
