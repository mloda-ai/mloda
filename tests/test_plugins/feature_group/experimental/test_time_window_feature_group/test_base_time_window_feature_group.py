"""
Tests for the TimeWindowFeatureGroup class.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser


class TestTimeWindowFeatureGroup:
    """Tests for the TimeWindowFeatureGroup class."""

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Test valid feature names
        feature_name = "avg_3_day_window__temperature"

        # Test that the PREFIX_PATTERN is correctly defined
        assert hasattr(TimeWindowFeatureGroup, "PREFIX_PATTERN")

        # Test that FeatureChainParser methods work with the PREFIX_PATTERN
        assert (
            FeatureChainParser.extract_source_feature(feature_name, TimeWindowFeatureGroup.PREFIX_PATTERN)
            == "temperature"
        )

    def test_parse_time_window_prefix(self) -> None:
        """Test parsing of time window prefix into components."""
        window_function, window_size, time_unit = TimeWindowFeatureGroup.parse_time_window_prefix(
            "avg_3_day_window__temperature"
        )
        assert window_function == "avg"
        assert window_size == 3
        assert time_unit == "day"

        window_function, window_size, time_unit = TimeWindowFeatureGroup.parse_time_window_prefix(
            "max_7_hour_window__humidity"
        )
        assert window_function == "max"
        assert window_size == 7
        assert time_unit == "hour"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.parse_time_window_prefix("invalid_feature_name")

        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.parse_time_window_prefix("avg_day_window_temperature")

        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.parse_time_window_prefix("avg_3_invalid_window_temperature")

        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.parse_time_window_prefix("invalid_3_day_window_temperature")

    def test_get_window_function(self) -> None:
        """Test extraction of window function from feature name."""
        assert TimeWindowFeatureGroup.get_window_function("avg_3_day_window__temperature") == "avg"
        assert TimeWindowFeatureGroup.get_window_function("max_7_hour_window__humidity") == "max"
        assert TimeWindowFeatureGroup.get_window_function("min_2_day_window__pressure") == "min"
        assert TimeWindowFeatureGroup.get_window_function("sum_4_day_window__wind_speed") == "sum"

    def test_get_window_size(self) -> None:
        """Test extraction of window size from feature name."""
        assert TimeWindowFeatureGroup.get_window_size("avg_3_day_window__temperature") == 3
        assert TimeWindowFeatureGroup.get_window_size("max_7_hour_window__humidity") == 7
        assert TimeWindowFeatureGroup.get_window_size("min_2_day_window__pressure") == 2
        assert TimeWindowFeatureGroup.get_window_size("sum_4_day_window__wind_speed") == 4

    def test_get_time_unit(self) -> None:
        """Test extraction of time unit from feature name."""
        assert TimeWindowFeatureGroup.get_time_unit("avg_3_day_window__temperature") == "day"
        assert TimeWindowFeatureGroup.get_time_unit("max_7_hour_window__humidity") == "hour"
        assert TimeWindowFeatureGroup.get_time_unit("min_2_minute_window__pressure") == "minute"
        assert TimeWindowFeatureGroup.get_time_unit("sum_4_second_window__wind_speed") == "second"

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert TimeWindowFeatureGroup.match_feature_group_criteria("avg_3_day_window__temperature", options)
        assert TimeWindowFeatureGroup.match_feature_group_criteria("max_7_hour_window__humidity", options)
        assert TimeWindowFeatureGroup.match_feature_group_criteria("min_2_day_window__pressure", options)
        assert TimeWindowFeatureGroup.match_feature_group_criteria("sum_4_day_window__wind_speed", options)

        # Test with FeatureName objects
        assert TimeWindowFeatureGroup.match_feature_group_criteria(
            FeatureName("avg_3_day_window__temperature"), options
        )
        assert TimeWindowFeatureGroup.match_feature_group_criteria(FeatureName("max_7_hour_window__humidity"), options)

        # Test with invalid feature names
        assert not TimeWindowFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not TimeWindowFeatureGroup.match_feature_group_criteria("avg_day_window_temperature", options)
        assert not TimeWindowFeatureGroup.match_feature_group_criteria("avg_3_invalid_window_temperature", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = TimeWindowFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("avg_3_day_window__temperature"))
        assert input_features == {
            Feature("temperature"),
            Feature(DefaultOptionKeys.reference_time),
        }

        input_features = feature_group.input_features(options, FeatureName("max_7_hour_window__humidity"))
        assert input_features == {Feature("humidity"), Feature(DefaultOptionKeys.reference_time)}

        input_features = feature_group.input_features(options, FeatureName("min_2_day_window__pressure"))
        assert input_features == {Feature("pressure"), Feature(DefaultOptionKeys.reference_time)}

        input_features = feature_group.input_features(options, FeatureName("sum_4_day_window__wind_speed"))
        assert input_features == {
            Feature("wind_speed"),
            Feature(DefaultOptionKeys.reference_time),
        }

    def test_get_time_filter_feature(self) -> None:
        """Test get_time_filter_feature method."""
        # Test with default options
        assert TimeWindowFeatureGroup.get_time_filter_feature() == DefaultOptionKeys.reference_time

        # Test with custom options
        options = Options()
        options.add(DefaultOptionKeys.reference_time, "custom_time_column")
        assert TimeWindowFeatureGroup.get_time_filter_feature(options) == "custom_time_column"

        # Test with custom options
        options = Options()
        options.add(DefaultOptionKeys.reference_time.value, "custom_time_column")
        assert TimeWindowFeatureGroup.get_time_filter_feature(options) == "custom_time_column"

        # Test with invalid options
        options = Options()
        options.add(DefaultOptionKeys.reference_time.value, 123)  # Not a string
        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.get_time_filter_feature(options)
