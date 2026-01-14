"""
Tests for the TimeWindowFeatureGroup class.
"""

import pytest

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda.provider import FeatureChainParser


class TestTimeWindowFeatureGroup:
    """Tests for the TimeWindowFeatureGroup class."""

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Test valid feature names
        feature_name = "temperature__avg_3_day_window"

        # Test that the PREFIX_PATTERN is correctly defined
        assert hasattr(TimeWindowFeatureGroup, "PREFIX_PATTERN")

        # Test that FeatureChainParser methods work with the PREFIX_PATTERN
        assert (
            FeatureChainParser.extract_in_feature(feature_name, TimeWindowFeatureGroup.PREFIX_PATTERN) == "temperature"
        )

    def test_parse_time_window_prefix(self) -> None:
        """Test parsing of time window prefix into components."""
        window_function, window_size, time_unit = TimeWindowFeatureGroup.parse_time_window_prefix(
            "temperature__avg_3_day_window"
        )
        assert window_function == "avg"
        assert window_size == 3
        assert time_unit == "day"

        window_function, window_size, time_unit = TimeWindowFeatureGroup.parse_time_window_prefix(
            "humidity__max_7_hour_window"
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
        assert TimeWindowFeatureGroup.get_window_function("temperature__avg_3_day_window") == "avg"
        assert TimeWindowFeatureGroup.get_window_function("humidity__max_7_hour_window") == "max"
        assert TimeWindowFeatureGroup.get_window_function("pressure__min_2_day_window") == "min"
        assert TimeWindowFeatureGroup.get_window_function("wind_speed__sum_4_day_window") == "sum"

    def test_get_window_size(self) -> None:
        """Test extraction of window size from feature name."""
        assert TimeWindowFeatureGroup.get_window_size("temperature__avg_3_day_window") == 3
        assert TimeWindowFeatureGroup.get_window_size("humidity__max_7_hour_window") == 7
        assert TimeWindowFeatureGroup.get_window_size("pressure__min_2_day_window") == 2
        assert TimeWindowFeatureGroup.get_window_size("wind_speed__sum_4_day_window") == 4

    def test_get_time_unit(self) -> None:
        """Test extraction of time unit from feature name."""
        assert TimeWindowFeatureGroup.get_time_unit("temperature__avg_3_day_window") == "day"
        assert TimeWindowFeatureGroup.get_time_unit("humidity__max_7_hour_window") == "hour"
        assert TimeWindowFeatureGroup.get_time_unit("pressure__min_2_minute_window") == "minute"
        assert TimeWindowFeatureGroup.get_time_unit("wind_speed__sum_4_second_window") == "second"

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert TimeWindowFeatureGroup.match_feature_group_criteria("temperature__avg_3_day_window", options)
        assert TimeWindowFeatureGroup.match_feature_group_criteria("humidity__max_7_hour_window", options)
        assert TimeWindowFeatureGroup.match_feature_group_criteria("pressure__min_2_day_window", options)
        assert TimeWindowFeatureGroup.match_feature_group_criteria("wind_speed__sum_4_day_window", options)

        # Test with FeatureName objects
        assert TimeWindowFeatureGroup.match_feature_group_criteria(
            FeatureName("temperature__avg_3_day_window"), options
        )
        assert TimeWindowFeatureGroup.match_feature_group_criteria(FeatureName("humidity__max_7_hour_window"), options)

        # Test with invalid feature names
        assert not TimeWindowFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not TimeWindowFeatureGroup.match_feature_group_criteria("avg_day_window_temperature", options)
        assert not TimeWindowFeatureGroup.match_feature_group_criteria("avg_3_invalid_window_temperature", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        from typing import Any, List, Set
        from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

        class ConcreteTimeWindowFeatureGroup(TimeWindowFeatureGroup):
            @classmethod
            def _check_reference_time_column_exists(cls, data: Any, reference_time_column: str) -> None:
                pass

            @classmethod
            def _check_reference_time_column_is_datetime(cls, data: Any, reference_time_column: str) -> None:
                pass

            @classmethod
            def _get_available_columns(cls, data: Any) -> Set[str]:
                return set()

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
                pass

            @classmethod
            def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
                return data

            @classmethod
            def _perform_window_operation(
                cls,
                data: Any,
                window_function: str,
                window_size: int,
                time_unit: str,
                in_features: List[str],
                time_filter_feature: str | None = None,
            ) -> Any:
                return None

        options = Options()
        feature_group = ConcreteTimeWindowFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("temperature__avg_3_day_window"))
        assert input_features == {
            Feature("temperature"),
            Feature(DefaultOptionKeys.reference_time),
        }

        input_features = feature_group.input_features(options, FeatureName("humidity__max_7_hour_window"))
        assert input_features == {Feature("humidity"), Feature(DefaultOptionKeys.reference_time)}

        input_features = feature_group.input_features(options, FeatureName("pressure__min_2_day_window"))
        assert input_features == {Feature("pressure"), Feature(DefaultOptionKeys.reference_time)}

        input_features = feature_group.input_features(options, FeatureName("wind_speed__sum_4_day_window"))
        assert input_features == {
            Feature("wind_speed"),
            Feature(DefaultOptionKeys.reference_time),
        }

    def test_get_reference_time_column_default(self) -> None:
        """Test get_reference_time_column method returns default column name when no options provided."""
        # Test with no options - should return default column name
        assert TimeWindowFeatureGroup.get_reference_time_column() == DefaultOptionKeys.reference_time.value

    def test_get_reference_time_column_custom(self) -> None:
        """Test get_reference_time_column method returns custom column name when reference_time option is set."""
        # Test with custom options using DefaultOptionKeys.reference_time
        options = Options()
        options.add(DefaultOptionKeys.reference_time, "custom_time_column")
        assert TimeWindowFeatureGroup.get_reference_time_column(options) == "custom_time_column"

        # Test with custom options using DefaultOptionKeys.reference_time.value
        options = Options()
        options.add(DefaultOptionKeys.reference_time.value, "another_custom_column")
        assert TimeWindowFeatureGroup.get_reference_time_column(options) == "another_custom_column"

    def test_get_reference_time_column_invalid_type(self) -> None:
        """Test get_reference_time_column method raises ValueError when option value is not a string."""
        # Test with invalid options (non-string value)
        options = Options()
        options.add(DefaultOptionKeys.reference_time.value, 123)  # Not a string
        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.get_reference_time_column(options)

    def test_check_reference_time_column_exists_method_exists(self) -> None:
        """Test that _check_reference_time_column_exists method can be called (verifies method rename)."""
        from typing import Any, List, Set

        class ConcreteTimeWindowFeatureGroup(TimeWindowFeatureGroup):
            @classmethod
            def _check_reference_time_column_exists(cls, data: Any, time_filter_feature: str) -> None:
                # Simple implementation for testing
                pass

            @classmethod
            def _check_reference_time_column_is_datetime(cls, data: Any, time_filter_feature: str) -> None:
                pass

            @classmethod
            def _get_available_columns(cls, data: Any) -> Set[str]:
                return set()

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
                pass

            @classmethod
            def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
                return data

            @classmethod
            def _perform_window_operation(
                cls,
                data: Any,
                window_function: str,
                window_size: int,
                time_unit: str,
                in_features: List[str],
                time_filter_feature: str | None = None,
            ) -> Any:
                return None

        # Verify that the method can be called without error
        concrete_fg = ConcreteTimeWindowFeatureGroup()
        concrete_fg._check_reference_time_column_exists(None, "test_column")

    def test_check_reference_time_column_is_datetime_method_exists(self) -> None:
        """Test that _check_reference_time_column_is_datetime method can be called (verifies method rename)."""
        from typing import Any, List, Set

        class ConcreteTimeWindowFeatureGroup(TimeWindowFeatureGroup):
            @classmethod
            def _check_reference_time_column_exists(cls, data: Any, time_filter_feature: str) -> None:
                pass

            @classmethod
            def _check_reference_time_column_is_datetime(cls, data: Any, time_filter_feature: str) -> None:
                # Simple implementation for testing
                pass

            @classmethod
            def _get_available_columns(cls, data: Any) -> Set[str]:
                return set()

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
                pass

            @classmethod
            def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
                return data

            @classmethod
            def _perform_window_operation(
                cls,
                data: Any,
                window_function: str,
                window_size: int,
                time_unit: str,
                in_features: List[str],
                time_filter_feature: str | None = None,
            ) -> Any:
                return None

        # Verify that the method can be called without error
        concrete_fg = ConcreteTimeWindowFeatureGroup()
        concrete_fg._check_reference_time_column_is_datetime(None, "test_column")
