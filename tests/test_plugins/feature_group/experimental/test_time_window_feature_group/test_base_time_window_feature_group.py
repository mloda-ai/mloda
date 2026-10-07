"""
Tests for the TimeWindowFeatureGroup class.
"""

import pytest

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
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

    def test_invalid_named_value_is_rejected_despite_valid_explicit_options(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Valid window options no longer hide a bogus window function carried by the name."""
        options = Options(
            context={
                TimeWindowFeatureGroup.WINDOW_FUNCTION: "avg",
                TimeWindowFeatureGroup.WINDOW_SIZE: 7,
                TimeWindowFeatureGroup.TIME_UNIT: "day",
            }
        )
        result = TimeWindowFeatureGroup.match_feature_group_criteria("x__bogus_7_day_window", options)

        assert result is False
        recorded = [r.reason for r in rejection_window.values()]
        assert len(recorded) == 1
        assert "bogus" in recorded[0]

    def test_declared_window_function_contradicting_the_name_aborts(self) -> None:
        """On a secondary-free named capture, the name says sum, the option says max."""
        options = Options(context={TimeWindowFeatureGroup.WINDOW_FUNCTION: "max"})

        with pytest.raises(ValueError) as exc_info:
            TimeWindowFeatureGroup.match_feature_group_criteria("x__sum_7_day_window", options)

        message = str(exc_info.value)
        assert TimeWindowFeatureGroup.WINDOW_FUNCTION in message
        assert "max" in message
        assert "sum" in message

    def test_declared_window_values_agreeing_with_the_name_match(self) -> None:
        options = Options(
            context={
                TimeWindowFeatureGroup.WINDOW_FUNCTION: "sum",
                TimeWindowFeatureGroup.WINDOW_SIZE: 7,
                TimeWindowFeatureGroup.TIME_UNIT: "day",
            }
        )

        assert TimeWindowFeatureGroup.match_feature_group_criteria("x__sum_7_day_window", options) is True

    def test_name_path_source_count_above_max_is_a_recorded_non_match(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The name path validates the source count via the mixin (MAX_IN_FEATURES = 1)."""
        name = "a&b__sum_7_day_window"
        assert not TimeWindowFeatureGroup.match_feature_group_criteria(name, Options())
        reason = rejection_window["TimeWindowFeatureGroup"].reason
        assert "at most 1" in reason
        assert "found 2" in reason

        with pytest.raises(ValueError) as exc_info:
            PandasTimeWindowFeatureGroup().input_features(Options(), FeatureName(name))
        assert str(exc_info.value) == reason

    def test_input_features(self) -> None:
        """Test input_features method."""
        from typing import Any
        from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

        class ConcreteTimeWindowFeatureGroup(TimeWindowFeatureGroup):
            @classmethod
            def _check_reference_time_column_exists(cls, data: Any, reference_time_column: str) -> None:
                pass

            @classmethod
            def _check_reference_time_column_is_datetime(cls, data: Any, reference_time_column: str) -> None:
                pass

            @classmethod
            def _get_available_columns(cls, data: Any) -> set[str]:
                return set()

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
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
                in_features: list[str],
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

        source = Feature(
            "sales_max", Options(group={"aggregation_type": "max", DefaultOptionKeys.in_features: "sales"})
        )
        chained = Options(
            context={
                "window_function": "sum",
                "window_size": 2,
                "time_unit": "day",
                DefaultOptionKeys.in_features: frozenset([source]),
            }
        )
        result = feature_group.input_features(chained, FeatureName("tw"))
        assert result is not None
        by_name = {str(f.name): f for f in result}
        time_feature = by_name[DefaultOptionKeys.reference_time.value]
        assert dict(time_feature.options.group) == {"aggregation_type": "max"}
        assert DefaultOptionKeys.in_features not in time_feature.options.group

    def test_get_reference_time_column_default(self) -> None:
        """Test get_reference_time_column method returns default column name when no options provided."""
        # Test with no options - should return default column name
        assert TimeWindowFeatureGroup.get_reference_time_column() == DefaultOptionKeys.reference_time

    def test_get_reference_time_column_custom(self) -> None:
        """Test get_reference_time_column method returns custom column name when reference_time option is set."""
        # Test with custom options using DefaultOptionKeys.reference_time
        options = Options()
        options.add_to_group(DefaultOptionKeys.reference_time, "custom_time_column")
        assert TimeWindowFeatureGroup.get_reference_time_column(options) == "custom_time_column"

        # Test with custom options using DefaultOptionKeys.reference_time.value
        options = Options()
        options.add_to_group(DefaultOptionKeys.reference_time, "another_custom_column")
        assert TimeWindowFeatureGroup.get_reference_time_column(options) == "another_custom_column"

    def test_get_reference_time_column_invalid_type(self) -> None:
        """Test get_reference_time_column method raises ValueError when option value is not a string."""
        # Test with invalid options (non-string value)
        options = Options()
        options.add_to_group(DefaultOptionKeys.reference_time, 123)  # Not a string
        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.get_reference_time_column(options)

    def test_check_reference_time_column_exists_method_exists(self) -> None:
        """Test that _check_reference_time_column_exists method can be called (verifies method rename)."""
        from typing import Any

        class ConcreteTimeWindowFeatureGroup(TimeWindowFeatureGroup):
            @classmethod
            def _check_reference_time_column_exists(cls, data: Any, time_filter_feature: str) -> None:
                # Simple implementation for testing
                pass

            @classmethod
            def _check_reference_time_column_is_datetime(cls, data: Any, time_filter_feature: str) -> None:
                pass

            @classmethod
            def _get_available_columns(cls, data: Any) -> set[str]:
                return set()

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
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
                in_features: list[str],
                time_filter_feature: str | None = None,
            ) -> Any:
                return None

        # Verify that the method can be called without error
        concrete_fg = ConcreteTimeWindowFeatureGroup()
        concrete_fg._check_reference_time_column_exists(None, "test_column")

    def test_check_reference_time_column_is_datetime_method_exists(self) -> None:
        """Test that _check_reference_time_column_is_datetime method can be called (verifies method rename)."""
        from typing import Any

        class ConcreteTimeWindowFeatureGroup(TimeWindowFeatureGroup):
            @classmethod
            def _check_reference_time_column_exists(cls, data: Any, time_filter_feature: str) -> None:
                pass

            @classmethod
            def _check_reference_time_column_is_datetime(cls, data: Any, time_filter_feature: str) -> None:
                # Simple implementation for testing
                pass

            @classmethod
            def _get_available_columns(cls, data: Any) -> set[str]:
                return set()

            @classmethod
            def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
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
                in_features: list[str],
                time_filter_feature: str | None = None,
            ) -> Any:
                return None

        # Verify that the method can be called without error
        concrete_fg = ConcreteTimeWindowFeatureGroup()
        concrete_fg._check_reference_time_column_is_datetime(None, "test_column")

    def test_has_valid_time_window_suffix_valid(self) -> None:
        """Test that _has_valid_time_window_suffix returns True for valid feature names."""
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("temperature__avg_3_day_window") is True
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("humidity__max_7_hour_window") is True
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("pressure__min_2_minute_window") is True
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("cpu__sum_10_second_window") is True
        # Chained feature: the last __ segment should still be valid
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("price__mean_imputed__sum_7_day_window") is True

    def test_has_valid_time_window_suffix_invalid(self) -> None:
        """Test that _has_valid_time_window_suffix returns False for invalid feature names."""
        # No double underscore separator
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("invalid_feature_name") is False
        # Missing "window" suffix
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("temp__avg_3_day") is False
        # Unsupported window function
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("temp__invalid_3_day_window") is False
        # Unsupported time unit
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("temp__avg_3_invalid_window") is False
        # Non-numeric window size
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("temp__avg_zero_day_window") is False
        # Zero window size
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix("temp__avg_0_day_window") is False

    def test_extract_time_window_params_string_based(self) -> None:
        """Test _extract_time_window_params with a string-parseable feature name."""
        feature = Feature("temperature__avg_3_day_window")
        result = TimeWindowFeatureGroup._extract_time_window_params(feature)
        assert result == ("avg", 3, "day")

    def test_extract_time_window_params_chained_name(self) -> None:
        feature = Feature("price__mean_imputed__sum_7_day_window")
        assert TimeWindowFeatureGroup._extract_time_window_params(feature) == ("sum", 7, "day")

    def test_extract_time_window_params_config_fallback(self) -> None:
        """Test _extract_time_window_params falls back to configuration-based options."""
        options = Options()
        options.add_to_group(TimeWindowFeatureGroup.WINDOW_FUNCTION, "sum")
        options.add_to_group(TimeWindowFeatureGroup.WINDOW_SIZE, "5")
        options.add_to_group(TimeWindowFeatureGroup.TIME_UNIT, "hour")
        feature = Feature("some_feature", options)
        result = TimeWindowFeatureGroup._extract_time_window_params(feature)
        assert result == ("sum", 5, "hour")

    @pytest.mark.parametrize("name", ["x__mean_imputed__sum_7_day_window", "a__b__c__sum_7_day_window"])
    def test_chained_source_name_parses_from_the_last_suffix(self, name: str) -> None:
        assert TimeWindowFeatureGroup.parse_time_window_prefix(name) == ("sum", 7, "day")

    @pytest.mark.parametrize(
        "name",
        [
            "x__sum_7_day_extra_window",
            "x__su_m_7_day_window",
            "x__sum_7_da_y_window",
            "x__sum_0_day_window",
            "x__mean_imputed__sum_7_day_extra_window",
        ],
    )
    def test_names_the_hand_parser_rejected_are_still_rejected(self, name: str) -> None:
        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.parse_time_window_prefix(name)
        assert TimeWindowFeatureGroup._has_valid_time_window_suffix(name) is False

    def test_empty_source_raises(self) -> None:
        name = "__sum_7_day_window"
        with pytest.raises(ValueError):
            TimeWindowFeatureGroup.parse_time_window_prefix(name)
        with pytest.raises(ValueError):
            TimeWindowFeatureGroup._has_valid_time_window_suffix(name)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("x__sum_7_day_win", ("sum", 7, "day")), ("x__mean_imputed__max_3_hour_win", ("max", 3, "hour"))],
    )
    def test_parse_time_window_prefix_follows_an_overridden_prefix_pattern(
        self, name: str, expected: tuple[str, int, str]
    ) -> None:
        """Parts are read from PREFIX_PATTERN."""

        class WinPatternGroup(TimeWindowFeatureGroup):
            PREFIX_PATTERN = r".*__(?P<window_function>[\w]+)_(?P<window_size>\d+)_(?P<time_unit>[\w]+)_win$"

        assert WinPatternGroup.parse_time_window_prefix(name) == expected
        assert WinPatternGroup._has_valid_time_window_suffix(name) is True
        with pytest.raises(ValueError):
            WinPatternGroup.parse_time_window_prefix("x__sum_7_day_window")
