"""
Tests for the TimeWindowFeatureGroup class.
"""

import datetime
import math
import statistics
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import FeatureSet, SubtypeDeclaration
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.feature_group.experimental.reference_contract import same, same_on_pandas
from mloda_plugins.compute_framework.base_implementations.pandas import pandas_type_semantics
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindow
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindow
from mloda.provider import FeatureChainParser

REFERENCE_TIME = DefaultOptionKeys.reference_time.value


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
            TimeWindowFeatureGroup().input_features(Options(), FeatureName(name))
        assert str(exc_info.value) == reason

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = TimeWindowFeatureGroup()

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

        scoped = Options(
            context={
                "window_function": "sum",
                "window_size": 2,
                "time_unit": "day",
                DefaultOptionKeys.in_features: frozenset([Feature("sales", Options(group={"scope": "a"}))]),
            }
        )
        result = feature_group.input_features(scoped, FeatureName("tw"))
        assert result is not None
        by_name = {str(f.name): f for f in result}
        assert dict(by_name[DefaultOptionKeys.reference_time.value].options.group) == {"scope": "a"}

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

    def test_window_hands_the_group_time_delta_to_perform(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0], DefaultOptionKeys.reference_time: pd.date_range("2023-01-01", periods=2)})
        feature_set = FeatureSet()
        feature_set.add(Feature("x__sum_2_hour_window"))
        perform = Mock(return_value=[1.0, 3.0])
        add_column = Mock(return_value="done")

        result = TimeWindowFeatureGroup.window(
            df,
            feature_set,
            perform=perform,
            add_column=add_column,
            columns=PandasDataFrame.extract_column_names,
            column_semantics=pandas_type_semantics.column_semantics,
        )

        span = TimeWindowFeatureGroup._get_time_delta(2, "hour")
        perform.assert_called_once_with(df, "sum", span, ["x"], DefaultOptionKeys.reference_time)
        add_column.assert_called_once_with(df, "x__sum_2_hour_window", [1.0, 3.0])
        assert result == "done"

    @pytest.mark.parametrize(
        ("impl", "data"),
        [
            (PandasTimeWindow, pd.DataFrame({"b": [1], "a": [2], REFERENCE_TIME: [pd.Timestamp(0)]})),
            (PyArrowTimeWindow, pa.table({"b": [1], "a": [2], REFERENCE_TIME: [pd.Timestamp(0)]})),
        ],
        ids=["pandas", "pyarrow"],
    )
    def test_window_raises_when_no_source_exists_and_lists_sorted_columns(self, impl: Any, data: Any) -> None:
        feature_set = FeatureSet()
        feature_set.add(Feature("missing__sum_2_hour_window"))
        expected = "None of the source features ['missing'] found in data. Available columns: "
        with pytest.raises(ValueError) as exc_info:
            impl.calculate(TimeWindowFeatureGroup, data, feature_set, None)
        assert str(exc_info.value) == expected + str(sorted(["b", "a", REFERENCE_TIME], key=str))

    def test_window_raises_one_wording_for_a_missing_reference_time(self) -> None:
        df = pd.DataFrame({"x": [1.0]})
        feature_set = FeatureSet()
        feature_set.add(Feature("x__sum_2_hour_window"))
        with pytest.raises(ValueError, match="Reference time column 'reference_time' not found in data. Please ensure"):
            TimeWindowFeatureGroup.window(
                df,
                feature_set,
                perform=Mock(),
                add_column=Mock(),
                columns=PandasDataFrame.extract_column_names,
                column_semantics=pandas_type_semantics.column_semantics,
            )

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


WINDOW_UNIVERSE = frozenset(TimeWindowFeatureGroup.WINDOW_FUNCTIONS)


class TestTimeWindowSubtypeDeclaration:
    """TimeWindowFeatureGroup declares its window functions via SUBTYPES."""

    def test_declaration_key_is_window_function(self) -> None:
        declaration = TimeWindowFeatureGroup.SUBTYPES
        assert isinstance(declaration, SubtypeDeclaration)
        assert declaration.key == TimeWindowFeatureGroup.WINDOW_FUNCTION
        assert declaration.key == "window_function"

    def test_universe_matches_declared_window_functions(self) -> None:
        assert TimeWindowFeatureGroup.subtype_universe() == WINDOW_UNIVERSE

    def test_resolves_from_feature_name(self) -> None:
        assert TimeWindowFeatureGroup.resolve_subtype("temperature__avg_3_day_window", Options()) == "avg"

    def test_resolves_from_options_path(self) -> None:
        options = Options(context={TimeWindowFeatureGroup.WINDOW_FUNCTION: "max"})
        assert TimeWindowFeatureGroup.resolve_subtype("window_config_placeholder", options) == "max"


class TestTimeWindowBehaviorNeutral:
    """The declaration is behavior-neutral: every window function stays supported on both frameworks."""

    @pytest.mark.parametrize("framework", [PandasDataFrame, PyArrowTable], ids=lambda fw: fw.__name__)
    def test_every_window_function_stays_supported(self, framework: Any) -> None:
        assert TimeWindowFeatureGroup.subtype_universe() == WINDOW_UNIVERSE
        for window_function in TimeWindowFeatureGroup.WINDOW_FUNCTIONS:
            supported = TimeWindowFeatureGroup.supports_compute_framework(
                f"temperature__{window_function}_3_day_window", Options(), framework
            )
            assert supported is True, f"'{window_function}' must stay supported on {framework.__name__}"


_NAN = float("nan")
_DAY = datetime.timedelta(days=1)
_START = datetime.datetime(2024, 1, 1)

_DIVERGENT_DECLARATIONS: dict[type[Implementation[Any]], dict[str, str]] = {
    PandasTimeWindow: {"sum": "gives 0 for a multi-column row with no value"},
    PyArrowTimeWindow: {},
}

# (columns, span in days): NaN and null inside windows; a NaN next to values inside one column's window;
# rows with no value in any column.
_SHAPES: dict[str, tuple[dict[str, list[float | None]], int]] = {
    "single": ({"a": [1.0, 2.0, _NAN, None, 3.0, 4.0, None]}, 3),
    "multi_nan_beside_values": ({"a": [1.0, _NAN, 2.0, 3.0], "b": [10.0, 20.0, 30.0, 40.0]}, 2),
    "multi_rows_without_value": ({"a": [None, 1.0, None, 2.0], "b": [None, 3.0, None, None]}, 1),
}


def _is_missing(cell: float | None) -> bool:
    return cell is None or math.isnan(cell)


def _column_window_result(window_function: str, window: list[float | None]) -> float | int | None:
    """Reference for one column and one window: null and NaN are missing and skipped; first/last keep the cell."""
    clean: list[float] = [cell for cell in window if cell is not None and not math.isnan(cell)]
    if window_function == "count":
        return len(clean)
    if window_function == "first":
        return window[0]
    if window_function == "last":
        return window[-1]
    if window_function in ("min", "max", "median"):
        if not clean:
            return None
        reducers: dict[str, Callable[[list[float]], float]] = {"min": min, "max": max, "median": statistics.median}
        return reducers[window_function](clean)
    if not clean:
        return None
    if window_function == "sum":
        return sum(clean)
    if window_function in ("avg", "mean"):
        return statistics.fmean(clean)
    if len(clean) < 2:
        return None
    return statistics.stdev(clean) if window_function == "std" else statistics.variance(clean)


def _across_columns(window_function: str, results: list[float | int | None]) -> float | int:
    """Reference reduction of the per-column window results, skipping missing ones."""
    values = [float(r) for r in results if r is not None and not math.isnan(r)]
    if window_function == "count":
        return len(values)
    if not values:
        return _NAN
    if window_function == "sum":
        return sum(values)
    if window_function in ("avg", "mean", "first", "last"):
        return statistics.fmean(values)
    if window_function in ("min", "max", "median"):
        reducers: dict[str, Callable[[list[float]], float]] = {"min": min, "max": max, "median": statistics.median}
        return reducers[window_function](values)
    if len(values) < 2:
        return _NAN
    return statistics.stdev(values) if window_function == "std" else statistics.variance(values)


def _window_reference(window_function: str, columns: dict[str, list[float | None]], span_days: int) -> list[Any]:
    names = list(columns)
    expected: list[Any] = []
    for row in range(len(columns[names[0]])):
        start = max(0, row - span_days + 1)
        per_column = [_column_window_result(window_function, columns[name][start : row + 1]) for name in names]
        expected.append(per_column[0] if len(names) == 1 else _across_columns(window_function, per_column))
    return expected


def _pandas_window(window_function: str, columns: dict[str, list[float | None]], span_days: int) -> list[Any]:
    cells = {k: [np.nan if v is None else v for v in vs] for k, vs in columns.items()}
    times = [_START + _DAY * i for i in range(len(next(iter(columns.values()))))]
    frame = pd.DataFrame({**cells, "t": times})
    result = PandasTimeWindow._perform_window_operation(frame, window_function, _DAY * span_days, list(columns), "t")
    return list(result.tolist())


def _pyarrow_window(window_function: str, columns: dict[str, list[float | None]], span_days: int) -> list[Any]:
    times = [_START + _DAY * i for i in range(len(next(iter(columns.values()))))]
    table = pa.table({**{k: pa.array(vs, type=pa.float64()) for k, vs in columns.items()}, "t": pa.array(times)})
    result = PyArrowTimeWindow._perform_window_operation(table, window_function, _DAY * span_days, list(columns), "t")
    return list(result.to_pylist())


_WINDOW_RESULT_OF: dict[type[Implementation[Any]], Callable[[str, dict[str, list[float | None]], int], list[Any]]] = {
    PandasTimeWindow: _pandas_window,
    PyArrowTimeWindow: _pyarrow_window,
}


class TestImplementationsConformToTheReferenceContract:
    """An implementation differs from the reference exactly where it declares a divergence."""

    @pytest.mark.parametrize("window_function", sorted(WINDOW_UNIVERSE))
    @pytest.mark.parametrize("implementation", list(_WINDOW_RESULT_OF), ids=lambda impl: impl.__name__)
    def test_result_differs_from_the_reference_exactly_when_declared(
        self, implementation: type[Implementation[Any]], window_function: str
    ) -> None:
        assert dict(implementation.divergent) == _DIVERGENT_DECLARATIONS[implementation]
        equal = same_on_pandas if implementation is PandasTimeWindow else same
        differing: list[str] = []
        for shape, (columns, span_days) in _SHAPES.items():
            actual = _WINDOW_RESULT_OF[implementation](window_function, columns, span_days)
            expected = _window_reference(window_function, columns, span_days)
            assert len(actual) == len(expected), f"{shape}: {actual} vs {expected}"
            if not all(equal(a, e) for a, e in zip(actual, expected)):
                differing.append(f"{shape}: {actual} != reference {expected}")
        assert bool(differing) == (window_function in implementation.divergent), differing
