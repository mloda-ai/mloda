"""
Tests for the ForecastingFeatureGroup.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any
import pytest

from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecastingFeatureGroup
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.forecasting.forecasting_artifact import ForecastingArtifact


class TestForecastingFeatureGroup:
    """Test cases for the ForecastingFeatureGroup."""

    def setup_method(self) -> None:
        """Set up test data."""
        # Create a sample DataFrame with time series data
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)]
        values = [10 + i + np.sin(i * 0.5) * 5 for i in range(30)]

        self.df = pd.DataFrame({"time_filter": dates, "sales": values})

        # Create a feature set
        self.feature_set = FeatureSet()
        self.feature_set.add(Feature("sales__linear_forecast_7day"))

        # Create options
        self.options = Options({DefaultOptionKeys.reference_time: "time_filter"})
        self.feature_set.options = self.options

    def test_feature_name_parsing(self) -> None:
        """Test parsing of feature names."""
        feature_name = "sales__linear_forecast_7day"
        algorithm, horizon, time_unit = ForecastingFeatureGroup.parse_forecast_suffix(feature_name)

        assert algorithm == "linear"
        assert horizon == 7
        assert time_unit == "day"

        chained = "category__onehot_encoded__linear_forecast_7day"
        assert ForecastingFeatureGroup.parse_forecast_suffix(chained) == ("linear", 7, "day")

        assert ForecastingFeatureGroup.parse_forecast_suffix("sales__linear_forecast_7day~0") == ("linear", 7, "day")
        selected_chain = "x__linear_forecast_7day~0__linear_forecast_3day"
        assert ForecastingFeatureGroup.parse_forecast_suffix(selected_chain) == ("linear", 3, "day")

    def test_match_feature_group_criteria(self) -> None:
        """Test matching of feature names to the feature group criteria."""
        # Valid feature names
        assert ForecastingFeatureGroup.match_feature_group_criteria("sales__linear_forecast_7day", Options())

        # This test is failing because the feature name doesn't match the expected pattern
        # Let's modify it to use a valid feature name
        assert ForecastingFeatureGroup.match_feature_group_criteria("sales__randomforest_forecast_3day", Options())
        assert ForecastingFeatureGroup.match_feature_group_criteria("sales__linear_forecast_7day~0", Options())
        chained = "x__linear_forecast_7day~0__linear_forecast_3day"
        assert ForecastingFeatureGroup.match_feature_group_criteria(chained, Options())

        # Invalid feature names
        assert not ForecastingFeatureGroup.match_feature_group_criteria("invalid_feature_name", Options())
        assert not ForecastingFeatureGroup.match_feature_group_criteria("sales__linear_7day", Options())

    def test_input_features(self) -> None:
        """Test extraction of input features."""
        feature_name = "sales__linear_forecast_7day"
        feature_group = PandasForecastingFeatureGroup()

        input_features = feature_group.input_features(self.options, Feature(feature_name).name)

        assert len(input_features) == 2  # type: ignore
        assert any(f.name == "sales" for f in input_features)  # type: ignore
        assert any(f.name == "time_filter" for f in input_features)  # type: ignore

    def test_zero_horizon_is_a_recorded_non_match_naming_the_horizon(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        assert not ForecastingFeatureGroup.match_feature_group_criteria("x__linear_forecast_0day", Options())
        assert "horizon" in rejection_window["ForecastingFeatureGroup"].reason

    def test_unknown_name_algorithm_is_a_recorded_non_match_even_with_a_valid_option(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options(context={ForecastingFeatureGroup.ALGORITHM: "linear"})
        assert not ForecastingFeatureGroup.match_feature_group_criteria("x__bogus_forecast_7day", options)
        assert "bogus" in rejection_window["ForecastingFeatureGroup"].reason

    def test_trailing_suffix_after_time_unit_is_a_recorded_non_match(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """time_unit would bind `day__mean_imputed`, so forecasting does not claim a missing_value name."""
        name = "s__linear_forecast_7day__mean_imputed"
        assert not ForecastingFeatureGroup.match_feature_group_criteria(name, Options())
        assert "time_unit" in rejection_window["ForecastingFeatureGroup"].reason

    def test_chained_source_name_matches_and_extracts_the_chained_source(self) -> None:
        name = "s__mean_imputed__linear_forecast_7day"
        assert ForecastingFeatureGroup.match_feature_group_criteria(name, Options())
        assert ForecastingFeatureGroup._extract_source_features(Feature(name)) == ["s__mean_imputed"]

    def test_selector_name_matches_and_extracts_parameters_and_source(self) -> None:
        name = "sales__linear_forecast_7day~10"
        assert ForecastingFeatureGroup.match_feature_group_criteria(name, Options())
        assert ForecastingFeatureGroup._extract_forecast_params(Feature(name)) == ("linear", 7, "day")
        assert ForecastingFeatureGroup._extract_source_features(Feature(name)) == ["sales"]

    def test_valid_name_records_no_rejection(self, rejection_window: dict[str, MatchRejection]) -> None:
        assert ForecastingFeatureGroup.match_feature_group_criteria("x__linear_forecast_7day", Options())
        assert rejection_window == {}

    def test_name_path_source_count_above_max_is_a_recorded_non_match(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The name path validates the source count via the mixin (MAX_IN_FEATURES = 1)."""
        name = "a&b__linear_forecast_7day"
        assert not ForecastingFeatureGroup.match_feature_group_criteria(name, Options())
        reason = rejection_window["ForecastingFeatureGroup"].reason
        assert "at most 1" in reason
        assert "found 2" in reason

        with pytest.raises(ValueError) as exc_info:
            PandasForecastingFeatureGroup().input_features(self.options, FeatureName(name))
        assert str(exc_info.value) == reason

    def test_pandas_forecasting(self) -> None:
        """Test forecasting with the Pandas implementation."""
        # Perform forecasting
        result, artifact = PandasForecastingFeatureGroup._perform_forecasting(
            self.df, "linear", 7, "day", "sales", "time_filter", None
        )

        # Check that the result is a pandas Series
        assert isinstance(result, pd.Series)

        # Check that the artifact contains the expected keys
        assert "model" in artifact
        assert "scaler" in artifact
        assert "last_trained_timestamp" in artifact
        assert "feature_names" in artifact

        # Check that the result contains forecasts for the future
        assert len(result) == 37  # 30 original points + 7 forecast points

        # Test with a pre-trained model
        result2, artifact2 = PandasForecastingFeatureGroup._perform_forecasting(
            self.df, "linear", 7, "day", "sales", "time_filter", artifact
        )

        # Check that the result is a pandas Series
        assert isinstance(result2, pd.Series)

        # Check that the artifact contains the expected keys
        assert "model" in artifact2
        assert "scaler" in artifact2
        assert "last_trained_timestamp" in artifact2
        assert "feature_names" in artifact2

    def test_different_algorithms(self) -> None:
        """Test different forecasting algorithms."""
        algorithms = ["linear", "ridge", "lasso", "randomforest", "gbr", "svr", "knn"]

        for algorithm in algorithms:
            # Perform forecasting
            result, artifact = PandasForecastingFeatureGroup._perform_forecasting(
                self.df,
                algorithm,
                3,  # Use a smaller horizon for faster tests
                "day",
                "sales",
                "time_filter",
                None,
            )

            # Check that the result is a pandas Series
            assert isinstance(result, pd.Series)

            # Check that the artifact contains the expected keys
            assert "model" in artifact
            assert "scaler" in artifact
            assert "last_trained_timestamp" in artifact
            assert "feature_names" in artifact

            # Check that the result contains forecasts for the future
            assert len(result) == 33  # 30 original points + 3 forecast points

    def test_calculate_feature(self) -> None:
        """Test the calculate_feature method."""
        feature_name = "sales__linear_forecast_7day"

        # Create a feature set with artifact saving enabled
        feature_set = FeatureSet()
        feature_set.add(Feature(feature_name, self.options))
        feature_set.artifact_to_save = feature_name

        # Calculate the feature
        result_df = PandasForecastingFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        # Check that the result contains the forecast feature
        assert feature_name in result_df.columns

        # Check that an artifact was saved
        assert feature_set.save_artifact is not None

        # Get the saved artifact
        saved_artifact = feature_set.save_artifact

        # Create a new feature set with artifact loading enabled
        feature_set2 = FeatureSet()
        feature_set2.add(Feature(feature_name))

        # Create new options with the saved artifact
        options2 = Options(group=self.options.group.copy(), context=self.options.context.copy())

        # We need to serialize the artifact before setting it in the options
        serialized_artifact = ForecastingArtifact._serialize_artifact(saved_artifact)

        # Set the serialized artifact in the options using the feature name as the key
        options2.add_to_group(feature_name, serialized_artifact)

        for feature in feature_set2.features:
            feature.options = options2

        # Set the artifact to load
        feature_set2.artifact_to_load = feature_name

        # Calculate the feature using the saved artifact
        result_df2 = PandasForecastingFeatureGroup.calculate_feature(self.df.copy(), feature_set2)

        # Check that the result contains the forecast feature
        assert feature_name in result_df2.columns

    def test_calculate_feature_populates_forecast_values(self) -> None:
        """Forecast values must survive: N+horizon rows, no NaN, historical actuals preserved."""
        feature_name = "sales__linear_forecast_7day"

        feature_set = FeatureSet()
        feature_set.add(Feature(feature_name, self.options))
        feature_set.artifact_to_save = feature_name

        result_df = PandasForecastingFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        assert feature_name in result_df.columns
        assert len(result_df) == 37
        assert result_df[feature_name].notna().all()

        forecast_values = result_df[feature_name].to_numpy()
        assert np.allclose(forecast_values[:30], self.df["sales"].to_numpy())
        assert np.isfinite(forecast_values[30:]).all()

    def test_calculate_feature_unsorted_input_preserves_row_alignment(self) -> None:
        """Unsorted input must keep each row's historical forecast value aligned to its own source value."""
        df_unsorted = self.df.iloc[::-1].reset_index(drop=True)  # reverse-chronological, default RangeIndex
        feature_name = "sales__linear_forecast_7day"

        feature_set = FeatureSet()
        feature_set.add(Feature(feature_name, self.options))
        feature_set.artifact_to_save = feature_name

        result_df = PandasForecastingFeatureGroup.calculate_feature(df_unsorted.copy(), feature_set)

        assert len(result_df) == 37
        assert feature_name in result_df.columns

        forecast_values = result_df[feature_name].to_numpy()
        assert np.allclose(forecast_values[:30], df_unsorted["sales"].to_numpy())
        assert np.isfinite(forecast_values[30:]).all()

    def test_calculate_feature_multiple_horizons_not_truncated(self) -> None:
        """Requesting two horizons together must not truncate the longer forecast; the shorter is NaN-padded."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sales__linear_forecast_7day", self.options))
        feature_set.add(Feature("sales__linear_forecast_14day", self.options))

        result_df = PandasForecastingFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        assert len(result_df) == 44
        assert "sales__linear_forecast_7day" in result_df.columns
        assert "sales__linear_forecast_14day" in result_df.columns

        assert result_df["sales__linear_forecast_14day"].notna().all()
        assert len(result_df["sales__linear_forecast_14day"]) == 44

        assert result_df["sales__linear_forecast_7day"].iloc[:37].notna().all()
        assert result_df["sales__linear_forecast_7day"].iloc[37:].isna().all()

        assert np.allclose(result_df["sales__linear_forecast_14day"].to_numpy()[:30], self.df["sales"].to_numpy())
        assert np.allclose(result_df["sales__linear_forecast_7day"].to_numpy()[:30], self.df["sales"].to_numpy())

    def test_get_reference_time_column_default(self) -> None:
        """Test get_reference_time_column method returns default column name when no options provided."""
        # Test with no options - should return default column name
        assert ForecastingFeatureGroup.get_reference_time_column() == DefaultOptionKeys.reference_time

    def test_get_reference_time_column_custom(self) -> None:
        """Test get_reference_time_column method returns custom column name when reference_time option is set."""
        # Test with custom options using DefaultOptionKeys.reference_time
        options = Options()
        options.add_to_group(DefaultOptionKeys.reference_time, "custom_time_column")
        assert ForecastingFeatureGroup.get_reference_time_column(options) == "custom_time_column"

        # Test with custom options using DefaultOptionKeys.reference_time.value
        options = Options()
        options.add_to_group(DefaultOptionKeys.reference_time, "another_custom_column")
        assert ForecastingFeatureGroup.get_reference_time_column(options) == "another_custom_column"

    def test_get_reference_time_column_invalid_type(self) -> None:
        """Test get_reference_time_column method raises ValueError when option value is not a string."""
        # Test with invalid options (non-string value)
        options = Options()
        options.add_to_group(DefaultOptionKeys.reference_time, 123)  # Not a string
        with pytest.raises(ValueError):
            ForecastingFeatureGroup.get_reference_time_column(options)

    def test_has_valid_forecast_suffix_valid(self) -> None:
        """Test that _has_valid_forecast_suffix returns True for valid feature names."""
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("sales__linear_forecast_7day") is True
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("price__ridge_forecast_30week") is True
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("demand__randomforest_forecast_3month") is True
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("temp__gbr_forecast_12hour") is True
        chained = "category__onehot_encoded__linear_forecast_7day"
        assert ForecastingFeatureGroup._has_valid_forecast_suffix(chained) is True
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("sales__linear_forecast_7day~0") is True
        selected_chain = "x__linear_forecast_7day~0__linear_forecast_3day"
        assert ForecastingFeatureGroup._has_valid_forecast_suffix(selected_chain) is True

    def test_has_valid_forecast_suffix_invalid(self) -> None:
        """Test that _has_valid_forecast_suffix returns False for invalid feature names."""
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("invalid_feature_name") is False
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("sales__linear_7day") is False
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("sales__unknown_forecast_7day") is False
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("sales__linear_forecast_day") is False
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("sales__linear_forecast_7invalid") is False
        assert ForecastingFeatureGroup._has_valid_forecast_suffix("a__b__linear_forecast_7day__foo") is False

    @pytest.mark.parametrize(
        "name",
        [
            "x__linear_forecast_0day",
            "a__lin_ear_forecast_7day",
            "x__mean_imputed__linear_forecast_0day",
            "x__linear_forecast_7day_extra",
            "x__mean_imputed__linear_forecast_7day_extra",
        ],
    )
    def test_names_the_hand_parser_rejected_are_still_rejected(self, name: str) -> None:
        with pytest.raises(ValueError):
            ForecastingFeatureGroup.parse_forecast_suffix(name)
        assert ForecastingFeatureGroup._has_valid_forecast_suffix(name) is False

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("x__linear_forecast_7day~0", "0"),
            ("x__linear_forecast_7day~10", "10"),
            ("x__mean_imputed__linear_forecast_7day~2", "2"),
            ("x__linear_forecast_7day", None),
            ("x__linear_forecast_7day~0__linear_forecast_3day", None),
            ("x__bogus_forecast_7day~0", None),
            ("x__linear_forecast_0day~0", None),
        ],
    )
    def test_extract_selector(self, name: str, expected: str | None) -> None:
        assert ForecastingFeatureGroup._extract_selector(name) == expected

    def test_empty_source_raises(self) -> None:
        name = "__linear_forecast_7day"
        with pytest.raises(ValueError):
            ForecastingFeatureGroup.parse_forecast_suffix(name)
        with pytest.raises(ValueError):
            ForecastingFeatureGroup._has_valid_forecast_suffix(name)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("x__linear_fcst_7day", ("linear", 7, "day")), ("x__mean_imputed__ridge_fcst_3hour", ("ridge", 3, "hour"))],
    )
    def test_parse_forecast_suffix_follows_an_overridden_prefix_pattern(
        self, name: str, expected: tuple[str, int, str]
    ) -> None:
        """The parts come from PREFIX_PATTERN, not from a hand-written copy of the grammar."""

        class FcstPatternGroup(ForecastingFeatureGroup):
            PREFIX_PATTERN = r".*__(?P<algorithm>[\w]+)_fcst_(?P<horizon>\d+)(?P<time_unit>[\w]+)$"

        assert FcstPatternGroup.parse_forecast_suffix(name) == expected
        assert FcstPatternGroup._has_valid_forecast_suffix(name) is True
        with pytest.raises(ValueError):
            FcstPatternGroup.parse_forecast_suffix("x__linear_forecast_7day")

    def test_extract_forecast_params_string_based(self) -> None:
        """Test that _extract_forecast_params extracts parameters from a string-based feature name."""
        feature = Feature("sales__linear_forecast_7day")
        algorithm, horizon, time_unit = ForecastingFeatureGroup._extract_forecast_params(feature)
        assert algorithm == "linear"
        assert horizon == 7
        assert time_unit == "day"

    def test_extract_forecast_params_config_fallback(self) -> None:
        """Test that _extract_forecast_params falls back to configuration-based options."""
        options = Options()
        options.add_to_group(ForecastingFeatureGroup.ALGORITHM, "ridge")
        options.add_to_group(ForecastingFeatureGroup.HORIZON, "14")
        options.add_to_group(ForecastingFeatureGroup.TIME_UNIT, "day")
        feature = Feature("some_feature", options)
        algorithm, horizon, time_unit = ForecastingFeatureGroup._extract_forecast_params(feature)
        assert algorithm == "ridge"
        assert horizon == 14
        assert time_unit == "day"

    FEATURE_NAME = "sales__linear_forecast_7day"

    def _expanded_df(self, suffixes: list[str]) -> pd.DataFrame:
        """Build a frame whose source column is expanded into `sales~<suffix>` columns."""
        df = self.df[["time_filter"]].copy()
        for i, suffix in enumerate(suffixes):
            df[f"sales~{suffix}"] = self.df["sales"] * (i + 1) + i * 3
        return df

    def _calculate(self, df: pd.DataFrame, feature_set: FeatureSet | None = None) -> pd.DataFrame:
        if feature_set is None:
            feature_set = FeatureSet()
            feature_set.add(Feature(self.FEATURE_NAME, self.options))
        result: pd.DataFrame = PandasForecastingFeatureGroup.calculate_feature(df.copy(), feature_set)
        return result

    def _options(self, confidence_intervals: bool = False) -> Options:
        options = Options(group=self.options.group.copy(), context=self.options.context.copy())
        if confidence_intervals:
            options.add_to_group(ForecastingFeatureGroup.OUTPUT_CONFIDENCE_INTERVALS, True)
        return options

    def _save_artifact(
        self, df: pd.DataFrame, confidence_intervals: bool = False, name: str | None = None
    ) -> tuple[pd.DataFrame, str]:
        name = name or self.FEATURE_NAME
        feature_set = FeatureSet()
        feature_set.add(Feature(name, self._options(confidence_intervals)))
        feature_set.artifact_to_save = name
        result = self._calculate(df, feature_set)

        assert feature_set.save_artifact is not None
        return result, ForecastingArtifact._serialize_artifact(feature_set.save_artifact)

    def _load_feature_set(
        self, serialized: str, confidence_intervals: bool = False, name: str | None = None
    ) -> FeatureSet:
        name = name or self.FEATURE_NAME
        options = self._options(confidence_intervals)
        options.add_to_group(name, serialized)
        feature_set = FeatureSet()
        feature_set.add(Feature(name, options))
        feature_set.artifact_to_load = name
        return feature_set

    def test_expanded_source_forecasts_each_column(self) -> None:
        """Each `sales~<s>` column gets its own `name~<s>` forecast equal to a plain single-source forecast."""
        df = self._expanded_df(["2", "10"])
        result = self._calculate(df)

        assert self.FEATURE_NAME not in result.columns
        for suffix in ("2", "10"):
            plain = self._calculate(pd.DataFrame({"time_filter": df["time_filter"], "sales": df[f"sales~{suffix}"]}))
            assert np.allclose(result[f"{self.FEATURE_NAME}~{suffix}"].to_numpy(), plain[self.FEATURE_NAME].to_numpy())

    def test_single_expanded_column_is_suffixed(self) -> None:
        """A lone `sales~0` column still yields `name~0`, not `name`."""
        result = self._calculate(self._expanded_df(["0"]))

        assert f"{self.FEATURE_NAME}~0" in result.columns
        assert self.FEATURE_NAME not in result.columns

    def test_expanded_source_confidence_intervals(self) -> None:
        """Confidence bounds on an expanded source are named `name~<s>~lower/~upper`."""
        options = Options(group=self.options.group.copy(), context=self.options.context.copy())
        options.add_to_group(ForecastingFeatureGroup.OUTPUT_CONFIDENCE_INTERVALS, True)
        feature_set = FeatureSet()
        feature_set.add(Feature(self.FEATURE_NAME, options))

        result = self._calculate(self._expanded_df(["2", "10"]), feature_set)

        for suffix in ("2", "10"):
            base = f"{self.FEATURE_NAME}~{suffix}"
            assert base in result.columns
            assert f"{base}~lower" in result.columns
            assert f"{base}~upper" in result.columns

    @pytest.mark.parametrize("output_confidence_intervals", [False, True])
    def test_expanded_source_artifact_round_trip(self, output_confidence_intervals: bool) -> None:
        """A nested multi-column artifact serializes, loads and reproduces the forecasts."""
        df = self._expanded_df(["2", "10"])
        trained, serialized = self._save_artifact(df, output_confidence_intervals)

        loaded = self._calculate(df, self._load_feature_set(serialized, output_confidence_intervals))

        for suffix in ("2", "10"):
            column = f"{self.FEATURE_NAME}~{suffix}"
            columns = [column]
            if output_confidence_intervals:
                columns += [f"{column}~lower", f"{column}~upper"]
            for name in columns:
                assert np.allclose(trained[name].to_numpy(), loaded[name].to_numpy())

    def test_flat_artifact_with_expanded_source_raises(self) -> None:
        """Loading a flat artifact against an expanded source raises ValueError."""
        _, serialized = self._save_artifact(self.df)

        with pytest.raises(ValueError, match="artifact"):
            self._calculate(self._expanded_df(["2", "10"]), self._load_feature_set(serialized))

    def test_nested_artifact_with_literal_source_raises(self) -> None:
        """Loading a nested artifact against a literal source raises ValueError."""
        _, serialized = self._save_artifact(self._expanded_df(["2", "10"]))

        with pytest.raises(ValueError, match="artifact"):
            self._calculate(self.df, self._load_feature_set(serialized))

    def test_nested_artifact_with_different_suffix_set_raises(self) -> None:
        """Loading a nested artifact whose suffix set differs from the resolved columns raises ValueError."""
        _, serialized = self._save_artifact(self._expanded_df(["2", "10"]))

        with pytest.raises(ValueError, match="artifact"):
            self._calculate(self._expanded_df(["2", "3"]), self._load_feature_set(serialized))

    SELECTED_NAME = "sales__linear_forecast_7day~10"

    def _selected_feature_set(self, confidence_intervals: bool = False) -> FeatureSet:
        feature_set = FeatureSet()
        feature_set.add(Feature(self.SELECTED_NAME, self._options(confidence_intervals)))
        return feature_set

    def test_selector_forecasts_only_selected_column(self) -> None:
        """`name~10` forecasts only `sales~10`, matching that column of the full run."""
        df = self._expanded_df(["2", "10"])
        full = self._calculate(df)
        result = self._calculate(df, self._selected_feature_set())

        assert self.SELECTED_NAME in result.columns
        assert f"{self.FEATURE_NAME}~2" not in result.columns
        assert self.FEATURE_NAME not in result.columns
        assert np.allclose(result[self.SELECTED_NAME].to_numpy(), full[f"{self.FEATURE_NAME}~10"].to_numpy())

    def test_selector_with_confidence_intervals(self) -> None:
        """A selected column gets `~lower/~upper` bounds and no other column is forecast."""
        result = self._calculate(self._expanded_df(["2", "10"]), self._selected_feature_set(True))

        for name in (self.SELECTED_NAME, f"{self.SELECTED_NAME}~lower", f"{self.SELECTED_NAME}~upper"):
            assert name in result.columns
        assert not [c for c in result.columns if c.startswith(f"{self.FEATURE_NAME}~2")]

    def test_selector_artifact_round_trip_is_flat(self) -> None:
        """A selected forecast saves a flat single-model artifact and reproduces the forecast."""
        df = self._expanded_df(["2", "10"])
        trained, serialized = self._save_artifact(df, name=self.SELECTED_NAME)

        assert '"columns"' not in serialized
        loaded = self._calculate(df, self._load_feature_set(serialized, name=self.SELECTED_NAME))

        assert np.allclose(trained[self.SELECTED_NAME].to_numpy(), loaded[self.SELECTED_NAME].to_numpy())

    def test_selector_uses_nested_artifact_column(self) -> None:
        """A nested artifact saved from `name` serves `name~10` with the `~10` model."""
        df = self._expanded_df(["2", "10"])
        trained, serialized = self._save_artifact(df)

        loaded = self._calculate(df, self._load_feature_set(serialized, name=self.SELECTED_NAME))

        expected = trained[f"{self.FEATURE_NAME}~10"].to_numpy()
        assert np.allclose(loaded[self.SELECTED_NAME].to_numpy(), expected)

    def test_selector_missing_from_nested_artifact_raises(self) -> None:
        """A selector absent from a nested artifact raises ValueError mentioning the artifact."""
        df = self._expanded_df(["2", "10"])
        _, serialized = self._save_artifact(df)

        with pytest.raises(ValueError, match="artifact"):
            self._calculate(
                self._expanded_df(["2", "3"]),
                self._load_feature_set(serialized, name="sales__linear_forecast_7day~3"),
            )

    @pytest.mark.parametrize(
        "name",
        [
            "sales__linear_forecast_7day~lower",
            "sales__linear_forecast_7day~upper",
            "sales__linear_forecast_7day~0_x",
            "sales__linear_forecast_7day~a\n",
        ],
    )
    def test_invalid_selector_tails_do_not_match(self, name: str) -> None:
        """Bound names, selectors containing `_`, and a trailing newline are not forecasting features."""
        assert not ForecastingFeatureGroup.match_feature_group_criteria(name, Options())
        assert ForecastingFeatureGroup._has_valid_forecast_suffix(name) is False

    def test_unknown_selector_raises(self) -> None:
        """A selector matching no resolved column raises ValueError naming the selector."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sales__linear_forecast_7day~3", self._options()))

        with pytest.raises(ValueError, match="selector") as excinfo:
            self._calculate(self._expanded_df(["2", "10"]), feature_set)
        assert "~3" in str(excinfo.value)

    def test_selector_on_literal_source_raises(self) -> None:
        """A selector on a literal single-column source raises ValueError naming the selector."""
        feature_set = FeatureSet()
        feature_set.add(Feature("sales__linear_forecast_7day~0", self._options()))

        with pytest.raises(ValueError, match=r"single column.*selector|selector.*single column") as excinfo:
            self._calculate(self.df, feature_set)
        assert "~0" in str(excinfo.value)

    def test_chaining_skips_bound_columns(self) -> None:
        """Chaining on a multi-column forecast forecasts `~0`, `~1` but not the `~lower/~upper` bounds."""
        df = self._expanded_df(["0", "1"])
        df["sales~0~lower"] = df["sales~0"] - 1.0
        df["sales~0~upper"] = df["sales~0"] + 1.0

        result = self._calculate(df)

        forecast_columns = [c for c in result.columns if c.startswith(self.FEATURE_NAME)]
        assert sorted(forecast_columns) == [f"{self.FEATURE_NAME}~0", f"{self.FEATURE_NAME}~1"]

    def test_chaining_keeps_deeper_sub_columns(self) -> None:
        """A deeper non-bound sub-column such as `sales~a~b` is still forecast."""
        df = self._expanded_df(["a~b"])

        result = self._calculate(df)

        assert f"{self.FEATURE_NAME}~a~b" in result.columns

    @pytest.mark.parametrize("extra", [np.arange(30, dtype=float) * 3.0, ["x"] * 30])
    def test_unrelated_columns_are_not_regressors(self, extra: Any) -> None:
        """Unrelated numeric or string columns in the frame do not affect single-column forecasting."""
        df = self.df.copy()
        df["other"] = extra

        result = self._calculate(df)
        expected = self._calculate(self.df)

        assert np.allclose(result[self.FEATURE_NAME].to_numpy(), expected[self.FEATURE_NAME].to_numpy())
