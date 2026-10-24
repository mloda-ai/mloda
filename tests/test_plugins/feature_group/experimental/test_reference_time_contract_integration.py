"""Red-phase integration tests for reference-time contract validation (epic #518, Phase 5).

Behavior that Phase 5 must preserve while unifying the per-framework datetime
checks onto the shared ComparisonContract:

- A NON-temporal reference-time column still raises a clear ValueError that
  names the column. The assertions match on the column NAME only (not the exact
  wording), so they hold both for the current per-framework messages
  ("... must be a datetime/timestamp column ...") and for the future contract
  message ("Column '...' must be ordered/temporal."), avoiding any contradiction
  with the Green refactor.
- A tz-AWARE datetime reference column is ACCEPTED (does not raise). These are
  GUARDS: they pass today (pandas ``is_datetime64_any_dtype`` and pyarrow
  ``is_timestamp`` both accept tz-aware) and must keep passing after the refactor
  (the contract's default TzPolicy.ANY accepts both naive and aware).
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from mloda.user import Feature, Options
from mloda.provider import FeatureSet, DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindow
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindow
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.forecasting.pandas import PandasForecasting
from mloda_plugins.compute_framework.base_implementations.pandas import pandas_type_semantics


class TestPandasTimeWindowReferenceTimeContract:
    """Reference-time validation for the pandas time-window feature group."""

    def test_non_temporal_reference_time_raises_naming_column(self) -> None:
        """A string reference-time column raises ValueError naming the column."""
        df = pd.DataFrame(
            {
                "temperature": [20, 22, 19, 23, 25],
                DefaultOptionKeys.reference_time: ["a", "b", "c", "d", "e"],
            }
        )
        feature_set = FeatureSet()
        feature_set.add(Feature("temperature__avg_3_day_window"))

        with pytest.raises(ValueError, match=str(DefaultOptionKeys.reference_time)):
            PandasTimeWindow.calculate(TimeWindowFeatureGroup, df, feature_set, None)

    def test_tz_aware_reference_time_accepted_guard(self) -> None:
        """GUARD: a tz-aware datetime reference-time column is accepted."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "temperature": [20, 22, 19, 23, 25],
                DefaultOptionKeys.reference_time: dates,
            }
        )
        feature_set = FeatureSet()
        feature_set.add(Feature("temperature__avg_3_day_window"))

        result = PandasTimeWindow.calculate(TimeWindowFeatureGroup, df, feature_set, None)
        assert "temperature__avg_3_day_window" in result.columns


class TestPyArrowTimeWindowReferenceTimeContract:
    """Reference-time validation for the pyarrow time-window feature group."""

    def test_non_temporal_reference_time_raises_naming_column(self) -> None:
        """A string reference-time column raises ValueError naming the column."""
        table = pa.table(
            {
                "temperature": [20, 22, 19, 23, 25],
                DefaultOptionKeys.reference_time: ["a", "b", "c", "d", "e"],
            }
        )
        feature_set = FeatureSet()
        feature_set.add(Feature("temperature__avg_3_day_window"))

        with pytest.raises(ValueError, match=str(DefaultOptionKeys.reference_time)):
            PyArrowTimeWindow.calculate(TimeWindowFeatureGroup, table, feature_set, None)

    def test_tz_aware_reference_time_accepted_guard(self) -> None:
        """GUARD: a tz-aware timestamp reference-time column is accepted."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "temperature": [20, 22, 19, 23, 25],
                DefaultOptionKeys.reference_time: dates,
            }
        )
        table = pa.Table.from_pandas(df)
        feature_set = FeatureSet()
        feature_set.add(Feature("temperature__avg_3_day_window"))

        result = PyArrowTimeWindow.calculate(TimeWindowFeatureGroup, table, feature_set, None)
        assert "temperature__avg_3_day_window" in result.schema.names


class TestPandasForecastingReferenceTimeContract:
    """Reference-time validation for the pandas forecasting feature group."""

    def test_non_temporal_reference_time_raises_naming_column(self) -> None:
        """A string reference-time column raises ValueError naming the column."""
        options = Options({DefaultOptionKeys.reference_time: "time_filter"})
        df = pd.DataFrame(
            {
                "time_filter": ["a", "b", "c", "d", "e"],
                "sales": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        feature_set = FeatureSet()
        feature_set.add(Feature("sales__linear_forecast_7day", options))

        with pytest.raises(ValueError, match="time_filter"):
            PandasForecasting.calculate(ForecastingFeatureGroup, df, feature_set, None)

    def test_tz_aware_reference_time_accepted_guard(self) -> None:
        """GUARD: the datetime check accepts a tz-aware reference-time column."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "time_filter": dates,
                "sales": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        ForecastingFeatureGroup._validate_reference_time_column(
            pandas_type_semantics.column_semantics(df, "time_filter"), "time_filter"
        )
