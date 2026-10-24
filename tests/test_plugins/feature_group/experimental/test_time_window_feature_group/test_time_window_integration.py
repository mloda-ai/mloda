"""
Integration tests for time window feature groups.
"""

from typing import Any

import pandas as pd
import pytest

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.provider import ComputeFramework, DefaultOptionKeys, FeatureSet
from mloda.user import mloda
from mloda.user import Feature
from mloda.user import Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import (
    PandasAggregation as PandasAggregation,
)
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindow as PandasTimeWindow
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindow as PyArrowTimeWindow

from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import column_values
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator
from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.test_time_window_utils import (
    PandasTimeWindowTestDataCreator,
    PyArrowTimeWindowTestDataCreator,
    validate_time_window_features,
)


# List of time window features to test
TIME_WINDOW_FEATURES: list[Feature | str] = [
    "temperature__avg_3_day_window",  # 3-day average temperature
    "humidity__max_5_day_window",  # 5-day maximum humidity
    "pressure__min_2_day_window",  # 2-day minimum pressure
    "wind_speed__sum_4_day_window",  # 4-day sum of wind speed
]


class TestTimeWindowPandasIntegration:
    """Integration tests for the time window feature group using Pandas."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mloda using GeneratedFeatureGroup."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasTimeWindowTestDataCreator, TimeWindowFeatureGroup}
        )

        # Run the mloda with multiple time window features
        result = mloda.run_all(
            [
                "temperature",  # Source data
                "temperature__avg_3_day_window",  # 3-day average temperature
                "humidity__max_5_day_window",  # 5-day maximum humidity
                "pressure__min_2_day_window",  # 2-day minimum pressure
                "wind_speed__sum_4_day_window",  # 4-day sum of wind speed
            ],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for time window features

        # Find the DataFrame with the time window features
        window_df = None
        for df in result:
            if "temperature__avg_3_day_window" in df.columns:
                window_df = df
                break

        assert window_df is not None, "DataFrame with time window features not found"

        # Validate the time window features
        validate_time_window_features(window_df, TIME_WINDOW_FEATURES)

    def test_chained_aggregated_source_shares_one_root_step_with_reference_time(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasTimeWindowTestDataCreator, TimeWindowFeatureGroup, AggregatedFeatureGroup}
        )
        source = Feature(
            "temp_max", Options(group={"aggregation_type": "max", DefaultOptionKeys.in_features: "temperature"})
        )
        f = Feature(
            "tw",
            Options(
                context={
                    "window_function": "sum",
                    "window_size": 2,
                    "time_unit": "day",
                    DefaultOptionKeys.in_features: frozenset([source]),
                }
            ),
        )
        session = mloda.prepare([f], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)
        assert session.engine is not None
        root_steps = [
            s
            for s in session.engine.execution_planner
            if isinstance(s, FeatureGroupStep) and s.feature_group is PandasTimeWindowTestDataCreator
        ]
        assert len(root_steps) == 1
        assert {str(n) for n in root_steps[0].features.get_all_names()} == {
            "temperature",
            DefaultOptionKeys.reference_time.value,
        }

        results = session.run()
        assert len(results) == 1
        # max(temperature)=25 for every row; 2-day sum is 25 on the first row (min_periods=1), then 50.
        assert results[0]["tw"].tolist() == [25] + [50] * 9


class TestTimeWindowPyArrowIntegration:
    """Integration tests for the time window feature group using PyArrow."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mloda using GeneratedFeatureGroup."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowTimeWindowTestDataCreator, TimeWindowFeatureGroup}
        )

        # Run the mloda with multiple time window features
        result = mloda.run_all(
            [
                "temperature",  # Source data
                "temperature__avg_3_day_window",  # 3-day average temperature
                "humidity__max_5_day_window",  # 5-day maximum humidity
                "pressure__min_2_day_window",  # 2-day minimum pressure
                "wind_speed__sum_4_day_window",  # 4-day sum of wind speed
            ],
            compute_frameworks=[PyArrowTable],
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two Tables: one for source data, one for time window features

        # Find the Table with the time window features
        window_table = None
        for table in result:
            if "temperature__avg_3_day_window" in table.schema.names:
                window_table = table
                break

        assert window_table is not None, "Table with time window features not found"

        # Convert PyArrow Table to Pandas DataFrame for validation
        window_df = window_table.to_pandas()

        # Validate the time window features
        validate_time_window_features(window_df, TIME_WINDOW_FEATURES)


class TimeWindowNanSourceCreator(ATestDataCreator):
    """Serves a float column holding a NaN on both Pandas and PyArrow, so the window framework is free."""

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {
            "tw_nan_x": [1.0, float("nan"), 3.0, 4.0, 5.0],
            DefaultOptionKeys.reference_time: pd.date_range(start="2023-01-01", periods=5, freq="D"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame, PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return cls.get_raw_data()


class TimeWindowNanPandasOnlySourceCreator(TimeWindowNanSourceCreator):
    """Serves the same NaN column on Pandas only, so PyArrow needs a transfer."""

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        return {
            "tw_nan_p": [1.0, float("nan"), 3.0, 4.0, 5.0],
            DefaultOptionKeys.reference_time: pd.date_range(start="2023-01-01", periods=5, freq="D"),
        }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


_NAN_WINDOW = "tw_nan_x__sum_3_day_window"
_PANDAS_ONLY_WINDOW = "tw_nan_p__sum_3_day_window"
_WINDOW_PLUGINS_OF = {
    _NAN_WINDOW: PluginCollector.enabled_feature_groups({TimeWindowNanSourceCreator, TimeWindowFeatureGroup}),
    _PANDAS_ONLY_WINDOW: PluginCollector.enabled_feature_groups(
        {TimeWindowNanPandasOnlySourceCreator, TimeWindowFeatureGroup}
    ),
}
_PANDAS_DIVERGENT = "pinned; runs a version that differs: gives 0 for a multi-column row with no value"


class TestTimeWindowImplementationChoice:
    """With Pandas and PyArrow both enabled the matching implementation wins unless one is pinned."""

    @pytest.mark.parametrize(
        ("name", "pin", "framework", "reason"),
        [
            pytest.param(
                _NAN_WINDOW, {}, "PyArrowTable", "prefers a matching version", id="matching_implementation_wins"
            ),
            pytest.param(
                _NAN_WINDOW,
                {"implementation": "PandasTimeWindow"},
                "PandasDataFrame",
                _PANDAS_DIVERGENT,
                id="implementation_pin",
            ),
            pytest.param(
                _PANDAS_ONLY_WINDOW, {}, "PyArrowTable", "prefers a matching version", id="pandas_only_source_unpinned"
            ),
        ],
    )
    def test_sum_over_a_nan_runs_the_chosen_implementation(
        self, name: str, pin: dict[str, Any], framework: str, reason: str
    ) -> None:
        feature = Feature(name, **pin)
        frameworks: list[type[ComputeFramework]] = [PandasDataFrame, PyArrowTable]
        plugins = _WINDOW_PLUGINS_OF[name]

        steps = mloda.explain([feature], compute_frameworks=frameworks, plugin_collector=plugins)
        step = next(s for s in steps if s.step_kind == "compute" and s.feature_group is TimeWindowFeatureGroup)
        assert step.compute_framework_name == framework
        assert step.compute_framework_reason is not None
        if reason == _PANDAS_DIVERGENT:
            assert step.compute_framework_reason.startswith(reason)
        else:
            assert step.compute_framework_reason == reason

        results = mloda.run_all([feature], compute_frameworks=frameworks, plugin_collector=plugins)
        # NaN is missing and skipped: 3-day sums of [1, NaN, 3, 4, 5].
        assert column_values(results, name) == [1.0, 1.0, 4.0, 7.0, 12.0]
