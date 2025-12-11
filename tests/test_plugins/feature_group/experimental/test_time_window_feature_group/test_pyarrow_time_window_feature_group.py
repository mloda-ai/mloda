"""
Tests for the PyArrowTimeWindowFeatureGroup class.
"""

from typing import Any
import pyarrow as pa
import pytest
import datetime

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup
from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.conftest import EXPECTED_VALUES


class TestPyArrowTimeWindowFeatureGroup:
    """Tests for the PyArrowTimeWindowFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PyArrowTimeWindowFeatureGroup.compute_framework_rule() == {PyArrowTable}

    def test_get_time_delta(self) -> None:
        """Test _get_time_delta method."""
        assert PyArrowTimeWindowFeatureGroup._get_time_delta(3, "day") == datetime.timedelta(days=3)
        assert PyArrowTimeWindowFeatureGroup._get_time_delta(7, "hour") == datetime.timedelta(hours=7)
        assert PyArrowTimeWindowFeatureGroup._get_time_delta(2, "minute") == datetime.timedelta(minutes=2)
        assert PyArrowTimeWindowFeatureGroup._get_time_delta(4, "second") == datetime.timedelta(seconds=4)
        assert PyArrowTimeWindowFeatureGroup._get_time_delta(1, "week") == datetime.timedelta(weeks=1)
        assert PyArrowTimeWindowFeatureGroup._get_time_delta(6, "month") == datetime.timedelta(days=180)  # 6 * 30
        assert PyArrowTimeWindowFeatureGroup._get_time_delta(2, "year") == datetime.timedelta(days=730)  # 2 * 365

        # Test with invalid time unit
        with pytest.raises(ValueError):
            PyArrowTimeWindowFeatureGroup._get_time_delta(3, "invalid")

    @pytest.mark.parametrize(
        "window_function,expected_values",
        [
            ("avg", EXPECTED_VALUES["avg"]),
            ("sum", EXPECTED_VALUES["sum"]),
            ("min", EXPECTED_VALUES["min"]),
            ("max", EXPECTED_VALUES["max"]),
        ],
    )
    def test_perform_window_operation(
        self, sample_time_table: pa.Table, window_function: str, expected_values: Any
    ) -> None:
        """Test _perform_window_operation method with different window functions."""
        result = PyArrowTimeWindowFeatureGroup._perform_window_operation(
            sample_time_table,
            window_function,
            3,
            "day",
            ["temperature"],
            DefaultOptionKeys.reference_time,
        )

        # Check the first few values
        for i in range(3):
            if isinstance(expected_values[i], float):
                assert abs(result[i].as_py() - expected_values[i]) < 0.1
            else:
                assert result[i].as_py() == expected_values[i]

    def test_perform_window_operation_invalid(self, sample_time_table: pa.Table) -> None:
        """Test _perform_window_operation method with invalid window function."""
        with pytest.raises(ValueError):
            PyArrowTimeWindowFeatureGroup._perform_window_operation(
                sample_time_table,
                "invalid",
                3,
                "day",
                ["temperature"],
                DefaultOptionKeys.reference_time,
            )

    def test_calculate_feature_single(self, sample_time_table: pa.Table, feature_set_avg_window: FeatureSet) -> None:
        """Test calculate_feature method with a single time window feature."""
        result = PyArrowTimeWindowFeatureGroup.calculate_feature(sample_time_table, feature_set_avg_window)

        # Check that the result contains the original data plus the time window feature
        assert "temperature__avg_3_day_window" in result.schema.names

        # Check the values of the time window feature
        # First value should be the temperature itself (20)
        # Second value should be average of first two days (20+22)/2 = 21
        # Third value should be average of first three days (20+22+19)/3 = 20.33
        assert result.column("temperature__avg_3_day_window")[0].as_py() == 20
        assert abs(result.column("temperature__avg_3_day_window")[1].as_py() - 21) < 0.1
        assert abs(result.column("temperature__avg_3_day_window")[2].as_py() - 20.33) < 0.1

        # Check that the original data is preserved
        assert "temperature" in result.schema.names
        assert "humidity" in result.schema.names
        assert "pressure" in result.schema.names
        assert "wind_speed" in result.schema.names

    def test_calculate_feature_multiple(
        self, sample_time_table: pa.Table, feature_set_multiple_windows: FeatureSet
    ) -> None:
        """Test calculate_feature method with multiple time window features."""
        result = PyArrowTimeWindowFeatureGroup.calculate_feature(sample_time_table, feature_set_multiple_windows)

        # Check that the result contains all time window features
        assert "temperature__avg_3_day_window" in result.schema.names
        assert "humidity__max_5_day_window" in result.schema.names
        assert "pressure__min_2_day_window" in result.schema.names
        assert "wind_speed__sum_4_day_window" in result.schema.names

        # Check that the original data is preserved
        assert "temperature" in result.schema.names
        assert "humidity" in result.schema.names
        assert "pressure" in result.schema.names
        assert "wind_speed" in result.schema.names

    def test_calculate_feature_missing_source(self, sample_time_table: pa.Table) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("missing__avg_3_day_window"))

        with pytest.raises(ValueError, match="None of the source features .* found in data"):
            PyArrowTimeWindowFeatureGroup.calculate_feature(sample_time_table, feature_set)

    def test_calculate_feature_missing_time_filter(self) -> None:
        """Test calculate_feature method with Table that has no time filter feature."""
        # Create a Table without the time filter feature
        table = pa.table(
            {
                "temperature": [20, 22, 19, 23, 25],
                "humidity": [65, 70, 75, 60, 55],
            }
        )

        feature_set = FeatureSet()
        feature_set.add(Feature("temperature__avg_3_day_window"))

        with pytest.raises(
            ValueError,
            match=f"Reference time column '{DefaultOptionKeys.reference_time.value}' not found in data.*",
        ):
            PyArrowTimeWindowFeatureGroup.calculate_feature(table, feature_set)

    def test_calculate_feature_invalid_time_column(self, sample_time_table: pa.Table) -> None:
        """Test calculate_feature method with invalid time column type."""
        # Create a Table with a non-timestamp time column
        table = pa.table(
            {
                "temperature": [20, 22, 19, 23, 25],
                DefaultOptionKeys.reference_time: [1, 2, 3, 4, 5],  # Not a timestamp
            }
        )

        feature_set = FeatureSet()
        feature_set.add(Feature("temperature__avg_3_day_window"))

        with pytest.raises(
            ValueError,
            match=f"Reference time column '{DefaultOptionKeys.reference_time.value}' must be a timestamp column.*",
        ):
            PyArrowTimeWindowFeatureGroup.calculate_feature(table, feature_set)
