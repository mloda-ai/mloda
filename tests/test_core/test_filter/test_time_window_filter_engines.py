"""
Integration tests for GlobalFilter time range filtering against tz-aware
timestamp columns across compute frameworks.

These tests verify that ``GlobalFilter.add_time_and_time_travel_filters`` produces
filter bounds that filter engines can compare against tz-aware (UTC) temporal
columns. On the current implementation the bounds are stringified ISO 8601
values which the PyArrow and python_dict filter engines cannot compare to a
real timestamp/datetime column:

* PyArrow raises ``ArrowNotImplementedError`` from
  ``pc.greater_equal(timestamp[us, tz=UTC], string)``.
* python_dict raises ``TypeError: '<=' not supported between instances of 'str'
  and 'datetime.datetime'``.

The tests are designed to fail on current main and pass once the bounds are
returned as UTC-normalized ``datetime`` objects.

Note on pandas: empirically the pandas filter engine silently coerces the
stringified bound back into a tz-aware Timestamp via ``Series.__ge__``, so it
does not raise on current main. A pandas regression guard can be added as a
follow-up but is not part of this Red phase, since it would pass both before
and after the fix.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import pyarrow as pa

from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import GlobalFilter
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


# TODO(#435): Polars is not covered in this integration test. The shared
# ``ATestDataCreator`` only maps Pandas/PyArrow conversions and there is no
# polars time-window plugin to reuse; wiring a polars-native source FG is
# scope creep for the Red phase. PyArrow + python_dict already exercise the
# GlobalFilter -> tz-aware-column code path. A polars regression test can be
# added in a follow-up after the Green-phase fix lands.


# Test data covering rows inside and outside the filter window [Jan 5, Jan 11).
# Indices 4..9 (timestamps Jan 5..Jan 10) are the only rows that must survive.
TIMESTAMPS_ISO: list[str] = [
    "2023-01-01T00:00:00+00:00",
    "2023-01-02T00:00:00+00:00",
    "2023-01-03T00:00:00+00:00",
    "2023-01-04T00:00:00+00:00",
    "2023-01-05T00:00:00+00:00",  # in window
    "2023-01-06T00:00:00+00:00",  # in window
    "2023-01-07T00:00:00+00:00",  # in window
    "2023-01-08T00:00:00+00:00",  # in window
    "2023-01-09T00:00:00+00:00",  # in window
    "2023-01-10T00:00:00+00:00",  # in window
    "2023-01-11T00:00:00+00:00",  # boundary, excluded (max_exclusive=True)
    "2023-01-12T00:00:00+00:00",
]
TEMPERATURES: list[int] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

EVENT_FROM = datetime(2023, 1, 5, tzinfo=timezone.utc)
EVENT_TO = datetime(2023, 1, 11, tzinfo=timezone.utc)

# Rows that must survive the filter (Jan 5..Jan 10, inclusive of start, exclusive of end).
EXPECTED_TEMPERATURES: list[int] = [14, 15, 16, 17, 18, 19]
EXPECTED_ROW_COUNT: int = len(EXPECTED_TEMPERATURES)


class TestGlobalFilterTimeRangePyArrow:
    """GlobalFilter time range filtering on a tz-aware PyArrow timestamp column."""

    def test_pyarrow_tz_aware_time_range_filter(self) -> None:
        """The PyArrow filter engine must accept the GlobalFilter bound against a tz-aware timestamp column."""

        class PyArrowTzAwareSource(FeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator({"temperature", DefaultOptionKeys.reference_time})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                # Build a tz-aware UTC timestamp column natively in PyArrow so the
                # filter engine has to compare timestamp[us, tz=UTC] against the
                # GlobalFilter bound directly.
                timestamps_naive = [datetime.fromisoformat(ts).replace(tzinfo=None) for ts in TIMESTAMPS_ISO]
                ts_array = pa.array(timestamps_naive, type=pa.timestamp("us", tz="UTC"))
                return pa.table(
                    {
                        "temperature": pa.array(TEMPERATURES),
                        DefaultOptionKeys.reference_time: ts_array,
                    }
                )

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PyArrowTable}

        global_filter = GlobalFilter()
        global_filter.add_time_and_time_travel_filters(event_from=EVENT_FROM, event_to=EVENT_TO)

        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowTzAwareSource})

        # On current main this call raises because the PyArrow filter engine cannot
        # compare timestamp[us, tz=UTC] to the stringified bound produced by
        # _check_and_convert_time_info. After the Green-phase fix the call succeeds
        # and we verify the surviving rows.
        result = mloda.run_all(
            ["temperature", DefaultOptionKeys.reference_time],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
            global_filter=global_filter,
        )

        assert len(result) > 0, "No results returned from mloda.run_all"

        temperature_table = None
        for table in result:
            if "temperature" in table.schema.names:
                temperature_table = table
                break

        assert temperature_table is not None, "PyArrow Table with temperature column not found"

        actual_temps = sorted(temperature_table.column("temperature").to_pylist())
        assert actual_temps == EXPECTED_TEMPERATURES, (
            f"Expected only in-window temperatures {EXPECTED_TEMPERATURES}, got {actual_temps}"
        )
        assert temperature_table.num_rows == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} rows after tz-aware filtering, got {temperature_table.num_rows}"
        )


class TestGlobalFilterTimeRangePythonDict:
    """GlobalFilter time range filtering on a python_dict list of tz-aware datetimes."""

    def test_python_dict_tz_aware_time_range_filter(self) -> None:
        """The python_dict filter engine must accept the GlobalFilter bound against tz-aware datetime values."""

        class PythonDictTzAwareSource(FeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator({"temperature", DefaultOptionKeys.reference_time})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                # Emit native tz-aware datetime values inside the row dicts so the
                # filter engine compares datetime <-> bound directly.
                return [
                    {
                        "temperature": TEMPERATURES[i],
                        DefaultOptionKeys.reference_time: datetime.fromisoformat(TIMESTAMPS_ISO[i]),
                    }
                    for i in range(len(TIMESTAMPS_ISO))
                ]

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PythonDictFramework}

        global_filter = GlobalFilter()
        global_filter.add_time_and_time_travel_filters(event_from=EVENT_FROM, event_to=EVENT_TO)

        plugin_collector = PluginCollector.enabled_feature_groups({PythonDictTzAwareSource})

        # On current main this raises ``TypeError: '<=' not supported between
        # instances of 'str' and 'datetime.datetime'`` because the GlobalFilter
        # bound is an ISO 8601 string. After the Green-phase fix the bound is a
        # ``datetime`` and the filter returns the in-window rows.
        result = mloda.run_all(
            ["temperature", DefaultOptionKeys.reference_time],
            compute_frameworks={PythonDictFramework},
            plugin_collector=plugin_collector,
            global_filter=global_filter,
        )

        assert len(result) > 0, "No results returned from mloda.run_all"

        rows = None
        for candidate in result:
            if candidate and isinstance(candidate, list) and "temperature" in candidate[0]:
                rows = candidate
                break

        assert rows is not None, "python_dict result with temperature column not found"

        actual_temps = sorted(row["temperature"] for row in rows)
        assert actual_temps == EXPECTED_TEMPERATURES, (
            f"Expected only in-window temperatures {EXPECTED_TEMPERATURES}, got {actual_temps}"
        )
        assert len(rows) == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} rows after tz-aware filtering, got {len(rows)}"
        )
