"""
Regression guard for #435: GlobalFilter time-range bounds must be comparable against
tz-aware temporal columns at each compute framework's filter engine.

Before the fix, ``GlobalFilter.add_time_and_time_travel_filters`` stringified bounds
via ``datetime.isoformat()``. The PyArrow filter engine then raised
``ArrowNotImplementedError: Function 'greater_equal' has no kernel matching input
types (timestamp[us, tz=UTC], string)``, and the python_dict engine raised
``TypeError: '<=' not supported between instances of 'str' and 'datetime.datetime'``.

These tests pin the contract that bounds reach engines as tz-aware ``datetime``
objects and survive the round trip end-to-end through ``mloda.run_all`` for the
three engines that previously broke under tz-aware columns: PyArrow, python_dict,
and Polars.

Note on pandas: the pandas filter engine silently coerced the stringified bound
back into a tz-aware Timestamp via ``Series.__ge__`` even before the fix, so a
dedicated pandas regression test would pass both before and after #435 and is
omitted as redundant.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import pyarrow as pa
import pytest

from mloda.provider import (
    BaseInputData,
    ComputeFramework,
    DataCreator,
    DefaultOptionKeys,
    FeatureGroup,
    FeatureSet,
)
from mloda.user import GlobalFilter, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


# Timestamps Jan 1..Jan 12 at midnight UTC. Indices 4..9 (Jan 5..Jan 10) are the
# only rows that must survive the [Jan 5, Jan 11) window with max_exclusive=True.
TIMESTAMPS_UTC: list[datetime] = [datetime(2023, 1, day, tzinfo=timezone.utc) for day in range(1, 13)]
TEMPERATURES: list[int] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

EVENT_FROM = datetime(2023, 1, 5, tzinfo=timezone.utc)
EVENT_TO = datetime(2023, 1, 11, tzinfo=timezone.utc)

EXPECTED_TEMPERATURES: list[int] = [14, 15, 16, 17, 18, 19]
EXPECTED_ROW_COUNT: int = len(EXPECTED_TEMPERATURES)


def test_pyarrow_tz_aware_time_range_filter() -> None:
    """Regression guard for #435 on the PyArrow filter engine."""

    class PyArrowTzAwareSource(FeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({"temperature", DefaultOptionKeys.reference_time})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            # pa.array with a tz-typed timestamp interprets naive datetimes as already in that tz;
            # tz-aware datetimes are rejected on some PyArrow versions.
            naive_ts = [ts.replace(tzinfo=None) for ts in TIMESTAMPS_UTC]
            ts_array = pa.array(naive_ts, type=pa.timestamp("us", tz="UTC"))
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

    result = mloda.run_all(
        ["temperature", DefaultOptionKeys.reference_time],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        global_filter=global_filter,
    )

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
    assert temperature_table.num_rows == EXPECTED_ROW_COUNT


def test_pyarrow_tz_aware_time_range_filter_with_validity_window() -> None:
    """Regression guard for #435 on the time-travel (validity_from / validity_to) path.

    The fix sits in the shared ``_add_range_filter`` helper, so both the event-time
    and validity-time bounds go through the same normalization. This case pins the
    second invocation explicitly.
    """

    class PyArrowTzAwareValiditySource(FeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({"temperature", DefaultOptionKeys.reference_time, "valid_time"})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            naive_ts = [ts.replace(tzinfo=None) for ts in TIMESTAMPS_UTC]
            ts_array = pa.array(naive_ts, type=pa.timestamp("us", tz="UTC"))
            return pa.table(
                {
                    "temperature": pa.array(TEMPERATURES),
                    DefaultOptionKeys.reference_time: ts_array,
                    "valid_time": ts_array,
                }
            )

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {PyArrowTable}

    global_filter = GlobalFilter()
    global_filter.add_time_and_time_travel_filters(
        event_from=EVENT_FROM,
        event_to=EVENT_TO,
        valid_from=EVENT_FROM,
        valid_to=EVENT_TO,
        validity_time_column="valid_time",
    )

    plugin_collector = PluginCollector.enabled_feature_groups({PyArrowTzAwareValiditySource})

    result = mloda.run_all(
        ["temperature", DefaultOptionKeys.reference_time, "valid_time"],
        compute_frameworks={PyArrowTable},
        plugin_collector=plugin_collector,
        global_filter=global_filter,
    )

    temperature_table = None
    for table in result:
        if "temperature" in table.schema.names:
            temperature_table = table
            break

    assert temperature_table is not None, "PyArrow Table with temperature column not found"

    actual_temps = sorted(temperature_table.column("temperature").to_pylist())
    assert actual_temps == EXPECTED_TEMPERATURES
    assert temperature_table.num_rows == EXPECTED_ROW_COUNT


def test_python_dict_tz_aware_time_range_filter() -> None:
    """Regression guard for #435 on the python_dict filter engine."""

    class PythonDictTzAwareSource(FeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({"temperature", DefaultOptionKeys.reference_time})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return [
                {
                    "temperature": TEMPERATURES[i],
                    DefaultOptionKeys.reference_time: TIMESTAMPS_UTC[i],
                }
                for i in range(len(TIMESTAMPS_UTC))
            ]

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {PythonDictFramework}

    global_filter = GlobalFilter()
    global_filter.add_time_and_time_travel_filters(event_from=EVENT_FROM, event_to=EVENT_TO)

    plugin_collector = PluginCollector.enabled_feature_groups({PythonDictTzAwareSource})

    result = mloda.run_all(
        ["temperature", DefaultOptionKeys.reference_time],
        compute_frameworks={PythonDictFramework},
        plugin_collector=plugin_collector,
        global_filter=global_filter,
    )

    rows = None
    for candidate in result:
        if candidate and isinstance(candidate, list) and "temperature" in candidate[0]:
            rows = candidate
            break

    assert rows is not None, "python_dict result with temperature column not found"

    actual_temps = sorted(row["temperature"] for row in rows)
    assert actual_temps == EXPECTED_TEMPERATURES
    assert len(rows) == EXPECTED_ROW_COUNT


@pytest.mark.skipif(pl is None, reason="Polars is not installed")
def test_polars_tz_aware_time_range_filter() -> None:
    """Regression guard for #435 on the Polars filter engine."""

    class PolarsTzAwareSource(FeatureGroup):
        @classmethod
        def input_data(cls) -> Optional[BaseInputData]:
            return DataCreator({"temperature", DefaultOptionKeys.reference_time})

        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return pl.DataFrame(
                {
                    "temperature": TEMPERATURES,
                    DefaultOptionKeys.reference_time: TIMESTAMPS_UTC,
                },
                schema={
                    "temperature": pl.Int64,
                    DefaultOptionKeys.reference_time: pl.Datetime(time_unit="us", time_zone="UTC"),
                },
            )

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
            return {PolarsDataFrame}

    global_filter = GlobalFilter()
    global_filter.add_time_and_time_travel_filters(event_from=EVENT_FROM, event_to=EVENT_TO)

    plugin_collector = PluginCollector.enabled_feature_groups({PolarsTzAwareSource})

    result = mloda.run_all(
        ["temperature", DefaultOptionKeys.reference_time],
        compute_frameworks={PolarsDataFrame},
        plugin_collector=plugin_collector,
        global_filter=global_filter,
    )

    temperature_df = None
    for df in result:
        if "temperature" in df.columns:
            temperature_df = df
            break

    assert temperature_df is not None, "Polars DataFrame with temperature column not found"

    actual_temps = sorted(temperature_df["temperature"].to_list())
    assert actual_temps == EXPECTED_TEMPERATURES
    assert temperature_df.height == EXPECTED_ROW_COUNT
