"""Verify the pyarrow aggregation/time-window feature groups treat numpy as a point-of-use optional backend."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_IMPORT_BODY: str = """
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup
print("OK")
"""

_SINGLE_COLUMN_BODY: str = """
import sys

import pyarrow as pa

from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup

table = pa.table({"a": [1.0, 2.0, 3.0]})
result = PyArrowAggregatedFeatureGroup._perform_aggregation(table, "sum", ["a"])
out = PyArrowAggregatedFeatureGroup._add_result_to_data(table, "agg", result)

assert out.column("agg").to_pylist() == [6.0, 6.0, 6.0], f"unexpected result: {out.column('agg').to_pylist()}"
assert "numpy" not in sys.modules, "single-column aggregation must not import numpy"
print("OK")
"""

_AGGREGATED_MULTI_COLUMN_BODY: str = """
import pyarrow as pa

from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup

table = pa.table({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

try:
    PyArrowAggregatedFeatureGroup._perform_aggregation(table, "sum", ["a", "b"])
    print("NO_RAISE")
except ImportError as e:
    if "mloda[numpy]" in str(e):
        print("IMPORTERROR_WITH_HINT")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__ + ":" + str(e))
"""

_TIME_WINDOW_MULTI_COLUMN_BODY: str = """
import datetime

import pyarrow as pa

from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup

table = pa.table({
    "event_time": pa.array(
        [datetime.datetime(2024, 1, 1, 0, 0, i) for i in range(5)], type=pa.timestamp("us")
    ),
    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    "b": [5.0, 4.0, 3.0, 2.0, 1.0],
})

try:
    PyArrowTimeWindowFeatureGroup._perform_window_operation(
        table, "sum", 1, "second", ["a", "b"], time_filter_feature="event_time"
    )
    print("NO_RAISE")
except ImportError as e:
    if "mloda[numpy]" in str(e):
        print("IMPORTERROR_WITH_HINT")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__ + ":" + str(e))
"""


@pytest.mark.timeout(30)
def test_aggregated_feature_group_imports_with_numpy_blocked() -> None:
    """PyArrowAggregatedFeatureGroup and PyArrowTimeWindowFeatureGroup must both import without numpy installed."""
    pytest.importorskip("pyarrow")
    result = run_blocked(_IMPORT_BODY, module="numpy")
    assert result.returncode == 0, f"Import failed.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"Expected OK sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"


@pytest.mark.timeout(30)
def test_aggregated_feature_group_single_column_works_without_numpy() -> None:
    """Single-column aggregation must produce correct results without numpy installed at all."""
    pytest.importorskip("pyarrow")
    result = run_blocked(_SINGLE_COLUMN_BODY, module="numpy")
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    assert "OK" in result.stdout, f"Expected OK sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"


@pytest.mark.timeout(30)
def test_aggregated_feature_group_multi_column_raises_clear_error_without_numpy() -> None:
    """Multi-column aggregation without numpy must raise ImportError naming the mloda[numpy] extra."""
    pytest.importorskip("pyarrow")
    result = run_blocked(_AGGREGATED_MULTI_COLUMN_BODY, module="numpy")
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    assert "IMPORTERROR_WITH_HINT" in result.stdout, (
        f"Expected IMPORTERROR_WITH_HINT sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )


@pytest.mark.timeout(30)
def test_time_window_multi_column_raises_clear_error_without_numpy() -> None:
    """Multi-column time-window aggregation without numpy must raise ImportError naming the mloda[numpy] extra."""
    pytest.importorskip("pyarrow")
    result = run_blocked(_TIME_WINDOW_MULTI_COLUMN_BODY, module="numpy")
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    assert "IMPORTERROR_WITH_HINT" in result.stdout, (
        f"Expected IMPORTERROR_WITH_HINT sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
