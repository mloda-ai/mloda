"""Cross-framework merge-engine conformance suite.

Runs EVERY merge engine against the SAME canonical expected frames (see
``merge_conformance_scenarios``). Results are compared as an order-independent,
column-order-independent, null-normalized multiset of rows, so each engine must
agree on the exact contract regardless of internal representation.
"""

import collections
import math
import sqlite3
from typing import Any, Callable, NamedTuple

import pytest

from mloda.provider import BaseMergeEngine
from mloda.user import Index, JoinType
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import _regexp
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine import SqliteMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_plugins.compute_framework.test_tooling.merge_conformance.merge_conformance_scenarios import (
    JOIN_TYPE_NAMES,
    MERGE_CONFORMANCE_SCENARIOS,
)
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link
from tests.test_plugins.compute_framework.test_tooling.multi_index.test_data_converter import DataConverter

try:
    import pandas as pd

    _PANDAS_TYPE: type[Any] | None = pd.DataFrame
except ImportError:
    _PANDAS_TYPE = None

try:
    import pyarrow as pa

    _PYARROW_TYPE: type[Any] | None = pa.Table
except ImportError:
    _PYARROW_TYPE = None

try:
    import polars as pl

    _POLARS_TYPE: type[Any] | None = pl.DataFrame
    _POLARS_LAZY_TYPE: type[Any] | None = pl.LazyFrame
except ImportError:
    _POLARS_TYPE = None
    _POLARS_LAZY_TYPE = None

try:
    import duckdb

    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

    _DUCKDB_TYPE: type[Any] | None = DuckdbRelation
except ImportError:
    duckdb = None  # type: ignore[assignment]
    _DUCKDB_TYPE = None


def _duckdb_connect() -> Any:
    return duckdb.connect()


def _sqlite_connect() -> Any:
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP", 2, _regexp)
    return conn


class _Framework(NamedTuple):
    """One merge engine under test with its native type and optional connection factory."""

    name: str
    engine_class: type[BaseMergeEngine]
    framework_type: type[Any] | None
    make_connection: Callable[[], Any] | None


# sqlite3 is stdlib, so SqliteRelation is always available.
FRAMEWORKS: list[_Framework] = [
    _Framework("pandas", PandasMergeEngine, _PANDAS_TYPE, None),
    _Framework("pyarrow", PyArrowMergeEngine, _PYARROW_TYPE, None),
    _Framework("polars", PolarsMergeEngine, _POLARS_TYPE, None),
    _Framework("polars_lazy", PolarsLazyMergeEngine, _POLARS_LAZY_TYPE, None),
    _Framework("python_dict", PythonDictMergeEngine, dict, None),
    _Framework("duckdb", DuckDBMergeEngine, _DUCKDB_TYPE, _duckdb_connect),
    _Framework("sqlite", SqliteMergeEngine, SqliteRelation, _sqlite_connect),
]

_PARAMS = [
    pytest.param(fw, scenario_key, join_name, id=f"{fw.name}-{scenario_key}-{join_name}")
    for fw in FRAMEWORKS
    for scenario_key in MERGE_CONFORMANCE_SCENARIOS
    for join_name in JOIN_TYPE_NAMES
]


def _norm(value: Any) -> Any:
    """None stays None; float NaN collapses to None so nullable-int widening never matters."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _multiset(rows: list[dict[str, Any]]) -> collections.Counter[frozenset[tuple[str, Any]]]:
    """Order- and column-order-independent, null-normalized bag of rows (1 == 1.0)."""
    return collections.Counter(frozenset((col, _norm(val)) for col, val in row.items()) for row in rows)


@pytest.mark.parametrize("framework, scenario_key, join_name", _PARAMS)
def test_merge_conformance(framework: _Framework, scenario_key: str, join_name: str) -> None:
    if framework.framework_type is None:
        pytest.skip(f"{framework.name} is not installed")

    scenario = MERGE_CONFORMANCE_SCENARIOS[scenario_key]
    join_type = JoinType[join_name]
    framework_type = framework.framework_type

    converter = DataConverter()
    connection = framework.make_connection() if framework.make_connection is not None else None

    left = converter.to_framework(scenario["left"], framework_type, connection)
    right = converter.to_framework(scenario["right"], framework_type, connection)
    link = make_merge_link(join_type, Index(scenario["left_index"]), Index(scenario["right_index"]))

    engine = framework.engine_class(connection) if connection is not None else framework.engine_class()
    result = engine.merge(left, right, link)
    actual = converter.from_framework(result, framework_type)

    expected = scenario["expected"][join_name]
    assert _multiset(actual) == _multiset(expected), (
        f"Merge conformance mismatch [{framework.name} / {scenario_key} / {join_name}]:\n"
        f"  expected={expected}\n"
        f"  actual={actual}"
    )
