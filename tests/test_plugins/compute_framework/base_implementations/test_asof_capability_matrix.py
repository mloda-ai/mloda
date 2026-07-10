"""Regression-lock for the per-backend as-of (point-in-time join) capability matrix.

Pins each real merge engine's ``nearest`` / ``timedelta`` / ``exclude_exact`` behavior.
Keep ``ASOF_CAPABILITY_MATRIX`` in sync with docs/docs/in_depth/join_data.md.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pytest

from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda.user import Index
from tests.test_plugins.compute_framework.test_tooling.multi_index.test_data_converter import DataConverter

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import _regexp
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine import SqliteMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


# Mirrors docs/docs/in_depth/join_data.md; update both together.
# True = the capability succeeds, False = merge_asof raises ValueError.
# spark is intentionally excluded (not exercised in CI); it mirrors duckdb/sqlite:
# nearest=False, timedelta=False, exclude_exact=True.
ASOF_CAPABILITY_MATRIX: dict[str, dict[str, bool]] = {
    "pandas": {"nearest": True, "timedelta": True, "exclude_exact": True},
    "polars": {"nearest": True, "timedelta": True, "exclude_exact": True},
    "python_dict": {"nearest": True, "timedelta": True, "exclude_exact": True},
    "pyarrow": {"nearest": False, "timedelta": False, "exclude_exact": False},
    "duckdb": {"nearest": False, "timedelta": False, "exclude_exact": True},
    "sqlite": {"nearest": False, "timedelta": False, "exclude_exact": True},
}

CAPABILITIES = ("nearest", "timedelta", "exclude_exact")

# Substring each rejecting backend's ValueError message must contain, so an unrelated
# ValueError (e.g. from validate_asof_time_columns) cannot silently satisfy the test.
_RAISE_MATCH = {"nearest": "nearest", "timedelta": "timedelta", "exclude_exact": "allow_exact_matches"}


def _sqlite_connection() -> Any:
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP", 2, _regexp)
    return conn


class BackendSpec:
    """Bundles the engine class, framework type, and connection factory for a backend."""

    def __init__(
        self,
        name: str,
        engine_class: type[BaseMergeEngine] | None,
        framework_type: type[Any] | None,
        connection_factory: Callable[[], Optional[Any]],
        missing: bool,
    ) -> None:
        self.name = name
        self.engine_class = engine_class
        self.framework_type = framework_type
        self.connection_factory = connection_factory
        self.missing = missing


# PyArrow is the DataConverter interchange format, so every backend needs it.
BACKEND_SPECS: dict[str, BackendSpec] = {
    "pandas": BackendSpec(
        "pandas",
        PandasMergeEngine,
        pd.DataFrame if pd is not None else None,
        lambda: None,
        missing=pd is None or pa is None,
    ),
    "polars": BackendSpec(
        "polars",
        PolarsMergeEngine,
        pl.DataFrame if pl is not None else None,
        lambda: None,
        missing=pl is None or pa is None,
    ),
    "python_dict": BackendSpec(
        "python_dict",
        PythonDictMergeEngine,
        dict,
        lambda: None,
        missing=pa is None,
    ),
    "pyarrow": BackendSpec(
        "pyarrow",
        PyArrowMergeEngine,
        pa.Table if pa is not None else None,
        lambda: None,
        missing=pa is None,
    ),
    "duckdb": BackendSpec(
        "duckdb",
        DuckDBMergeEngine,
        DuckdbRelation,
        lambda: duckdb.connect() if duckdb is not None else None,
        missing=duckdb is None or pa is None,
    ),
    "sqlite": BackendSpec(
        "sqlite",
        SqliteMergeEngine,
        SqliteRelation,
        _sqlite_connection,
        missing=pa is None,
    ),
}


def _config_for(capability: str) -> AsOfJoinConfig:
    if capability == "nearest":
        return AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
    if capability == "exclude_exact":
        return AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=False
        )
    # timedelta
    return AsOfJoinConfig(
        left_time_column="t", right_time_column="t", direction="backward", tolerance=timedelta(seconds=5)
    )


def _rows_for(capability: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if capability == "timedelta":
        # Datetime time column makes the timedelta tolerance meaningful for succeeding backends.
        left = [{"k": 1, "t": datetime(2021, 1, 1, 0, 0, 10), "lv": 100}]
        right = [{"k": 1, "t": datetime(2021, 1, 1, 0, 0, 8), "rv": 1.0}]
        return left, right
    # Numeric time column for nearest and exclude_exact.
    left = [{"k": 1, "t": 10, "lv": 100}]
    right = [{"k": 1, "t": 10, "rv": 1.0}, {"k": 1, "t": 8, "rv": 2.0}]
    return left, right


def _params() -> list[Any]:
    params: list[Any] = []
    for backend, capabilities in ASOF_CAPABILITY_MATRIX.items():
        spec = BACKEND_SPECS[backend]
        for capability in CAPABILITIES:
            _ = capabilities[capability]
            params.append(
                pytest.param(
                    backend,
                    capability,
                    id=f"{backend}-{capability}",
                    marks=pytest.mark.skipif(
                        spec.missing, reason=f"Optional dependency for {backend} is not installed."
                    ),
                )
            )
    return params


@pytest.mark.parametrize("backend, capability", _params())
def test_asof_capability_matrix(backend: str, capability: str) -> None:
    """Each (backend, capability) must match the documented matrix against the real engine."""
    spec = BACKEND_SPECS[backend]
    expected = ASOF_CAPABILITY_MATRIX[backend][capability]

    conv = DataConverter()
    connection = spec.connection_factory()
    assert spec.engine_class is not None
    assert spec.framework_type is not None
    framework_type = spec.framework_type
    engine = spec.engine_class(connection) if connection is not None else spec.engine_class()

    left_rows, right_rows = _rows_for(capability)
    left = conv.to_framework(left_rows, framework_type, connection)
    right = conv.to_framework(right_rows, framework_type, connection)
    index = Index(("k",))
    config = _config_for(capability)

    if expected:
        result = engine.merge_asof(left, right, index, index, config)
        # Materialize lazy backends (duckdb / lazy relations); the call must not raise.
        conv.from_framework(result, framework_type)
    else:
        with pytest.raises(ValueError, match=_RAISE_MATCH[capability]):
            engine.merge_asof(left, right, index, index, config)
