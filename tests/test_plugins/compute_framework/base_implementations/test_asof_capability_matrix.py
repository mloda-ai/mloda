"""Regression-lock for the per-backend as-of (point-in-time join) capability matrix.

Pins each real merge engine's ``nearest`` / ``timedelta`` / ``exclude_exact`` behavior.
Keep ``ASOF_CAPABILITY_MATRIX`` in sync with docs/docs/in_depth/join_data.md.
"""

from datetime import datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

from mloda._testing.framework_adapter import (
    DuckDBAdapter,
    FrameworkAdapter,
    PandasAdapter,
    PolarsAdapter,
    PyArrowAdapter,
    PythonDictAdapter,
    SqliteAdapter,
)
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.user import Index


# Mirrors docs/docs/in_depth/join_data.md; update both together.
# True = the capability succeeds, False = merge_asof raises ValueError.
# spark is excluded (its tests run only in the spark tox env); it mirrors duckdb/sqlite:
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


BACKEND_ADAPTERS: dict[str, type[FrameworkAdapter]] = {
    "pandas": PandasAdapter,
    "polars": PolarsAdapter,
    "python_dict": PythonDictAdapter,
    "pyarrow": PyArrowAdapter,
    "duckdb": DuckDBAdapter,
    "sqlite": SqliteAdapter,
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
        missing = not BACKEND_ADAPTERS[backend]().is_available()
        for capability in CAPABILITIES:
            _ = capabilities[capability]
            params.append(
                pytest.param(
                    backend,
                    capability,
                    id=f"{backend}-{capability}",
                    marks=pytest.mark.skipif(missing, reason=f"Optional dependency for {backend} is not installed."),
                )
            )
    return params


@pytest.mark.parametrize("backend, capability", _params())
def test_asof_capability_matrix(backend: str, capability: str) -> None:
    """Each (backend, capability) must match the documented matrix against the real engine."""
    expected = ASOF_CAPABILITY_MATRIX[backend][capability]

    adapter = BACKEND_ADAPTERS[backend]()
    with adapter:
        engine = adapter.compute_framework().merge_engine()(adapter.connection)

        left_rows, right_rows = _rows_for(capability)
        left = adapter.from_arrow(pa.Table.from_pylist(left_rows))
        right = adapter.from_arrow(pa.Table.from_pylist(right_rows))
        index = Index(("k",))
        config = _config_for(capability)

        if expected:
            result = engine.merge_asof(left, right, index, index, config)
            # Materialize lazy backends (duckdb / lazy relations); the call must not raise.
            adapter.rows(result)
        else:
            with pytest.raises(ValueError, match=_RAISE_MATCH[capability]):
                engine.merge_asof(left, right, index, index, config)
