"""
Tests for DuckDBMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. Preserves the DuckDB-specific
test that 'nearest' direction raises ValueError.
"""

from datetime import timedelta
from typing import Any, Optional

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for DuckDBMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return DuckDBMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if duckdb is None:
            raise ImportError("DuckDB is not installed")
        return DuckdbRelation

    def get_connection(self) -> Optional[Any]:
        """DuckDB requires a connection object."""
        if not hasattr(self, "_connection"):
            self._connection = duckdb.connect()
        return self._connection

    def test_nearest_raises_value_error(self) -> None:
        """Vector F: DuckDB native ASOF cannot express 'nearest' in v1 -> ValueError."""
        import duckdb as _duckdb  # noqa: PLC0415

        conn = _duckdb.connect()
        left = DuckdbRelation.from_arrow(conn, pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]}))
        right = DuckdbRelation.from_arrow(conn, pa.Table.from_pydict({"k": [1], "t": [8], "rv": [7]}))

        engine = DuckDBMergeEngine(conn)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        with pytest.raises(ValueError):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_timedelta_tolerance_raises_value_error(self) -> None:
        """A timedelta tolerance is unsupported on the DuckDB SQL backend (which needs a numeric
        gap). It must raise a clear ValueError mentioning 'timedelta', not the confusing
        TypeError from float(timedelta)."""
        import duckdb as _duckdb  # noqa: PLC0415

        conn = _duckdb.connect()
        left = DuckdbRelation.from_arrow(conn, pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]}))
        right = DuckdbRelation.from_arrow(conn, pa.Table.from_pydict({"k": [1], "t": [8], "rv": [7]}))

        engine = DuckDBMergeEngine(conn)
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=timedelta(seconds=5),
        )
        with pytest.raises(ValueError, match="timedelta"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
