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
    pa = None  # type: ignore[assignment, unused-ignore]


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

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """DuckDB raises duckdb.Error subclasses on CAST AS TIMESTAMP failures (lazy, so
        possibly at materialization)."""
        return (duckdb.Error, ValueError)

    def test_coerce_offset_string_raises(self) -> None:
        """DuckDB's CAST AS TIMESTAMP silently DROPS a UTC offset ('2025-06-01T05:00:00+02:00'
        becomes 05:00 local wall time, true UTC is 03:00), which corrupts as-of ordering.
        With coerce_time_columns=True, any time string carrying a UTC offset or a trailing 'Z'
        must therefore raise ValueError eagerly (before the cast), naming the column 't' and
        mentioning the offset/timezone problem."""
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )
        engine = DuckDBMergeEngine(self.get_connection())

        left_offset = self.convert_dict_to_framework([{"k": "a", "t": "2025-06-03T00:00:00+02:00", "lv": 1}])
        right_offset = self.convert_dict_to_framework([{"k": "a", "t": "2025-06-01T00:00:00+02:00", "rv": 1.0}])
        with pytest.raises(ValueError, match=r"'t'.*(offset|timezone|tz)"):
            engine.merge_asof(left_offset, right_offset, Index(("k",)), Index(("k",)), cfg)

        left_z = self.convert_dict_to_framework([{"k": "a", "t": "2025-06-03T00:00:00Z", "lv": 1}])
        right_z = self.convert_dict_to_framework([{"k": "a", "t": "2025-06-01T00:00:00Z", "rv": 1.0}])
        with pytest.raises(ValueError, match=r"'t'.*(offset|timezone|tz)"):
            engine.merge_asof(left_z, right_z, Index(("k",)), Index(("k",)), cfg)

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
