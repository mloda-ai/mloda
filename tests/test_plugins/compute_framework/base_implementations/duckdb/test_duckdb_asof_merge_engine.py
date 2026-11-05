"""
Tests for DuckDBMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineSuite. Preserves the DuckDB-specific
test that 'nearest' direction raises ValueError.
"""

from datetime import timedelta
from typing import Any

import pytest

from mloda._testing.compute_framework.merge_suite import AsofMergeEngineSuite, _norm
from mloda._testing.framework_adapter import DuckDBAdapter, FrameworkAdapter
from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment, unused-ignore]


def _frame(adapter: FrameworkAdapter, rows: list[dict[str, Any]]) -> Any:
    return adapter.from_arrow(pa.Table.from_pylist(rows))


class TestDuckDBAsofMergeEngine(AsofMergeEngineSuite):
    """Unit tests for DuckDBMergeEngine.merge_asof."""

    adapter_class = DuckDBAdapter

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """DuckDB raises duckdb.Error subclasses on CAST AS TIMESTAMP failures (lazy, so
        possibly at materialization)."""
        return (duckdb.Error, ValueError)

    def test_coerce_offset_string_raises(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
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
        engine = merge_engine

        left_offset = _frame(adapter, [{"k": "a", "t": "2025-06-03T00:00:00+02:00", "lv": 1}])
        right_offset = _frame(adapter, [{"k": "a", "t": "2025-06-01T00:00:00+02:00", "rv": 1.0}])
        with pytest.raises(ValueError, match=r"'t'.*(offset|timezone|tz)"):
            engine.merge_asof(left_offset, right_offset, Index(("k",)), Index(("k",)), cfg)

        left_z = _frame(adapter, [{"k": "a", "t": "2025-06-03T00:00:00Z", "lv": 1}])
        right_z = _frame(adapter, [{"k": "a", "t": "2025-06-01T00:00:00Z", "rv": 1.0}])
        with pytest.raises(ValueError, match=r"'t'.*(offset|timezone|tz)"):
            engine.merge_asof(left_z, right_z, Index(("k",)), Index(("k",)), cfg)

    @pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed")
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

    @pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed")
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

    def test_case_only_output_collision_raises(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        left = _frame(adapter, [{"id": 1, "ts": 10, "val": 100}])
        right = _frame(adapter, [{"id": 1, "ts": 8, "Val": 7}])
        engine = merge_engine
        cfg = AsOfJoinConfig(left_time_column="ts", right_time_column="ts", direction="backward")
        with pytest.raises(ValueError, match="rename"):
            engine.merge_asof(left, right, Index(("id",)), Index(("id",)), cfg)

    @pytest.mark.parametrize("bad_side", ["left", "right"])
    @pytest.mark.parametrize("key_kind", ["by", "time"])
    def test_case_mismatched_key_raises(
        self, key_kind: str, bad_side: str, adapter: FrameworkAdapter, merge_engine: Any
    ) -> None:
        left = _frame(adapter, [{"id": 1, "ts": 10, "val": 100}])
        right = _frame(adapter, [{"id": 1, "ts": 8, "rv": 7}])
        engine = merge_engine
        left_by, right_by, left_t, right_t = "id", "id", "ts", "ts"
        if key_kind == "by":
            left_by, right_by = ("ID", "id") if bad_side == "left" else ("id", "ID")
            bad = "ID"
        else:
            left_t, right_t = ("TS", "ts") if bad_side == "left" else ("ts", "TS")
            bad = "TS"
        cfg = AsOfJoinConfig(left_time_column=left_t, right_time_column=right_t, direction="backward")
        with pytest.raises(ValueError, match=bad):
            engine.merge_asof(left, right, Index((left_by,)), Index((right_by,)), cfg)

    @pytest.mark.parametrize("helper", ["_mloda_lid", "_MLODA_RN", "_mloda_rn", "_MLODA_LID"])
    def test_left_column_named_like_helper_is_safe(
        self, helper: str, adapter: FrameworkAdapter, merge_engine: Any
    ) -> None:
        hv = [1, 1, 1] if "lid" in helper.lower() else [111, 222, 333]
        left = _frame(
            adapter,
            [{"id": 1, "ts": 10, helper: hv[0]}, {"id": 1, "ts": 20, helper: hv[1]}, {"id": 2, "ts": 5, helper: hv[2]}],
        )
        right = _frame(adapter, [{"id": 1, "ts": 8, "rv": 100}, {"id": 1, "ts": 15, "rv": 200}])
        engine = merge_engine
        cfg = AsOfJoinConfig(left_time_column="ts", right_time_column="ts", direction="backward")
        result = adapter.rows(engine.merge_asof(left, right, Index(("id",)), Index(("id",)), cfg))
        rows = sorted(result, key=lambda r: (r["id"], r["ts"]))
        assert [r[helper] for r in rows] == hv
        assert [_norm(r["rv"]) for r in rows] == [100, 200, None]

    def test_left_column_named_like_right_rename_helper_is_safe(
        self, adapter: FrameworkAdapter, merge_engine: Any
    ) -> None:
        left = _frame(adapter, [{"id": 1, "ts": 10, "_mloda_r_val": 5}, {"id": 1, "ts": 20, "_mloda_r_val": 6}])
        right = _frame(adapter, [{"id": 1, "ts": 8, "val": 100}, {"id": 1, "ts": 15, "val": 200}])
        engine = merge_engine
        cfg = AsOfJoinConfig(left_time_column="ts", right_time_column="ts", direction="backward")
        result = adapter.rows(engine.merge_asof(left, right, Index(("id",)), Index(("id",)), cfg))
        rows = sorted(result, key=lambda r: r["ts"])
        assert [r["_mloda_r_val"] for r in rows] == [5, 6]
        assert [_norm(r["val"]) for r in rows] == [100, 200]
