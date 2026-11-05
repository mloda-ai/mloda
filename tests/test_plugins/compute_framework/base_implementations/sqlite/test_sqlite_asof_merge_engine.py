"""
Tests for SqliteMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineSuite. SQLite native ASOF cannot express
'nearest', so a negative test pins that 'nearest' raises a ValueError.
"""

from datetime import timedelta
from typing import Any

import pytest

import pyarrow as pa

from mloda._testing.compute_framework.merge_suite import AsofMergeEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, SqliteAdapter
from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig


def _frame(adapter: FrameworkAdapter, rows: list[dict[str, Any]]) -> Any:
    return adapter.from_arrow(pa.Table.from_pylist(rows))


class TestSqliteAsofMergeEngine(AsofMergeEngineSuite):
    """Unit tests for SqliteMergeEngine.merge_asof."""

    adapter_class = SqliteAdapter

    def test_coerce_one_sided_string_vs_numeric_raises(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """sqlite coercion turns ISO TEXT into julianday() day numbers. If only ONE side is a
        string column and the other side is already numeric (unknown unit, e.g. epoch seconds),
        the join would silently compare julian day numbers against epoch values. With
        coerce_time_columns=True, when exactly one side's time column needs coercion and the
        other is already ordered, the engine must raise ValueError naming the coerced column
        and advising to cast manually."""
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )
        engine = merge_engine

        left_string = _frame(adapter, [{"k": "a", "t": "2025-06-03T00:00:00", "lv": 1}])
        right_numeric = _frame(adapter, [{"k": "a", "t": 1.0, "rv": 1.0}, {"k": "a", "t": 2.0, "rv": 2.0}])
        with pytest.raises(ValueError, match=r"'t'.*cast"):
            engine.merge_asof(left_string, right_numeric, Index(("k",)), Index(("k",)), cfg)

        # Symmetric case: left numeric, right string must raise as well.
        left_numeric = _frame(adapter, [{"k": "a", "t": 1.0, "lv": 1}, {"k": "a", "t": 2.0, "lv": 2}])
        right_string = _frame(adapter, [{"k": "a", "t": "2025-06-03T00:00:00", "rv": 1.0}])
        with pytest.raises(ValueError, match=r"'t'.*cast"):
            engine.merge_asof(left_numeric, right_string, Index(("k",)), Index(("k",)), cfg)

    def test_nearest_raises_value_error(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Vector F: SQLite native ASOF cannot express 'nearest' in v1 -> ValueError."""
        left = _frame(adapter, [{"k": 1, "t": 10, "lv": 100}])
        right = _frame(adapter, [{"k": 1, "t": 8, "rv": 7}])

        engine = merge_engine
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        with pytest.raises(ValueError, match="nearest"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_timedelta_tolerance_raises_value_error(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """A timedelta tolerance is unsupported on the SQLite SQL backend (which needs a numeric
        gap). It must raise a clear ValueError mentioning 'timedelta', not the confusing
        TypeError from float(timedelta)."""
        left = _frame(adapter, [{"k": 1, "t": 10, "lv": 100}])
        right = _frame(adapter, [{"k": 1, "t": 8, "rv": 7}])

        engine = merge_engine
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=timedelta(seconds=5),
        )
        with pytest.raises(ValueError, match="timedelta"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tie_yields_single_row(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        """Tie contract: when two right rows share the identical boundary time within the same
        by-key, the ASOF join must still yield exactly ONE row per left row (parity with
        duckdb/pandas/python_dict/pyarrow). Ties are broken deterministically by ordering the
        candidate right rows on their non-key columns ascending, so the smaller rv (100) wins.
        """
        left = _frame(adapter, [{"k": 1, "t": 10, "lv": 1}])
        right = _frame(adapter, [{"k": 1, "t": 5, "rv": 100}, {"k": 1, "t": 5, "rv": 200}])

        engine = merge_engine
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_dicts = adapter.rows(result)
        assert len(result_dicts) == 1, f"expected 1 row, got {len(result_dicts)}: {result_dicts}"
        assert self._normalize_value(result_dicts[0]["rv"]) == 100

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
        assert [self._normalize_value(r["rv"]) for r in rows] == [100, 200, None]
