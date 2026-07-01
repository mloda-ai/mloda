"""
Tests for PyArrowMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. PyArrow is both the framework
under test and the interchange format used by the DataConverter.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for PyArrowMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PyArrowMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pa is None:
            raise ImportError("PyArrow is not installed")
        table_type: type[Any] = pa.Table
        return table_type

    def get_connection(self) -> Optional[Any]:
        return None

    def test_nearest_direction(self) -> None:
        """Vector F: native PyArrow Acero (Table.join_asof) cannot express 'nearest' -> ValueError.

        Acero's asof join only supports a directional (forward/backward) tolerance match;
        a symmetric 'nearest' search is not available, so it must be rejected like the SQL engines.
        """
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_allow_exact_matches_false(self) -> None:
        """Override Vector C: Acero's asof match range always includes exact matches.

        Native PyArrow ``Table.join_asof`` performs an inclusive tolerance comparison, so
        there is no way to exclude an exact-time match. ``allow_exact_matches=False`` must
        therefore be rejected with a ValueError rather than silently returning the exact row.
        """
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            allow_exact_matches=False,
        )
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tolerance_timedelta_rejected(self) -> None:
        """Acero requires a numeric (integer) tolerance; a timedelta cannot be expressed -> ValueError."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=timedelta(seconds=5),
        )
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tolerance_float_rejected(self) -> None:
        """A non-integer float tolerance cannot be expressed as an Acero integer span -> ValueError."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=5.5,
        )
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tolerance_float_integer_valued_accepted(self) -> None:
        """An integer-valued float tolerance (5.0) must be accepted and behave like the int."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=5.0,
        )
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
        rows = result.to_pylist()
        assert len(rows) == 1
        assert rows[0]["rv"] == "A"

    def test_tolerance_bool_rejected(self) -> None:
        """A boolean tolerance must be rejected; bool is an int subclass and must not become 1."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [8, 15], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=True,
        )
        with pytest.raises(ValueError, match="PyArrowMergeEngine"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_unbounded_tolerance_extreme_int64_no_overflow(self) -> None:
        """Unbounded tolerance on extreme int64 on-keys (forward) must not raise OverflowError."""
        left = pa.Table.from_pydict({"k": [1], "t": pa.array([-(2**62)], pa.int64()), "lv": [1]})
        right = pa.Table.from_pydict({"k": [1], "t": pa.array([2**62], pa.int64()), "rv": [5]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
        rows = result.to_pylist()
        assert len(rows) == 1
        assert rows[0]["lv"] == 1

    def test_forward_tolerance_none(self) -> None:
        """Forward direction with unbounded (None) tolerance: nearest right.t >= left.t."""
        left = pa.Table.from_pydict({"k": [1, 1], "t": [10, 20], "lv": [100, 200]})
        right = pa.Table.from_pydict({"k": [1, 1], "t": [5, 18], "rv": ["A", "B"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
        rows = sorted(result.to_pylist(), key=lambda r: r["t"])
        assert len(rows) == 2
        assert rows[0]["lv"] == 100
        assert rows[0]["rv"] == "B"
        assert rows[1]["lv"] == 200
        assert rows[1]["rv"] is None

    def test_colliding_right_value_column_dropped(self) -> None:
        """A right VALUE column colliding with a left column is dropped; the left column survives."""
        left = pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100], "shared": ["L"]})
        right = pa.Table.from_pydict({"k": [1], "t": [5], "rv": ["X"], "shared": ["R"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
        rows = result.to_pylist()
        assert len(rows) == 1
        assert result.column_names.count("shared") == 1
        assert rows[0]["shared"] == "L"
        assert rows[0]["rv"] == "X"

    def test_asof_tz_aware_left_naive_right_raises(self) -> None:
        """Cross-side timezone guard (epic #518, Phase 2): a tz-AWARE left time column joined
        against a tz-NAIVE right time column must raise a clear ValueError mentioning timezone.

        Both columns are ordered timestamps, so the ordered-only guard admits them; the new
        ComparisonContract.require_compatible cross-side check must reject the naive-vs-aware
        mix BEFORE Acero surfaces its low-level on-key type-mismatch error.
        """
        left = pa.Table.from_pydict(
            {
                "k": [1, 1],
                "t": pa.array(
                    [
                        datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
                        datetime(2021, 1, 1, 0, 0, 20, tzinfo=timezone.utc),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "lv": [1, 2],
            }
        )
        right = pa.Table.from_pydict(
            {
                "k": [1],
                "t": pa.array([datetime(2021, 1, 1, 0, 0, 8)], type=pa.timestamp("us")),
                "rv": ["A"],
            }
        )

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_asof_both_tz_aware_succeeds(self) -> None:
        """False-positive guard for the cross-side timezone check: when BOTH sides are
        tz-aware (same tz), the as-of join must still succeed and match the prior right row."""
        left = pa.Table.from_pydict(
            {
                "k": [1],
                "t": pa.array([datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc)], type=pa.timestamp("us", tz="UTC")),
                "lv": [100],
            }
        )
        right = pa.Table.from_pydict(
            {
                "k": [1],
                "t": pa.array([datetime(2021, 1, 1, 0, 0, 8, tzinfo=timezone.utc)], type=pa.timestamp("us", tz="UTC")),
                "rv": ["A"],
            }
        )

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        rows = result.to_pylist()
        assert len(rows) == 1
        assert rows[0]["lv"] == 100
        assert rows[0]["rv"] == "A"

    def test_differing_string_key_carry(self) -> None:
        """Differing-name string by-keys (lk vs rk) survive via the carry column."""
        left = pa.Table.from_pydict({"lk": ["a"], "lt": [10], "lv": [100]})
        right = pa.Table.from_pydict({"rk": ["a"], "rt": [5], "rv": ["X"]})

        engine = PyArrowMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="lt", right_time_column="rt", direction="backward")
        result = engine.merge_asof(left, right, Index(("lk",)), Index(("rk",)), cfg)
        rows = result.to_pylist()
        assert len(rows) == 1
        assert "lk" in result.column_names
        assert "rk" in result.column_names
        assert rows[0]["lk"] == "a"
        assert rows[0]["rk"] == "a"
        assert rows[0]["rv"] == "X"
