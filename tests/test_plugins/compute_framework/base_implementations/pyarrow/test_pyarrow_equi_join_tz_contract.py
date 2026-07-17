"""
Cross-side timezone guard for EQUI-joins (epic #518, Phase 3), PyArrow engine.

Phase 2 wired ComparisonContract.require_compatible into as-of joins. Phase 3
extends the SAME strict-timezone cross-side guard to equi-joins (INNER/LEFT/
RIGHT/OUTER) but ONLY when BOTH join-key columns are temporal datetimes. The
policy is narrow: string / integer / other non-temporal equi-joins must remain
completely legal and raise nothing new.

The negative test below must FAIL today: no timezone guard exists on the
equi-join path yet, so Acero either merges or surfaces its own low-level on-key
type mismatch, neither of which mentions timezone.
"""

from datetime import datetime, timezone

import pytest

from mloda.user import Index
from mloda.user import JoinType
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_merge_engine import PyArrowMergeEngine
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowEquiJoinTimezoneContract:
    """Narrow cross-side timezone guard on the equi-join path."""

    def test_inner_equi_join_tz_aware_left_naive_right_raises(self) -> None:
        """INNER equi-join whose join KEY is tz-AWARE on the left and tz-NAIVE on the
        right must raise a clear ValueError mentioning timezone.

        Both keys are ordered timestamp columns, so the current equi-join path admits
        them; the Phase 3 guard must call ComparisonContract.require_compatible on the
        aligned key pair and reject the naive-vs-aware mix.
        """
        left = pa.Table.from_pydict(
            {
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
                "t": pa.array([datetime(2021, 1, 1, 0, 0, 10)], type=pa.timestamp("us")),
                "rv": ["A"],
            }
        )

        engine = PyArrowMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            engine.merge(left, right, link)

    def test_inner_equi_join_both_tz_naive_succeeds(self) -> None:
        """False-positive guard: an equi-join with BOTH keys tz-naive must still succeed."""
        left = pa.Table.from_pydict(
            {
                "t": pa.array(
                    [datetime(2021, 1, 1, 0, 0, 10), datetime(2021, 1, 1, 0, 0, 20)],
                    type=pa.timestamp("us"),
                ),
                "lv": [1, 2],
            }
        )
        right = pa.Table.from_pydict(
            {
                "t": pa.array([datetime(2021, 1, 1, 0, 0, 10)], type=pa.timestamp("us")),
                "rv": ["A"],
            }
        )

        engine = PyArrowMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        result = engine.merge(left, right, link)

        rows = result.to_pylist()
        assert len(rows) == 1
        assert rows[0]["lv"] == 1
        assert rows[0]["rv"] == "A"

    def test_inner_equi_join_both_tz_aware_succeeds(self) -> None:
        """False-positive guard: an equi-join with BOTH keys tz-aware (same tz) must succeed."""
        left = pa.Table.from_pydict(
            {
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
                "t": pa.array(
                    [datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "rv": ["A"],
            }
        )

        engine = PyArrowMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        result = engine.merge(left, right, link)

        rows = result.to_pylist()
        assert len(rows) == 1
        assert rows[0]["lv"] == 1
        assert rows[0]["rv"] == "A"

    def test_inner_equi_join_string_keys_still_legal(self) -> None:
        """Narrow-policy guard: a STRING-key equi-join must not be touched by the temporal
        timezone guard and must keep working exactly as before."""
        left = pa.Table.from_pydict({"k": ["a", "b"], "lv": [1, 2]})
        right = pa.Table.from_pydict({"k": ["a"], "rv": ["A"]})

        engine = PyArrowMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        result = engine.merge(left, right, link)

        rows = result.to_pylist()
        assert len(rows) == 1
        assert rows[0]["k"] == "a"
        assert rows[0]["rv"] == "A"

    def test_inner_equi_join_integer_keys_still_legal(self) -> None:
        """Narrow-policy guard: an INTEGER-key equi-join must remain legal and unchanged."""
        left = pa.Table.from_pydict({"k": [1, 2], "lv": [1, 2]})
        right = pa.Table.from_pydict({"k": [1], "rv": ["A"]})

        engine = PyArrowMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        result = engine.merge(left, right, link)

        rows = result.to_pylist()
        assert len(rows) == 1
        assert rows[0]["k"] == 1
        assert rows[0]["rv"] == "A"
