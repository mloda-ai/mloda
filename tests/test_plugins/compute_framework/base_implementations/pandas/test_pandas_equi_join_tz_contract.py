"""
Cross-side timezone guard for EQUI-joins (epic #518, Phase 3), pandas engine.

Phase 2 wired ComparisonContract.require_compatible into as-of joins. Phase 3
extends the SAME strict-timezone cross-side guard to equi-joins (INNER/LEFT/
RIGHT/OUTER) but ONLY when BOTH join-key columns are temporal datetimes. The
policy is narrow: string / integer / other non-temporal equi-joins must remain
completely legal and raise nothing new.

The negative test below must FAIL today: no timezone guard exists on the
equi-join path yet, so pandas either merges or surfaces its own low-level
"incompatible merge keys" dtype error, neither of which mentions timezone.
"""

import pytest

from mloda.user import Index
from mloda.user import JoinType
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasEquiJoinTimezoneContract:
    """Narrow cross-side timezone guard on the equi-join path."""

    def test_inner_equi_join_tz_aware_left_naive_right_raises(self) -> None:
        """INNER equi-join whose join KEY is tz-AWARE on the left and tz-NAIVE on the
        right must raise a clear ValueError mentioning timezone.

        Both keys are ordered datetime64 columns, so nothing in the current equi-join
        path rejects the naive-vs-aware mix on semantic grounds; the Phase 3 guard must
        call ComparisonContract.require_compatible on the aligned key pair and reject it.
        """
        left = pd.DataFrame(
            {
                "t": [
                    pd.Timestamp("2021-01-01 00:00:10", tz="UTC"),
                    pd.Timestamp("2021-01-01 00:00:20", tz="UTC"),
                ],
                "lv": [1, 2],
            }
        )
        right = pd.DataFrame({"t": [pd.Timestamp("2021-01-01 00:00:10")], "rv": [7.0]})

        engine = PandasMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            engine.merge(left, right, link)

    def test_inner_equi_join_both_tz_naive_succeeds(self) -> None:
        """False-positive guard: an equi-join with BOTH keys tz-naive must still succeed."""
        left = pd.DataFrame(
            {
                "t": [
                    pd.Timestamp("2021-01-01 00:00:10"),
                    pd.Timestamp("2021-01-01 00:00:20"),
                ],
                "lv": [1, 2],
            }
        )
        right = pd.DataFrame({"t": [pd.Timestamp("2021-01-01 00:00:10")], "rv": [7.0]})

        engine = PandasMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        result = engine.merge(left, right, link)

        rows = result.to_dict(orient="records")
        assert len(rows) == 1
        assert rows[0]["lv"] == 1
        assert rows[0]["rv"] == 7.0

    def test_inner_equi_join_both_tz_aware_succeeds(self) -> None:
        """False-positive guard: an equi-join with BOTH keys tz-aware (same tz) must succeed."""
        left = pd.DataFrame(
            {
                "t": [
                    pd.Timestamp("2021-01-01 00:00:10", tz="UTC"),
                    pd.Timestamp("2021-01-01 00:00:20", tz="UTC"),
                ],
                "lv": [1, 2],
            }
        )
        right = pd.DataFrame({"t": [pd.Timestamp("2021-01-01 00:00:10", tz="UTC")], "rv": [7.0]})

        engine = PandasMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        result = engine.merge(left, right, link)

        rows = result.to_dict(orient="records")
        assert len(rows) == 1
        assert rows[0]["lv"] == 1
        assert rows[0]["rv"] == 7.0

    def test_inner_equi_join_string_keys_still_legal(self) -> None:
        """Narrow-policy guard: a STRING-key equi-join must not be touched by the temporal
        timezone guard and must keep working exactly as before."""
        left = pd.DataFrame({"k": ["a", "b"], "lv": [1, 2]})
        right = pd.DataFrame({"k": ["a"], "rv": [7.0]})

        engine = PandasMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        result = engine.merge(left, right, link)

        rows = result.to_dict(orient="records")
        assert len(rows) == 1
        assert rows[0]["k"] == "a"
        assert rows[0]["rv"] == 7.0

    def test_inner_equi_join_integer_keys_still_legal(self) -> None:
        """Narrow-policy guard: an INTEGER-key equi-join must remain legal and unchanged."""
        left = pd.DataFrame({"k": [1, 2], "lv": [1, 2]})
        right = pd.DataFrame({"k": [1], "rv": [7.0]})

        engine = PandasMergeEngine()
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        result = engine.merge(left, right, link)

        rows = result.to_dict(orient="records")
        assert len(rows) == 1
        assert rows[0]["k"] == 1
        assert rows[0]["rv"] == 7.0
