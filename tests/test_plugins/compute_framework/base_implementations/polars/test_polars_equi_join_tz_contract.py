"""
Cross-side timezone guard for EQUI-joins (epic #518, Phase 3), polars engines.

Phase 2 wired ComparisonContract.require_compatible into as-of joins. Phase 3
extends the SAME strict-timezone cross-side guard to equi-joins (INNER/LEFT/
RIGHT/OUTER) but ONLY when BOTH join-key columns are temporal datetimes. The
policy is narrow: string / integer / other non-temporal equi-joins must remain
completely legal and raise nothing new.

Both the eager PolarsMergeEngine (pl.DataFrame) and the lazy
PolarsLazyMergeEngine (pl.LazyFrame) share the same checks via a mixin base.

The negative test below must FAIL today: no timezone guard exists on the
equi-join path yet, so polars either merges or surfaces its own low-level
schema/dtype error, neither of which mentions timezone.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from mloda.user import Index
from mloda.user import JoinType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


class _PolarsEquiJoinTzChecks:
    """Cross-side timezone equi-join guard tests shared by the eager and lazy polars engines."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        raise NotImplementedError

    @classmethod
    def is_lazy(cls) -> bool:
        raise NotImplementedError

    def _as_framework(self, frame: Any) -> Any:
        return frame.lazy() if self.is_lazy() else frame

    def _rows(self, result: Any) -> list[dict[str, Any]]:
        if hasattr(result, "collect"):
            result = result.collect()
        rows: list[dict[str, Any]] = result.to_dicts()
        return rows

    def test_inner_equi_join_tz_aware_left_naive_right_raises(self) -> None:
        """INNER equi-join whose join KEY is tz-AWARE on the left and tz-NAIVE on the
        right must raise a clear ValueError mentioning timezone.

        Both keys are ordered Datetime columns, so the current equi-join path admits
        them; the Phase 3 guard must call ComparisonContract.require_compatible on the
        aligned key pair and reject the naive-vs-aware mix.
        """
        left = self._as_framework(
            pl.DataFrame(
                {
                    "t": [
                        datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
                        datetime(2021, 1, 1, 0, 0, 20, tzinfo=timezone.utc),
                    ],
                    "lv": [1, 2],
                }
            )
        )
        right = self._as_framework(pl.DataFrame({"t": [datetime(2021, 1, 1, 0, 0, 10)], "rv": [7.0]}))

        engine = self.merge_engine_class()()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            result = engine.merge(left, right, link)
            if hasattr(result, "collect"):
                result.collect()

    def test_inner_equi_join_both_tz_naive_succeeds(self) -> None:
        """False-positive guard: an equi-join with BOTH keys tz-naive must still succeed."""
        left = self._as_framework(
            pl.DataFrame(
                {
                    "t": [datetime(2021, 1, 1, 0, 0, 10), datetime(2021, 1, 1, 0, 0, 20)],
                    "lv": [1, 2],
                }
            )
        )
        right = self._as_framework(pl.DataFrame({"t": [datetime(2021, 1, 1, 0, 0, 10)], "rv": [7.0]}))

        engine = self.merge_engine_class()()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        rows = self._rows(engine.merge(left, right, link))

        assert len(rows) == 1
        assert rows[0]["lv"] == 1
        assert rows[0]["rv"] == 7.0

    def test_inner_equi_join_both_tz_aware_succeeds(self) -> None:
        """False-positive guard: an equi-join with BOTH keys tz-aware (same tz) must succeed."""
        left = self._as_framework(
            pl.DataFrame(
                {
                    "t": [
                        datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
                        datetime(2021, 1, 1, 0, 0, 20, tzinfo=timezone.utc),
                    ],
                    "lv": [1, 2],
                }
            )
        )
        right = self._as_framework(
            pl.DataFrame({"t": [datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc)], "rv": [7.0]})
        )

        engine = self.merge_engine_class()()
        link = make_merge_link(JoinType.INNER, Index(("t",)), Index(("t",)))
        rows = self._rows(engine.merge(left, right, link))

        assert len(rows) == 1
        assert rows[0]["lv"] == 1
        assert rows[0]["rv"] == 7.0

    def test_inner_equi_join_string_keys_still_legal(self) -> None:
        """Narrow-policy guard: a STRING-key equi-join must not be touched by the temporal
        timezone guard and must keep working exactly as before."""
        left = self._as_framework(pl.DataFrame({"k": ["a", "b"], "lv": [1, 2]}))
        right = self._as_framework(pl.DataFrame({"k": ["a"], "rv": [7.0]}))

        engine = self.merge_engine_class()()
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        rows = self._rows(engine.merge(left, right, link))

        assert len(rows) == 1
        assert rows[0]["k"] == "a"
        assert rows[0]["rv"] == 7.0

    def test_inner_equi_join_integer_keys_still_legal(self) -> None:
        """Narrow-policy guard: an INTEGER-key equi-join must remain legal and unchanged."""
        left = self._as_framework(pl.DataFrame({"k": [1, 2], "lv": [1, 2]}))
        right = self._as_framework(pl.DataFrame({"k": [1], "rv": [7.0]}))

        engine = self.merge_engine_class()()
        link = make_merge_link(JoinType.INNER, Index(("k",)), Index(("k",)))
        rows = self._rows(engine.merge(left, right, link))

        assert len(rows) == 1
        assert rows[0]["k"] == 1
        assert rows[0]["rv"] == 7.0


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsEquiJoinTimezoneContract(_PolarsEquiJoinTzChecks):
    """Eager PolarsMergeEngine narrow cross-side timezone equi-join guard."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PolarsMergeEngine

    @classmethod
    def is_lazy(cls) -> bool:
        return False


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyEquiJoinTimezoneContract(_PolarsEquiJoinTzChecks):
    """Lazy PolarsLazyMergeEngine narrow cross-side timezone equi-join guard."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PolarsLazyMergeEngine

    @classmethod
    def is_lazy(cls) -> bool:
        return True
