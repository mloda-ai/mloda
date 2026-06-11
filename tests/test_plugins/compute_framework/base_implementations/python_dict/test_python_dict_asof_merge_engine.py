"""
Tests for PythonDictMergeEngine.merge_asof (point-in-time / as-of join).

Consumes the shared AsofMergeEngineTestBase. PythonDict's native format is
list[dict], so conversion is a passthrough; the DataConverter still routes
through PyArrow for other frameworks, so PyArrow must be available.
"""

from typing import Any, Optional

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPythonDictAsofMergeEngine(AsofMergeEngineTestBase):
    """Unit tests for PythonDictMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PythonDictMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        return list

    def get_connection(self) -> Optional[Any]:
        return None

    def test_nearest_direction(self) -> None:
        """Vector F: PythonDict supports 'nearest' -> closer right row (t=8, gap 2) matched."""
        left = [{"k": 1, "t": 10, "lv": 100}]
        right = [{"k": 1, "t": 8, "rv": "A"}, {"k": 1, "t": 15, "rv": "B"}]

        engine = PythonDictMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result[0]["rv"] == "A"

    def test_mixed_time_column_raises_value_error(self) -> None:
        """A heterogeneous time column (int then str) is not orderable. The guard must scan all
        non-null values, not just the first, and raise a clear ValueError naming the column
        instead of letting a raw comparison TypeError surface later."""
        left = [{"k": "a", "t": 1, "lv": 1}, {"k": "a", "t": "2025-06-01", "lv": 2}]
        right = [{"k": "a", "t": 1, "rv": 1.0}]

        engine = PythonDictMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t")
        with pytest.raises(ValueError, match=r"'t'.*(datetime|numeric)"):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

    def test_tie_breaks_ascending_and_is_order_independent(self) -> None:
        """
        Tie determinism: two right rows share the identical boundary timestamp within a
        by-key. The winner must be the row whose right non-key column values sort ASCENDING
        (smallest wins), matching the sqlite backend fix (fc805f0), and the result must be
        INDEPENDENT of right-input ordering.

        Left {k:1, t:10, lv:1}; two right rows tie at t=10 (== left t, backward, allow_exact
        default True): {k:1, t:10, rv:5} and {k:1, t:10, rv:2}. Expected exactly one row with
        rv==2 regardless of which order the tied right rows are supplied in.
        """
        left = [{"k": 1, "t": 10, "lv": 1}]
        right_a = [{"k": 1, "t": 10, "rv": 5}, {"k": 1, "t": 10, "rv": 2}]
        right_b = [{"k": 1, "t": 10, "rv": 2}, {"k": 1, "t": 10, "rv": 5}]

        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            allow_exact_matches=True,
        )

        result_a = PythonDictMergeEngine().merge_asof(left, right_a, Index(("k",)), Index(("k",)), cfg)
        assert len(result_a) == 1
        assert result_a[0]["rv"] == 2

        # Order-independence: reversed right input must yield the identical winner.
        result_b = PythonDictMergeEngine().merge_asof(left, right_b, Index(("k",)), Index(("k",)), cfg)
        assert len(result_b) == 1
        assert result_b[0]["rv"] == 2

    def test_coerce_z_suffix_utc_strings_join(self) -> None:
        """datetime.fromisoformat on Python 3.10 rejects a trailing 'Z'. The engine must
        normalize 'Z' to '+00:00' before parsing so Z-suffixed UTC strings coerce on every
        supported interpreter. All values are tz-aware, so no mixed-tz error: the join
        succeeds with one row matching the earlier right row (rv == 1.0)."""
        left = [{"k": "a", "t": "2025-06-03T00:00:00Z", "lv": 1}]
        right = [
            {"k": "a", "t": "2025-06-01T00:00:00Z", "rv": 1.0},
            {"k": "a", "t": "2025-06-04T00:00:00Z", "rv": 2.0},
        ]

        engine = PythonDictMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result[0]["rv"] == 1.0

    def test_coerce_mixed_tz_naive_and_aware_raises(self) -> None:
        """Coercion of a column mixing tz-aware and tz-naive ISO strings must raise ValueError:
        datetime.fromisoformat parses both, but naive and aware datetimes are not mutually
        orderable, so the engine must reject the mix explicitly instead of failing later."""
        left = [
            {"k": "a", "t": "2025-06-01T00:00:00+00:00", "lv": 1},
            {"k": "a", "t": "2025-06-01T01:00:00", "lv": 2},
        ]
        right = [{"k": "a", "t": "2025-05-01T00:00:00", "rv": 1.0}]

        engine = PythonDictMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            coerce_time_columns=True,
        )
        with pytest.raises(ValueError):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
