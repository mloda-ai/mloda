"""
Tests for PandasMergeEngine.merge_asof (point-in-time / as-of join).

LEFT-asof semantics: every left row is preserved; for each left row, within the
same equi by-group, the matched right row is chosen by direction, honoring
tolerance and allow_exact_matches. Output columns = all left columns followed by
every right column whose name is not already in left.

The implementation does not exist yet; these tests are expected to FAIL.
"""

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasAsofMergeEngine:
    """Unit tests for PandasMergeEngine.merge_asof."""

    def test_backward_single_by_key(self) -> None:
        """Vector A: backward, single by-key. Right rows shuffled to prove internal sorting."""
        left = pd.DataFrame({"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        # right shuffled on purpose
        right = pd.DataFrame({"k": [2, 1, 1, 2], "t": [30, 5, 18, 5], "rv": [4, 1, 2, 3]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 3
        assert list(result.columns) == ["k", "t", "lv", "rv"]
        result_sorted = result.sort_values(["k", "t"]).reset_index(drop=True)
        assert result_sorted["rv"].tolist() == [1, 2, 3]

    def test_forward_single_by_key(self) -> None:
        """Vector B: forward; row (1,20) has no right_time >= 20 -> null."""
        left = pd.DataFrame({"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        right = pd.DataFrame({"k": [1, 1, 2, 2], "t": [5, 18, 5, 30], "rv": [1, 2, 3, 4]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 3
        assert list(result.columns) == ["k", "t", "lv", "rv"]
        result_sorted = result.sort_values(["k", "t"]).reset_index(drop=True)
        rv = result_sorted["rv"]
        assert rv.iloc[0] == 2
        assert pd.isna(rv.iloc[1])
        assert rv.iloc[2] == 4

    def test_allow_exact_matches_true(self) -> None:
        """Vector C: backward + allow_exact_matches=True -> exact-time row matched (rv=99)."""
        left = pd.DataFrame({"k": [1], "t": [10], "lv": [100]})
        right = pd.DataFrame({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=True
        )
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result["rv"].tolist() == [99]

    def test_allow_exact_matches_false(self) -> None:
        """Vector C: backward + allow_exact_matches=False -> exact excluded, prior row (rv=1)."""
        left = pd.DataFrame({"k": [1], "t": [10], "lv": [100]})
        right = pd.DataFrame({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=False
        )
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result["rv"].tolist() == [1]

    def test_tolerance_numeric(self) -> None:
        """Vector D: backward, tolerance=5 -> row t=100 gap 92 > 5 -> null."""
        left = pd.DataFrame({"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = pd.DataFrame({"k": [1], "t": [8], "rv": [7]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=5)
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 2
        result_sorted = result.sort_values(["k", "t"]).reset_index(drop=True)
        rv = result_sorted["rv"]
        assert rv.iloc[0] == 7
        assert pd.isna(rv.iloc[1])

    def test_tolerance_none(self) -> None:
        """Vector D: backward, tolerance=None -> both rows match (rv=7,7)."""
        left = pd.DataFrame({"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = pd.DataFrame({"k": [1], "t": [8], "rv": [7]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=None)
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 2
        result_sorted = result.sort_values(["k", "t"]).reset_index(drop=True)
        assert result_sorted["rv"].tolist() == [7, 7]

    def test_multi_by_key(self) -> None:
        """Vector E: multi by-key (k1, k2), backward."""
        left = pd.DataFrame({"k1": [1, 1], "k2": ["a", "b"], "t": [10, 10], "lv": [1, 2]})
        right = pd.DataFrame({"k1": [1, 1], "k2": ["a", "b"], "t": [5, 5], "rv": [10, 20]})

        engine = PandasMergeEngine()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k1", "k2")), Index(("k1", "k2")), cfg)

        assert len(result) == 2
        result_sorted = result.sort_values(["k1", "k2"]).reset_index(drop=True)
        row_a = result_sorted[result_sorted["k2"] == "a"].iloc[0]
        row_b = result_sorted[result_sorted["k2"] == "b"].iloc[0]
        assert row_a["rv"] == 10
        assert row_b["rv"] == 20
