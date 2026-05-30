"""
Tests for PolarsMergeEngine / PolarsLazyMergeEngine merge_asof (as-of join).

Two classes: one for the eager PolarsMergeEngine (pl.DataFrame), one for the
lazy PolarsLazyMergeEngine (pl.LazyFrame; result is collected before asserting).

The implementation does not exist yet; these tests are expected to FAIL.
"""

from typing import Any

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsAsofMergeEngine:
    """Unit tests for the eager PolarsMergeEngine.merge_asof."""

    def _result(self, left: Any, right: Any, left_idx: Any, right_idx: Any, cfg: Any) -> Any:
        engine = PolarsMergeEngine()
        return engine.merge_asof(left, right, left_idx, right_idx, cfg)

    def test_backward_single_by_key(self) -> None:
        """Vector A: backward, single by-key."""
        left = pl.DataFrame({"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        right = pl.DataFrame({"k": [1, 1, 2, 2], "t": [5, 18, 5, 30], "rv": [1, 2, 3, 4]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 3
        assert result.columns == ["k", "t", "lv", "rv"]
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [1, 2, 3]

    def test_forward_single_by_key(self) -> None:
        """Vector B: forward; row (1,20) has no right_time >= 20 -> null."""
        left = pl.DataFrame({"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        right = pl.DataFrame({"k": [1, 1, 2, 2], "t": [5, 18, 5, 30], "rv": [1, 2, 3, 4]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward")
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 3
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [2, None, 4]

    def test_allow_exact_matches_true(self) -> None:
        """Vector C: backward + allow_exact_matches=True -> rv=99."""
        left = pl.DataFrame({"k": [1], "t": [10], "lv": [100]})
        right = pl.DataFrame({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=True
        )
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result["rv"].to_list() == [99]

    def test_allow_exact_matches_false(self) -> None:
        """Vector C: backward + allow_exact_matches=False -> rv=1."""
        left = pl.DataFrame({"k": [1], "t": [10], "lv": [100]})
        right = pl.DataFrame({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=False
        )
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result["rv"].to_list() == [1]

    def test_tolerance_numeric(self) -> None:
        """Vector D: backward, tolerance=5 -> row t=100 gap 92 > 5 -> null."""
        left = pl.DataFrame({"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = pl.DataFrame({"k": [1], "t": [8], "rv": [7]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=5)
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 2
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [7, None]

    def test_tolerance_none(self) -> None:
        """Vector D: backward, tolerance=None -> both rows match (rv=7,7)."""
        left = pl.DataFrame({"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = pl.DataFrame({"k": [1], "t": [8], "rv": [7]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=None)
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 2
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [7, 7]

    def test_multi_by_key(self) -> None:
        """Vector E: multi by-key (k1, k2), backward."""
        left = pl.DataFrame({"k1": [1, 1], "k2": ["a", "b"], "t": [10, 10], "lv": [1, 2]})
        right = pl.DataFrame({"k1": [1, 1], "k2": ["a", "b"], "t": [5, 5], "rv": [10, 20]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = self._result(left, right, Index(("k1", "k2")), Index(("k1", "k2")), cfg)

        assert len(result) == 2
        result_sorted = result.sort(["k1", "k2"])
        rv_by_k2 = dict(zip(result_sorted["k2"].to_list(), result_sorted["rv"].to_list()))
        assert rv_by_k2["a"] == 10
        assert rv_by_k2["b"] == 20


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyAsofMergeEngine:
    """Unit tests for the lazy PolarsLazyMergeEngine.merge_asof."""

    def _result(self, left: Any, right: Any, left_idx: Any, right_idx: Any, cfg: Any) -> Any:
        engine = PolarsLazyMergeEngine()
        out = engine.merge_asof(left.lazy(), right.lazy(), left_idx, right_idx, cfg)
        return out.collect()

    def test_backward_single_by_key(self) -> None:
        """Vector A: backward, single by-key."""
        left = pl.DataFrame({"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        right = pl.DataFrame({"k": [1, 1, 2, 2], "t": [5, 18, 5, 30], "rv": [1, 2, 3, 4]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 3
        assert result.columns == ["k", "t", "lv", "rv"]
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [1, 2, 3]

    def test_forward_single_by_key(self) -> None:
        """Vector B: forward; row (1,20) has no right_time >= 20 -> null."""
        left = pl.DataFrame({"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        right = pl.DataFrame({"k": [1, 1, 2, 2], "t": [5, 18, 5, 30], "rv": [1, 2, 3, 4]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward")
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 3
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [2, None, 4]

    def test_allow_exact_matches_true(self) -> None:
        """Vector C: backward + allow_exact_matches=True -> rv=99."""
        left = pl.DataFrame({"k": [1], "t": [10], "lv": [100]})
        right = pl.DataFrame({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=True
        )
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result["rv"].to_list() == [99]

    def test_allow_exact_matches_false(self) -> None:
        """Vector C: backward + allow_exact_matches=False -> rv=1."""
        left = pl.DataFrame({"k": [1], "t": [10], "lv": [100]})
        right = pl.DataFrame({"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=False
        )
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 1
        assert result["rv"].to_list() == [1]

    def test_tolerance_numeric(self) -> None:
        """Vector D: backward, tolerance=5 -> row t=100 gap 92 > 5 -> null."""
        left = pl.DataFrame({"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = pl.DataFrame({"k": [1], "t": [8], "rv": [7]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=5)
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 2
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [7, None]

    def test_tolerance_none(self) -> None:
        """Vector D: backward, tolerance=None -> both rows match (rv=7,7)."""
        left = pl.DataFrame({"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = pl.DataFrame({"k": [1], "t": [8], "rv": [7]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=None)
        result = self._result(left, right, Index(("k",)), Index(("k",)), cfg)

        assert len(result) == 2
        result_sorted = result.sort(["k", "t"])
        assert result_sorted["rv"].to_list() == [7, 7]

    def test_multi_by_key(self) -> None:
        """Vector E: multi by-key (k1, k2), backward."""
        left = pl.DataFrame({"k1": [1, 1], "k2": ["a", "b"], "t": [10, 10], "lv": [1, 2]})
        right = pl.DataFrame({"k1": [1, 1], "k2": ["a", "b"], "t": [5, 5], "rv": [10, 20]})

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = self._result(left, right, Index(("k1", "k2")), Index(("k1", "k2")), cfg)

        assert len(result) == 2
        result_sorted = result.sort(["k1", "k2"])
        rv_by_k2 = dict(zip(result_sorted["k2"].to_list(), result_sorted["rv"].to_list()))
        assert rv_by_k2["a"] == 10
        assert rv_by_k2["b"] == 20
