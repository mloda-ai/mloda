"""
Tests for DuckDBMergeEngine.merge_asof (point-in-time / as-of join).

Uses the `connection` fixture from the duckdb conftest. Builds DuckdbRelation
from arrow tables and inspects results via `.df()`.

The implementation does not exist yet; these tests are expected to FAIL.
"""

from typing import Any

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
    import pandas as pd
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]
    pd = None


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBAsofMergeEngine:
    """Unit tests for DuckDBMergeEngine.merge_asof."""

    def _rel(self, connection: Any, data: dict[str, Any]) -> Any:
        return DuckdbRelation.from_arrow(connection, pa.Table.from_pydict(data))

    def test_backward_single_by_key(self, connection: Any) -> None:
        """Vector A: backward, single by-key."""
        left = self._rel(connection, {"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        right = self._rel(connection, {"k": [1, 1, 2, 2], "t": [5, 18, 5, 30], "rv": [1, 2, 3, 4]})

        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_df = result.df()
        assert len(result_df) == 3
        assert list(result_df.columns) == ["k", "t", "lv", "rv"]
        result_sorted = result_df.sort_values(["k", "t"]).reset_index(drop=True)
        assert result_sorted["rv"].tolist() == [1, 2, 3]

    def test_forward_single_by_key(self, connection: Any) -> None:
        """Vector B: forward; row (1,20) has no right_time >= 20 -> null."""
        left = self._rel(connection, {"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]})
        right = self._rel(connection, {"k": [1, 1, 2, 2], "t": [5, 18, 5, 30], "rv": [1, 2, 3, 4]})

        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_df = result.df()
        assert len(result_df) == 3
        result_sorted = result_df.sort_values(["k", "t"]).reset_index(drop=True)
        rv = result_sorted["rv"]
        assert rv.iloc[0] == 2
        assert pd.isna(rv.iloc[1])
        assert rv.iloc[2] == 4

    def test_allow_exact_matches_true(self, connection: Any) -> None:
        """Vector C: backward + allow_exact_matches=True -> rv=99."""
        left = self._rel(connection, {"k": [1], "t": [10], "lv": [100]})
        right = self._rel(connection, {"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=True
        )
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["rv"].tolist() == [99]

    def test_allow_exact_matches_false(self, connection: Any) -> None:
        """Vector C: backward + allow_exact_matches=False -> rv=1."""
        left = self._rel(connection, {"k": [1], "t": [10], "lv": [100]})
        right = self._rel(connection, {"k": [1, 1], "t": [10, 5], "rv": [99, 1]})

        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(
            left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=False
        )
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["rv"].tolist() == [1]

    def test_tolerance_numeric(self, connection: Any) -> None:
        """Vector D: backward, tolerance=5 -> row t=100 gap 92 > 5 -> null."""
        left = self._rel(connection, {"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = self._rel(connection, {"k": [1], "t": [8], "rv": [7]})

        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=5)
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_df = result.df()
        assert len(result_df) == 2
        result_sorted = result_df.sort_values(["k", "t"]).reset_index(drop=True)
        rv = result_sorted["rv"]
        assert rv.iloc[0] == 7
        assert pd.isna(rv.iloc[1])

    def test_tolerance_none(self, connection: Any) -> None:
        """Vector D: backward, tolerance=None -> both rows match (rv=7,7)."""
        left = self._rel(connection, {"k": [1, 1], "t": [10, 100], "lv": [1, 2]})
        right = self._rel(connection, {"k": [1], "t": [8], "rv": [7]})

        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=None)
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)

        result_df = result.df()
        assert len(result_df) == 2
        result_sorted = result_df.sort_values(["k", "t"]).reset_index(drop=True)
        assert result_sorted["rv"].tolist() == [7, 7]

    def test_nearest_raises_value_error(self, connection: Any) -> None:
        """Vector F: DuckDB native ASOF cannot express 'nearest' in v1 -> ValueError."""
        left = self._rel(connection, {"k": [1], "t": [10], "lv": [100]})
        right = self._rel(connection, {"k": [1], "t": [8], "rv": [7]})

        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest")
        with pytest.raises(ValueError):
            engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
