"""
Cross-backend equality tests for merge_asof.

Pandas is the reference. DuckDBMergeEngine and PolarsLazyMergeEngine outputs are
normalized to pandas DataFrames and compared against the reference for schema
(set of column names) and values (sorted by all by/time columns, nulls
normalized to a common sentinel).

This schema+value equality across backends is the most important contract. The
pandas reference itself is also asserted against the hardcoded expected contract
so that a uniformly-wrong implementation cannot pass.

The implementation does not exist yet; these tests are expected to FAIL.
"""

from typing import Any

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import polars as pl
    import duckdb
    import pyarrow as pa

    from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine
    from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import (
        PolarsLazyMergeEngine,
    )
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

    DEPS_AVAILABLE = True
except ImportError:
    logger.warning("pandas/polars/duckdb/pyarrow not all installed. Cross-backend tests will be skipped.")
    pd = None
    pl = None  # type: ignore
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]
    DEPS_AVAILABLE = False

_SENTINEL = -999999


def _normalize(df: Any, sort_cols: list[str]) -> Any:
    """Sort by sort_cols, reset index, fill NaN/None with a common sentinel."""
    out = df.sort_values(sort_cols).reset_index(drop=True)
    return out.where(out.notna(), _SENTINEL)


@pytest.mark.skipif(not DEPS_AVAILABLE, reason="pandas/polars/duckdb/pyarrow not all installed.")
class TestAsofCrossBackend:
    """Compare DuckDB and PolarsLazy against the pandas reference."""

    def _pandas_result(
        self, left: dict[str, Any], right: dict[str, Any], left_idx: Any, right_idx: Any, cfg: Any
    ) -> Any:
        engine = PandasMergeEngine()
        return engine.merge_asof(pd.DataFrame(left), pd.DataFrame(right), left_idx, right_idx, cfg)

    def _duckdb_result(
        self, left: dict[str, Any], right: dict[str, Any], left_idx: Any, right_idx: Any, cfg: Any
    ) -> Any:
        conn = duckdb.connect()
        left_rel = DuckdbRelation.from_arrow(conn, pa.Table.from_pydict(left))
        right_rel = DuckdbRelation.from_arrow(conn, pa.Table.from_pydict(right))
        engine = DuckDBMergeEngine(conn)
        return engine.merge_asof(left_rel, right_rel, left_idx, right_idx, cfg).df()

    def _polars_lazy_result(
        self, left: dict[str, Any], right: dict[str, Any], left_idx: Any, right_idx: Any, cfg: Any
    ) -> Any:
        engine = PolarsLazyMergeEngine()
        out = engine.merge_asof(pl.DataFrame(left).lazy(), pl.DataFrame(right).lazy(), left_idx, right_idx, cfg)
        return out.collect().to_pandas()

    def test_scenario_a_shared_names_with_unmatched(self) -> None:
        """Test A: shared by-key + shared time col names, with an unmatched left row (2,15)."""
        left = {"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]}
        right = {"k": [1, 1], "t": [5, 18], "rv": [1, 2]}
        left_idx = Index(("k",))
        right_idx = Index(("k",))
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        sort_cols = ["k", "t"]

        pandas_df = self._pandas_result(left, right, left_idx, right_idx, cfg)

        # Hardcoded contract for the reference.
        assert list(pandas_df.columns) == ["k", "t", "lv", "rv"]
        ref_sorted = pandas_df.sort_values(sort_cols).reset_index(drop=True)
        rv = ref_sorted["rv"]
        assert rv.iloc[0] == 1
        assert rv.iloc[1] == 2
        assert pd.isna(rv.iloc[2])

        reference = _normalize(pandas_df, sort_cols)

        for name, getter in (
            ("duckdb", self._duckdb_result),
            ("polars_lazy", self._polars_lazy_result),
        ):
            backend_df = getter(left, right, left_idx, right_idx, cfg)
            assert set(backend_df.columns) == set(reference.columns), name
            backend_norm = _normalize(backend_df, sort_cols)[list(reference.columns)]
            pd.testing.assert_frame_equal(backend_norm, reference, check_dtype=False, check_like=False, obj=name)

    def test_scenario_b_differing_names_all_matched(self) -> None:
        """Test B: differing by-key names AND differing time-col names, all matched."""
        left = {"lk": [1, 1, 2], "lt": [10, 20, 15], "lv": [100, 200, 300]}
        right = {"rk": [1, 1, 2, 2], "rt": [5, 18, 5, 30], "rv": [1, 2, 3, 4]}
        left_idx = Index(("lk",))
        right_idx = Index(("rk",))
        cfg = AsOfJoinConfig(left_time_column="lt", right_time_column="rt", direction="backward")
        sort_cols = ["lk", "lt"]

        pandas_df = self._pandas_result(left, right, left_idx, right_idx, cfg)

        # Hardcoded contract: right by-key rk IS present since names differ.
        assert list(pandas_df.columns) == ["lk", "lt", "lv", "rk", "rt", "rv"]
        assert len(pandas_df) == 3
        ref_sorted = pandas_df.sort_values(sort_cols).reset_index(drop=True)
        assert ref_sorted["rv"].notna().all()

        reference = _normalize(pandas_df, sort_cols)

        for name, getter in (
            ("duckdb", self._duckdb_result),
            ("polars_lazy", self._polars_lazy_result),
        ):
            backend_df = getter(left, right, left_idx, right_idx, cfg)
            assert set(backend_df.columns) == set(reference.columns), name
            backend_norm = _normalize(backend_df, sort_cols)[list(reference.columns)]
            pd.testing.assert_frame_equal(backend_norm, reference, check_dtype=False, check_like=False, obj=name)
