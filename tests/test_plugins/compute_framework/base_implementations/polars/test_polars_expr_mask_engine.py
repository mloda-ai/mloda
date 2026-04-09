"""Tests for PolarsExprMaskEngine -- returns pl.Expr for LazyFrame pipelines.

This file adds Polars-expr-specific unit tests that verify pl.Expr return types
and a LazyFrame end-to-end test.
"""

from typing import Any

import pytest

from mloda.provider import BaseMaskEngine
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

try:
    from mloda_plugins.compute_framework.base_implementations.polars.polars_expr_mask_engine import (
        PolarsExprMaskEngine,
    )
except ImportError:
    PolarsExprMaskEngine = None  # type: ignore[assignment, misc]


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsExprMaskEngine(MaskEngineTestMixin):
    """Tests for the PolarsExprMaskEngine that returns pl.Expr objects."""

    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return PolarsExprMaskEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pl.LazyFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        result: list[bool] = data.select(mask.alias("__mask")).collect()["__mask"].to_list()
        return result

    # -- Polars-expr-specific tests --

    def test_supported_data_type(self, engine: type[BaseMaskEngine]) -> None:
        assert engine.supported_data_type() is pl.LazyFrame

    def test_all_methods_return_expr(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        """Verify every engine method returns pl.Expr, not pl.Series or list."""
        assert isinstance(engine.all_true(sample_data), pl.Expr)
        assert isinstance(engine.equal(sample_data, "status", "active"), pl.Expr)
        assert isinstance(engine.greater_equal(sample_data, "value", 20), pl.Expr)
        assert isinstance(engine.less_equal(sample_data, "value", 30), pl.Expr)
        assert isinstance(engine.less_than(sample_data, "value", 30), pl.Expr)
        assert isinstance(engine.is_in(sample_data, "status", ("active",)), pl.Expr)
        combined = engine.combine(
            engine.greater_equal(sample_data, "value", 20),
            engine.less_equal(sample_data, "value", 30),
        )
        assert isinstance(combined, pl.Expr)

    def test_works_with_lazy_frame(self, engine: type[BaseMaskEngine]) -> None:
        """Verify end-to-end: build mask from LazyFrame, filter, collect."""
        lf = pl.LazyFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )
        mask = engine.equal(lf, "status", "active")
        result = lf.filter(mask).collect()
        assert result["status"].to_list() == ["active", "active"]
        assert result["value"].to_list() == [10, 30]


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsLazyDataFrameMaskEngine:
    """Verify that PolarsLazyDataFrame.mask_engine() returns PolarsExprMaskEngine."""

    def test_lazy_dataframe_uses_expr_engine(self) -> None:
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        engine = PolarsLazyDataFrame.mask_engine()
        assert engine is PolarsExprMaskEngine
