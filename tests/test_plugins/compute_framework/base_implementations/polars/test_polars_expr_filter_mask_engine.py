"""Tests for PolarsExprFilterMaskEngine -- returns pl.Expr for LazyFrame pipelines.

These tests verify that the engine returns pl.Expr objects (not pl.Series) and that
evaluating those expressions against a LazyFrame produces the correct boolean results.
"""

from typing import Any

import pytest

from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import FeatureSet, FilterMask, BaseFilterMaskEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

try:
    from mloda_plugins.compute_framework.base_implementations.polars.polars_expr_filter_mask_engine import (
        PolarsExprFilterMaskEngine,
    )
except ImportError:
    PolarsExprFilterMaskEngine = None  # type: ignore[assignment, misc]


def _make_features(
    filters: set[SingleFilter] | None,
    engine: type[BaseFilterMaskEngine],
) -> FeatureSet:
    """Create a FeatureSet with pre-resolved mask engine and filters."""
    fs = FeatureSet()
    fs.filters = filters
    fs.filter_mask_engine = engine
    return fs


def _eval_mask(mask: Any, data: Any) -> list[bool]:
    """Evaluate a pl.Expr mask against a LazyFrame and return a list of booleans."""
    result: list[bool] = data.select(mask.alias("__mask")).collect()["__mask"].to_list()
    return result


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsExprFilterMaskEngine:
    """Tests for the PolarsExprFilterMaskEngine that returns pl.Expr objects."""

    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return PolarsExprFilterMaskEngine

    @pytest.fixture
    def sample_data(self) -> pl.LazyFrame:
        return pl.LazyFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )

    def test_supported_data_type(self, engine: type[BaseFilterMaskEngine]) -> None:
        assert engine.supported_data_type() is pl.LazyFrame

    def test_returns_expr_type(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        """The engine must return pl.Expr, not pl.Series."""
        mask = engine.all_true(sample_data)
        assert isinstance(mask, pl.Expr)

    def test_all_true_returns_expr(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        mask = engine.all_true(sample_data)
        assert isinstance(mask, pl.Expr)
        result = _eval_mask(mask, sample_data)
        assert result == [True, True, True, True]

    def test_equal_returns_expr(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        mask = engine.equal(sample_data, "status", "active")
        assert isinstance(mask, pl.Expr)
        result = _eval_mask(mask, sample_data)
        assert result == [True, False, True, False]

    def test_greater_equal_returns_expr(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        mask = engine.greater_equal(sample_data, "value", 20)
        assert isinstance(mask, pl.Expr)
        result = _eval_mask(mask, sample_data)
        assert result == [False, True, True, True]

    def test_less_equal_returns_expr(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        mask = engine.less_equal(sample_data, "value", 30)
        assert isinstance(mask, pl.Expr)
        result = _eval_mask(mask, sample_data)
        assert result == [True, True, True, False]

    def test_less_than_returns_expr(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        mask = engine.less_than(sample_data, "value", 30)
        assert isinstance(mask, pl.Expr)
        result = _eval_mask(mask, sample_data)
        assert result == [True, True, False, False]

    def test_is_in_returns_expr(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        mask = engine.is_in(sample_data, "status", ("active",))
        assert isinstance(mask, pl.Expr)
        result = _eval_mask(mask, sample_data)
        assert result == [True, False, True, False]

    def test_combine_returns_expr(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        mask1 = engine.greater_equal(sample_data, "value", 20)
        mask2 = engine.less_equal(sample_data, "value", 30)
        combined = engine.combine(mask1, mask2)
        assert isinstance(combined, pl.Expr)
        result = _eval_mask(combined, sample_data)
        assert result == [False, True, True, False]

    # -- FilterMask.build integration tests --

    def test_equal_filter(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf = SingleFilter("status", "equal", {"value": "active"})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [True, False, True, False]

    def test_min_filter(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf = SingleFilter("value", "min", {"value": 20})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [False, True, True, True]

    def test_max_filter_simple(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf = SingleFilter("value", "max", {"value": 30})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [True, True, True, False]

    def test_max_filter_exclusive(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf = SingleFilter("value", "max", {"max": 30, "max_exclusive": True})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [True, True, False, False]

    def test_range_filter(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf = SingleFilter("value", "range", {"min": 15, "max": 35})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [False, True, True, False]

    def test_categorical_inclusion(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf = SingleFilter("status", "categorical_inclusion", {"values": ("active",)})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [True, False, True, False]

    def test_no_matching_filters(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf = SingleFilter("other", "equal", {"value": "x"})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [True, True, True, True]

    def test_multiple_filters_same_column(self, engine: type[BaseFilterMaskEngine], sample_data: pl.LazyFrame) -> None:
        sf_min = SingleFilter("value", "min", {"value": 20})
        sf_max = SingleFilter("value", "max", {"value": 30})
        features = _make_features({sf_min, sf_max}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, pl.Expr)
        assert _eval_mask(mask, sample_data) == [False, True, True, False]

    def test_works_with_lazy_frame(self, engine: type[BaseFilterMaskEngine]) -> None:
        """Verify end-to-end: build mask from LazyFrame, filter, collect."""
        lf = pl.LazyFrame(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )
        sf = SingleFilter("status", "equal", {"value": "active"})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(lf, features, column="status")
        # Use the mask to filter the LazyFrame and collect
        result = lf.filter(mask).collect()
        assert result["status"].to_list() == ["active", "active"]
        assert result["value"].to_list() == [10, 30]


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsLazyDataFrameFilterMaskEngine:
    """Verify that PolarsLazyDataFrame.filter_mask_engine() returns PolarsExprFilterMaskEngine."""

    def test_lazy_dataframe_uses_expr_engine(self) -> None:
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        engine = PolarsLazyDataFrame.filter_mask_engine()
        assert engine is PolarsExprFilterMaskEngine
