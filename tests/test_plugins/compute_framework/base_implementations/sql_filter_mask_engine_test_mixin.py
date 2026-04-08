"""Shared test mixin for SqlBaseFilterMaskEngine implementations.

This mixin verifies that SQL-based filter mask engines return SQL condition
strings instead of Python boolean lists. Each framework-specific test class
should inherit from this mixin and provide:
- engine fixture: Returns the SQL filter mask engine class
- sample_data fixture: Returns framework-specific test data
- evaluate_condition method: Executes a SQL condition against data, returns list[bool]
"""

from abc import abstractmethod
from typing import Any

import pytest

from mloda.core.filter.single_filter import SingleFilter
from mloda.provider import BaseFilterMaskEngine, FeatureSet, FilterMask

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_mask_engine import (
    SqlBaseFilterMaskEngine,
)


def _make_features(
    filters: set[SingleFilter] | None,
    engine: type[BaseFilterMaskEngine],
) -> FeatureSet:
    """Create a FeatureSet with pre-resolved mask engine and filters."""
    fs = FeatureSet()
    fs.filters = filters
    fs.filter_mask_engine = engine
    return fs


class SqlFilterMaskEngineTestMixin:
    """Shared tests for SQL-based filter mask engines.

    Each framework test class must provide:
    - engine fixture returning the engine class
    - sample_data fixture returning data with columns:
        status: ["active", "inactive", "active", "inactive"]
        value: [10, 20, 30, 40]
    - evaluate_condition(data, condition) executing the SQL condition
      and returning list[bool] per row
    """

    @pytest.fixture
    @abstractmethod
    def engine(self) -> type[SqlBaseFilterMaskEngine]:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def sample_data(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluate_condition(self, data: Any, condition: str) -> list[bool]:
        """Execute a SQL condition string against the data and return per-row booleans."""
        raise NotImplementedError

    # -- Unit tests for individual engine methods --

    def test_all_true_returns_string(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.all_true(sample_data)
        assert isinstance(result, str), f"all_true should return a string, got {type(result).__name__}"

    def test_all_true_condition_value(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.all_true(sample_data)
        assert result == "1 = 1"

    def test_equal_returns_condition_string(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.equal(sample_data, "status", "active")
        assert isinstance(result, str), f"equal should return a string, got {type(result).__name__}"
        assert '"status"' in result, "Condition should contain quoted column name"
        assert "'active'" in result, "Condition should contain quoted string value"

    def test_equal_numeric_value(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.equal(sample_data, "value", 10)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "10" in result

    def test_greater_equal_returns_condition_string(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        result = engine.greater_equal(sample_data, "value", 20)
        assert isinstance(result, str)
        assert '"value"' in result
        assert ">=" in result
        assert "20" in result

    def test_less_equal_returns_condition_string(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.less_equal(sample_data, "value", 30)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "<=" in result
        assert "30" in result

    def test_less_than_returns_condition_string(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.less_than(sample_data, "value", 30)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "<" in result
        assert "30" in result

    def test_is_in_returns_condition_string(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        result = engine.is_in(sample_data, "status", ("active", "inactive"))
        assert isinstance(result, str)
        assert '"status"' in result
        assert "IN" in result
        assert "'active'" in result
        assert "'inactive'" in result

    def test_combine_and_joins_conditions(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        cond1 = engine.greater_equal(sample_data, "value", 20)
        cond2 = engine.less_equal(sample_data, "value", 30)
        combined = engine.combine(cond1, cond2)
        assert isinstance(combined, str)
        assert "AND" in combined
        assert cond1 in combined or "20" in combined
        assert cond2 in combined or "30" in combined

    # -- Integration tests using FilterMask.build() --

    def test_build_equal_filter_produces_valid_sql(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        sf = SingleFilter("status", "equal", {"value": "active"})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert isinstance(mask, str), f"FilterMask.build should return a SQL string, got {type(mask).__name__}"
        result = self.evaluate_condition(sample_data, mask)
        assert result == [True, False, True, False]

    def test_build_min_filter_produces_valid_sql(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "min", {"value": 20})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [False, True, True, True]

    def test_build_max_filter_produces_valid_sql(self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any) -> None:
        sf = SingleFilter("value", "max", {"value": 30})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [True, True, True, False]

    def test_build_max_exclusive_filter_produces_valid_sql(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        sf = SingleFilter("value", "max", {"max": 30, "max_exclusive": True})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [True, True, False, False]

    def test_build_range_filter_produces_valid_sql(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        sf = SingleFilter("value", "range", {"min": 15, "max": 35})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [False, True, True, False]

    def test_build_categorical_inclusion_produces_valid_sql(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        sf = SingleFilter("status", "categorical_inclusion", {"values": ("active",)})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [True, False, True, False]

    def test_build_no_matching_filters_returns_all_true_sql(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        sf = SingleFilter("other", "equal", {"value": "x"})
        features = _make_features({sf}, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [True, True, True, True]

    def test_build_none_filters_returns_all_true_sql(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        features = _make_features(None, engine)
        mask = FilterMask.build(sample_data, features, column="status")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [True, True, True, True]

    def test_build_multiple_filters_combines_with_and(
        self, engine: type[SqlBaseFilterMaskEngine], sample_data: Any
    ) -> None:
        sf_min = SingleFilter("value", "min", {"value": 20})
        sf_max = SingleFilter("value", "max", {"value": 30})
        features = _make_features({sf_min, sf_max}, engine)
        mask = FilterMask.build(sample_data, features, column="value")
        assert isinstance(mask, str)
        result = self.evaluate_condition(sample_data, mask)
        assert result == [False, True, True, False]
