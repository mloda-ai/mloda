"""Shared test mixin for SqlBaseMaskEngine implementations.

Extends MaskEngineTestMixin with SQL-specific unit tests that verify
condition strings and their structure.

Each framework-specific test class should inherit from this mixin and provide:
- engine fixture: Returns the SQL mask engine class
- sample_data fixture: Returns framework-specific test data
- evaluate_mask method: Executes a SQL condition against data, returns list[bool]
"""

from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_mask_engine import (
    SqlBaseMaskEngine,
)
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    MaskEngineTestMixin,
)


class SqlMaskEngineTestMixin(MaskEngineTestMixin):
    """SQL-specific tests on top of the shared mask engine tests.

    These unit tests verify that individual engine methods return SQL condition
    strings with the expected structure.
    """

    def test_all_true_returns_string(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.all_true(sample_data)
        assert isinstance(result, str), f"all_true should return a string, got {type(result).__name__}"

    def test_all_true_condition_value(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.all_true(sample_data)
        assert result == "1 = 1"

    def test_equal_returns_condition_string(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.equal(sample_data, "status", "active")
        assert isinstance(result, str), f"equal should return a string, got {type(result).__name__}"
        assert '"status"' in result, "Condition should contain quoted column name"
        assert "'active'" in result, "Condition should contain quoted string value"

    def test_equal_numeric_value(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.equal(sample_data, "value", 10)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "10" in result

    def test_greater_equal_returns_condition_string(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.greater_equal(sample_data, "value", 20)
        assert isinstance(result, str)
        assert '"value"' in result
        assert ">=" in result
        assert "20" in result

    def test_less_equal_returns_condition_string(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.less_equal(sample_data, "value", 30)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "<=" in result
        assert "30" in result

    def test_less_than_returns_condition_string(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.less_than(sample_data, "value", 30)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "<" in result
        assert "30" in result

    def test_is_in_returns_condition_string(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        result = engine.is_in(sample_data, "status", ("active", "inactive"))
        assert isinstance(result, str)
        assert '"status"' in result
        assert "IN" in result
        assert "'active'" in result
        assert "'inactive'" in result

    def test_combine_and_joins_conditions(self, engine: type[SqlBaseMaskEngine], sample_data: Any) -> None:
        cond1 = engine.greater_equal(sample_data, "value", 20)
        cond2 = engine.less_equal(sample_data, "value", 30)
        combined = engine.combine(cond1, cond2)
        assert isinstance(combined, str)
        assert "AND" in combined
        assert cond1 in combined or "20" in combined
        assert cond2 in combined or "30" in combined
