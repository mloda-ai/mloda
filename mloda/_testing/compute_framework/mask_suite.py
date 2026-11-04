"""Conformance suites for BaseMaskEngine and SqlBaseMaskEngine implementations, run through a framework adapter."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from mloda._testing.compute_framework.filter_suite import NON_COLLECTION_VALUES
from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.optional_dependency import require

if TYPE_CHECKING:
    import pyarrow as pa

SAMPLE_DATA = TypedTestData(
    {
        "status": (DataType.STRING, ["active", "inactive", "active", "inactive"]),
        "value": (DataType.INT64, [10, 20, 30, 40]),
    }
)

EMPTY_DATA = TypedTestData({"status": (DataType.STRING, []), "value": (DataType.INT64, [])})

NULL_DATA = TypedTestData(
    {
        "status": (DataType.STRING, ["active", None, "inactive", None]),
        "value": (DataType.INT64, [10, 20, 30, 40]),
        "score": (DataType.INT64, [1, None, 3, None]),
        "ratio": (DataType.DOUBLE, [1.0, float("nan"), 3.0, None]),
    }
)

DECIMAL_DATA = TypedTestData({"d": (DataType.DECIMAL, [Decimal("12.34"), Decimal("5.50"), None])})


def _float32_nan() -> Any:
    return pytest.importorskip("numpy").float32("nan")


class MaskEngineSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of the mask engine of the host's compute framework."""

    aspect = "mask"
    required_hooks: ClassVar[tuple[str, ...]] = ("evaluate_mask", "is_boolean_mask", "select_rows")

    @pytest.fixture
    def engine(self, framework: type[ComputeFramework]) -> Any:
        return framework.mask_engine()

    @pytest.fixture
    def sample_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "sample", SAMPLE_DATA)

    @pytest.fixture
    def empty_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "empty", EMPTY_DATA)

    @pytest.fixture
    def null_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "null", NULL_DATA)

    @pytest.fixture
    def decimal_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "decimal", DECIMAL_DATA)

    def arrow_types(self, name: str) -> Mapping[str, pa.DataType] | None:
        if name == "decimal":
            return {"d": require("pyarrow", "building decimal test tables").decimal128(10, 2)}
        return None

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> list[bool]:
        raise NotImplementedError

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> bool:
        """List-based engines can only check element types, so test_all_true_is_boolean covers them on sample_data."""
        raise NotImplementedError

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> Any:
        """The native frame of the rows the mask keeps."""
        raise NotImplementedError

    def _selected(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> dict[str, list[Any]]:
        result = self.select_rows(adapter, mask, data)
        return {name: self.column_values(adapter, result, name) for name in adapter.column_names(result)}

    def test_equal(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.equal(sample_data, "status", "active")
        assert self.evaluate_mask(adapter, mask, sample_data) == [True, False, True, False]

    def test_greater_equal(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.greater_equal(sample_data, "value", 20)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, True, True]

    def test_less_equal(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.less_equal(sample_data, "value", 30)
        assert self.evaluate_mask(adapter, mask, sample_data) == [True, True, True, False]

    def test_less_than(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.less_than(sample_data, "value", 30)
        assert self.evaluate_mask(adapter, mask, sample_data) == [True, True, False, False]

    def test_greater_than(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.greater_than(sample_data, "value", 20)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, False, True, True]

    @pytest.mark.parametrize(
        "values",
        [["active"], ("active",), {"active"}, frozenset({"active"})],
        ids=["list", "tuple", "set", "frozenset"],
    )
    def test_is_in(
        self,
        engine: Any,
        adapter: FrameworkAdapter,
        sample_data: Any,
        values: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> None:
        mask = engine.is_in(sample_data, "status", values)
        assert self.evaluate_mask(adapter, mask, sample_data) == [True, False, True, False]

    @pytest.mark.parametrize("make_values", NON_COLLECTION_VALUES)
    def test_is_in_rejects_non_collection_values(self, engine: Any, sample_data: Any, make_values: Any) -> None:
        with pytest.raises(TypeError, match="list, tuple, set or frozenset"):
            engine.is_in(sample_data, "status", make_values())

    def test_all_true(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.all_true(sample_data)
        assert self.evaluate_mask(adapter, mask, sample_data) == [True, True, True, True]

    def test_combine_and(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 20)
        mask_max = engine.less_equal(sample_data, "value", 30)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, True, False]

    def test_range_via_combine(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 15)
        mask_max = engine.less_equal(sample_data, "value", 35)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, True, False]

    def test_range_exclusive_via_combine(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 15)
        mask_max = engine.less_than(sample_data, "value", 35)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, True, False]

    def test_between_inclusive(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 30)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, True, False]

    def test_between_min_exclusive(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 30, min_exclusive=True)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, False, True, False]

    def test_between_max_exclusive(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 30, max_exclusive=True)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, False, False]

    def test_between_both_exclusive(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 40, min_exclusive=True, max_exclusive=True)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, False, True, False]

    def test_all_of_empty(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.all_of(sample_data, [])
        assert self.evaluate_mask(adapter, mask, sample_data) == [True, True, True, True]

    def test_all_of_single(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        m = engine.greater_equal(sample_data, "value", 20)
        mask = engine.all_of(sample_data, [m])
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, True, True]

    def test_all_of_multi(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        m1 = engine.greater_equal(sample_data, "value", 20)
        m2 = engine.less_equal(sample_data, "value", 30)
        mask = engine.all_of(sample_data, [m1, m2])
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, True, True, False]

    @pytest.mark.parametrize("values", [[], (), set(), frozenset()], ids=["list", "tuple", "set", "frozenset"])
    def test_is_in_empty_values_is_all_false(
        self,
        engine: Any,
        adapter: FrameworkAdapter,
        sample_data: Any,
        values: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> None:
        mask = engine.is_in(sample_data, "status", values)
        assert self.evaluate_mask(adapter, mask, sample_data) == [False, False, False, False]

    @pytest.mark.parametrize("values", [[], (), set(), frozenset()], ids=["list", "tuple", "set", "frozenset"])
    def test_is_in_empty_values_on_empty_data(
        self,
        engine: Any,
        adapter: FrameworkAdapter,
        empty_data: Any,
        values: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> None:
        mask = engine.is_in(empty_data, "status", values)
        assert self.evaluate_mask(adapter, mask, empty_data) == []

    def test_all_true_is_boolean(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        assert self.is_boolean_mask(adapter, engine.all_true(sample_data), sample_data)

    def test_all_true_on_empty_data_is_boolean(self, engine: Any, adapter: FrameworkAdapter, empty_data: Any) -> None:
        mask = engine.all_true(empty_data)
        assert self.is_boolean_mask(adapter, mask, empty_data)
        assert self.evaluate_mask(adapter, mask, empty_data) == []

    def test_combine_on_empty_data(self, engine: Any, adapter: FrameworkAdapter, empty_data: Any) -> None:
        mask = engine.combine(engine.all_true(empty_data), engine.equal(empty_data, "status", "x"))
        assert self.evaluate_mask(adapter, mask, empty_data) == []

    def test_apply_mask_selects_rows(self, engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        mask = engine.equal(sample_data, "status", "active")
        assert self._selected(adapter, mask, sample_data) == {"status": ["active", "active"], "value": [10, 30]}

    def test_apply_all_true_on_empty_data_keeps_columns(
        self, engine: Any, adapter: FrameworkAdapter, empty_data: Any
    ) -> None:
        """Selecting with all_true on zero rows must keep every column."""
        assert self._selected(adapter, engine.all_true(empty_data), empty_data) == {"status": [], "value": []}

    @pytest.mark.parametrize("values", [[Decimal("12.34")], {Decimal("12.34")}], ids=["list", "set"])
    def test_is_in_decimal(
        self,
        engine: Any,
        adapter: FrameworkAdapter,
        decimal_data: Any,
        values: list[Decimal] | set[Decimal],
    ) -> None:
        mask = engine.is_in(decimal_data, "d", values)
        assert self.evaluate_mask(adapter, mask, decimal_data) == [True, False, False]

    def test_is_in_decimal_unrepresentable_values_match_nothing(
        self, engine: Any, adapter: FrameworkAdapter, decimal_data: Any
    ) -> None:
        mask = engine.is_in(decimal_data, "d", [Decimal("12.345"), Decimal("99999999999.99")])
        assert self.evaluate_mask(adapter, mask, decimal_data) == [False, False, False]

    @pytest.mark.parametrize(
        "method,column,arg,expected_mask",
        [
            ("equal", "status", None, [False, True, False, True]),
            ("equal", "score", None, [False, True, False, True]),
            ("equal", "status", "active", [True, False, False, False]),
            ("is_in", "status", ["active"], [True, False, False, False]),
            ("greater_equal", "score", 1, [True, False, True, False]),
            ("less_equal", "score", 3, [True, False, True, False]),
            ("less_than", "score", 3, [True, False, False, False]),
            ("greater_than", "score", 1, [False, False, True, False]),
            ("equal", "ratio", None, [False, True, False, True]),
            ("equal", "ratio", float("nan"), [False, True, False, True]),
            ("equal", "ratio", _float32_nan, [False, True, False, True]),
            ("greater_equal", "ratio", 1.0, [True, False, True, False]),
            ("greater_than", "ratio", 1.0, [False, False, True, False]),
            ("is_in", "ratio", [None], [False, True, False, True]),
            ("is_in", "ratio", [float("nan")], [False, True, False, True]),
            ("is_in", "status", ["active", None], [True, True, False, True]),
        ],
        ids=[
            "equal-status-none",
            "equal-score-none",
            "equal-status-active",
            "is_in-status-active",
            "greater_equal-score-1",
            "less_equal-score-3",
            "less_than-score-3",
            "greater_than-score-1",
            "equal-ratio-none",
            "equal-ratio-nan",
            "equal-ratio-float32-nan",
            "greater_equal-ratio-1",
            "greater_than-ratio-1",
            "is_in-ratio-none",
            "is_in-ratio-nan",
            "is_in-status-active-none",
        ],
    )
    def test_null_row_matches_only_none(
        self,
        engine: Any,
        adapter: FrameworkAdapter,
        null_data: Any,
        method: str,
        column: str,
        arg: Any,
        expected_mask: list[bool],
    ) -> None:
        if arg is _float32_nan:
            arg = arg()
        mask = getattr(engine, method)(null_data, column, arg)
        assert self.is_boolean_mask(adapter, mask, null_data)
        assert self.evaluate_mask(adapter, mask, null_data) == expected_mask
        expected_values = [v for v, keep in zip([10, 20, 30, 40], expected_mask) if keep]
        assert self._selected(adapter, mask, null_data)["value"] == expected_values

    @pytest.mark.parametrize(
        "method,expected_mask",
        [
            ("equal", [False, True, False]),
            ("greater_equal", [True, True, False]),
            ("less_equal", [False, True, False]),
            ("less_than", [False, False, False]),
            ("greater_than", [True, False, False]),
        ],
        ids=["equal", "greater_equal", "less_equal", "less_than", "greater_than"],
    )
    def test_decimal_comparison_null_row_is_false(
        self,
        engine: Any,
        adapter: FrameworkAdapter,
        decimal_data: Any,
        method: str,
        expected_mask: list[bool],
    ) -> None:
        mask = getattr(engine, method)(decimal_data, "d", Decimal("5.50"))
        assert self.is_boolean_mask(adapter, mask, decimal_data)
        assert self.evaluate_mask(adapter, mask, decimal_data) == expected_mask


class SqlMaskEngineSuite(MaskEngineSuite, suite_base=True):
    """SQL-specific tests on top of the shared mask engine tests: condition strings and their structure."""

    required_hooks = ("evaluate_mask", "select_rows")

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> bool:
        """SQL masks are condition strings, so this only checks the string form; override for a real type check."""
        return isinstance(mask, str)

    def test_all_true_returns_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.all_true(sample_data)
        assert isinstance(result, str), f"all_true should return a string, got {type(result).__name__}"

    def test_all_true_condition_value(self, engine: Any, sample_data: Any) -> None:
        result = engine.all_true(sample_data)
        assert result == "1 = 1"

    def test_equal_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.equal(sample_data, "status", "active")
        assert isinstance(result, str), f"equal should return a string, got {type(result).__name__}"
        assert '"status"' in result, "Condition should contain quoted column name"
        assert "'active'" in result, "Condition should contain quoted string value"

    def test_equal_numeric_value(self, engine: Any, sample_data: Any) -> None:
        result = engine.equal(sample_data, "value", 10)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "10" in result

    def test_equal_decimal_value(self, engine: Any, sample_data: Any) -> None:
        result = engine.equal(sample_data, "value", Decimal("12.34"))
        assert result == '"value" = 12.34'

    def test_greater_equal_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.greater_equal(sample_data, "value", 20)
        assert isinstance(result, str)
        assert '"value"' in result
        assert ">=" in result
        assert "20" in result

    def test_less_equal_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.less_equal(sample_data, "value", 30)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "<=" in result
        assert "30" in result

    def test_less_than_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.less_than(sample_data, "value", 30)
        assert isinstance(result, str)
        assert '"value"' in result
        assert "<" in result
        assert "30" in result

    def test_greater_than_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.greater_than(sample_data, "value", 20)
        assert isinstance(result, str)
        assert '"value"' in result
        assert ">" in result
        assert "20" in result

    def test_is_in_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.is_in(sample_data, "status", ("active", "inactive"))
        assert isinstance(result, str)
        assert '"status"' in result
        assert "IN" in result
        assert "'active'" in result
        assert "'inactive'" in result

    def test_is_in_decimal_values(self, engine: Any, sample_data: Any) -> None:
        result = engine.is_in(sample_data, "value", [Decimal("12.34"), Decimal("5.50")])
        assert result == '"value" IN (12.34, 5.50)'

    @pytest.mark.parametrize("bad_column", ["STATUS", "missing"], ids=["case_mismatch", "missing"])
    @pytest.mark.parametrize(
        "primitive, value",
        [
            ("equal", "active"),
            ("greater_equal", 20),
            ("less_equal", 30),
            ("less_than", 30),
            ("greater_than", 20),
            ("is_in", ["active"]),
            ("is_in", []),
        ],
        ids=["equal", "greater_equal", "less_equal", "less_than", "greater_than", "is_in", "is_in_empty"],
    )
    def test_primitive_inexact_column_raises(
        self, engine: Any, sample_data: Any, primitive: str, value: Any, bad_column: str
    ) -> None:
        with pytest.raises(ValueError, match=bad_column):
            getattr(engine, primitive)(sample_data, bad_column, value)

    def test_combine_and_joins_conditions(self, engine: Any, sample_data: Any) -> None:
        cond1 = engine.greater_equal(sample_data, "value", 20)
        cond2 = engine.less_equal(sample_data, "value", 30)
        combined = engine.combine(cond1, cond2)
        assert isinstance(combined, str)
        assert "AND" in combined
        assert cond1 in combined or "20" in combined
        assert cond2 in combined or "30" in combined

    def test_between_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        result = engine.between(sample_data, "value", 20, 30)
        assert isinstance(result, str)
        assert "AND" in result
        assert ">=" in result
        assert "<=" in result
        assert "20" in result
        assert "30" in result

    def test_all_of_returns_condition_string(self, engine: Any, sample_data: Any) -> None:
        m1 = engine.greater_equal(sample_data, "value", 20)
        m2 = engine.less_equal(sample_data, "value", 30)
        result = engine.all_of(sample_data, [m1, m2])
        assert isinstance(result, str)
        assert "AND" in result

    def test_all_of_empty_returns_all_true(self, engine: Any, sample_data: Any) -> None:
        result = engine.all_of(sample_data, [])
        assert result == "1 = 1"

    @pytest.mark.parametrize("values", [[], (), set(), frozenset()], ids=["list", "tuple", "set", "frozenset"])
    def test_is_in_empty_values_condition(
        self,
        engine: Any,
        sample_data: Any,
        values: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> None:
        result = engine.is_in(sample_data, "status", values)
        assert result == "1 = 0"

    def test_is_in_set_values_deterministic_order(self, engine: Any, sample_data: Any) -> None:
        """A mixed-type set has no inherent order; pin that rendering sorts by repr for deterministic SQL."""
        result = engine.is_in(sample_data, "status", {"b", 1, Decimal("2.5"), "a", 3.0})
        assert result == "\"status\" IN ('a', 'b', 1, 3.0, 2.5)"
