"""Shared test mixin for all BaseMaskEngine implementations.

This mixin provides common test methods that verify the mask engine contract.
Each framework-specific test class should inherit from this mixin and provide:
- mask_engine_class attribute: The mask engine class, served by the engine fixture
- sample_data fixture: Returns framework-specific test data
- empty_data fixture: Returns the same schema as sample_data, typed, with zero rows
- null_data fixture: Returns the same schema as sample_data plus a nullable numeric "score"
  column and a float "ratio" column, typed, with null status and score values, and a NaN
  distinct from null in ratio
- evaluate_mask method: Converts framework-specific mask to a Python list of booleans
- is_boolean_mask method: Checks that a mask is boolean-typed
- apply_mask method: Selects rows with the mask the framework's native way, returns column -> values
- decimal_sample_data fixture: Returns a decimal(10, 2) column d with values [12.34, 5.50, null]

A framework that cannot support a test overrides it and skips it with a reason.
"""

from abc import abstractmethod
from decimal import Decimal
from typing import Any, ClassVar

import pytest

from mloda.provider import BaseMaskEngine

# Zero-arg factories (not bare values) so a fresh generator is produced per test run.
NON_COLLECTION_VALUES: list[Any] = [
    pytest.param(lambda: "active", id="str"),
    pytest.param(lambda: b"active", id="bytes"),
    pytest.param(lambda: None, id="none"),
    pytest.param(lambda: 5, id="int"),
    pytest.param(lambda: range(2), id="range"),
    pytest.param(lambda: (v for v in ["active"]), id="generator"),
    pytest.param(lambda: {"active": 1}.keys(), id="dict_keys"),
    pytest.param(lambda: {"active": 1}, id="dict"),
]


class MaskEngineTestMixin:
    """Shared tests for all BaseMaskEngine implementations.

    Each framework test class must provide:
    - mask_engine_class attribute naming the engine class, also read by
      tests/test_plugins/test_mixin_consumer_coverage.py
    - sample_data fixture returning data with columns:
        status: ["active", "inactive", "active", "inactive"]
        value: [10, 20, 30, 40]
    - empty_data fixture returning the same schema (status: string, value: int), typed,
      with zero rows
    - null_data fixture returning the same schema (status: string, value: int) plus a nullable
      numeric score column and a float ratio column, typed, with status: ["active", None,
      "inactive", None], value: [10, 20, 30, 40], score: [1, None, 3, None], and
      ratio: [1.0, NaN, 3.0, None] (NaN distinct from null)
    - evaluate_mask(mask, data) converting the mask to list[bool]
    - is_boolean_mask(mask, data) checking that a mask is boolean-typed
    - apply_mask(mask, data) selecting rows with the mask the framework's native way and
      returning the result as a column -> values dict
    - decimal_sample_data fixture returning a decimal(10, 2) column d: [12.34, 5.50, null]
    """

    mask_engine_class: ClassVar[type[BaseMaskEngine]]

    @pytest.fixture
    def engine(self) -> type[BaseMaskEngine]:
        return self.mask_engine_class

    @pytest.fixture
    @abstractmethod
    def sample_data(self) -> Any:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def empty_data(self) -> Any:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def null_data(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluate_mask(self, mask: Any, data: Any) -> list[bool]:
        raise NotImplementedError

    @abstractmethod
    def is_boolean_mask(self, mask: Any, data: Any) -> bool:
        """List-based engines can only check element types, so test_all_true_is_boolean covers them on sample_data."""
        raise NotImplementedError

    @abstractmethod
    def apply_mask(self, mask: Any, data: Any) -> dict[str, list[Any]]:
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def decimal_sample_data(self) -> Any:
        raise NotImplementedError

    def test_equal(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.equal(sample_data, "status", "active")
        assert self.evaluate_mask(mask, sample_data) == [True, False, True, False]

    def test_greater_equal(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.greater_equal(sample_data, "value", 20)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, True]

    def test_less_equal(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.less_equal(sample_data, "value", 30)
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, False]

    def test_less_than(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.less_than(sample_data, "value", 30)
        assert self.evaluate_mask(mask, sample_data) == [True, True, False, False]

    def test_greater_than(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.greater_than(sample_data, "value", 20)
        assert self.evaluate_mask(mask, sample_data) == [False, False, True, True]

    @pytest.mark.parametrize(
        "values",
        [["active"], ("active",), {"active"}, frozenset({"active"})],
        ids=["list", "tuple", "set", "frozenset"],
    )
    def test_is_in(
        self,
        engine: type[BaseMaskEngine],
        sample_data: Any,
        values: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> None:
        mask = engine.is_in(sample_data, "status", values)
        assert self.evaluate_mask(mask, sample_data) == [True, False, True, False]

    @pytest.mark.parametrize("make_values", NON_COLLECTION_VALUES)
    def test_is_in_rejects_non_collection_values(
        self, engine: type[BaseMaskEngine], sample_data: Any, make_values: Any
    ) -> None:
        with pytest.raises(TypeError, match="list, tuple, set or frozenset"):
            engine.is_in(sample_data, "status", make_values())

    def test_all_true(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.all_true(sample_data)
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, True]

    def test_combine_and(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 20)
        mask_max = engine.less_equal(sample_data, "value", 30)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_range_via_combine(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 15)
        mask_max = engine.less_equal(sample_data, "value", 35)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_range_exclusive_via_combine(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask_min = engine.greater_equal(sample_data, "value", 15)
        mask_max = engine.less_than(sample_data, "value", 35)
        mask = engine.combine(mask_min, mask_max)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_between_inclusive(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 30)
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    def test_between_min_exclusive(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 30, min_exclusive=True)
        assert self.evaluate_mask(mask, sample_data) == [False, False, True, False]

    def test_between_max_exclusive(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 30, max_exclusive=True)
        assert self.evaluate_mask(mask, sample_data) == [False, True, False, False]

    def test_between_both_exclusive(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.between(sample_data, "value", 20, 40, min_exclusive=True, max_exclusive=True)
        assert self.evaluate_mask(mask, sample_data) == [False, False, True, False]

    def test_all_of_empty(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.all_of(sample_data, [])
        assert self.evaluate_mask(mask, sample_data) == [True, True, True, True]

    def test_all_of_single(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        m = engine.greater_equal(sample_data, "value", 20)
        mask = engine.all_of(sample_data, [m])
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, True]

    def test_all_of_multi(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        m1 = engine.greater_equal(sample_data, "value", 20)
        m2 = engine.less_equal(sample_data, "value", 30)
        mask = engine.all_of(sample_data, [m1, m2])
        assert self.evaluate_mask(mask, sample_data) == [False, True, True, False]

    @pytest.mark.parametrize("values", [[], (), set(), frozenset()], ids=["list", "tuple", "set", "frozenset"])
    def test_is_in_empty_values_is_all_false(
        self,
        engine: type[BaseMaskEngine],
        sample_data: Any,
        values: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> None:
        mask = engine.is_in(sample_data, "status", values)
        assert self.evaluate_mask(mask, sample_data) == [False, False, False, False]

    @pytest.mark.parametrize("values", [[], (), set(), frozenset()], ids=["list", "tuple", "set", "frozenset"])
    def test_is_in_empty_values_on_empty_data(
        self,
        engine: type[BaseMaskEngine],
        empty_data: Any,
        values: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> None:
        mask = engine.is_in(empty_data, "status", values)
        assert self.evaluate_mask(mask, empty_data) == []

    def test_all_true_is_boolean(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        assert self.is_boolean_mask(engine.all_true(sample_data), sample_data)

    def test_all_true_on_empty_data_is_boolean(self, engine: type[BaseMaskEngine], empty_data: Any) -> None:
        mask = engine.all_true(empty_data)
        assert self.is_boolean_mask(mask, empty_data)
        assert self.evaluate_mask(mask, empty_data) == []

    def test_combine_on_empty_data(self, engine: type[BaseMaskEngine], empty_data: Any) -> None:
        mask = engine.combine(engine.all_true(empty_data), engine.equal(empty_data, "status", "x"))
        assert self.evaluate_mask(mask, empty_data) == []

    def test_apply_mask_selects_rows(self, engine: type[BaseMaskEngine], sample_data: Any) -> None:
        mask = engine.equal(sample_data, "status", "active")
        assert self.apply_mask(mask, sample_data) == {"status": ["active", "active"], "value": [10, 30]}

    def test_apply_all_true_on_empty_data_keeps_columns(self, engine: type[BaseMaskEngine], empty_data: Any) -> None:
        """Pins #1535: selecting with all_true on zero rows must keep every column."""
        assert self.apply_mask(engine.all_true(empty_data), empty_data) == {"status": [], "value": []}

    @pytest.mark.parametrize("values", [[Decimal("12.34")], {Decimal("12.34")}], ids=["list", "set"])
    def test_is_in_decimal(
        self, engine: type[BaseMaskEngine], decimal_sample_data: Any, values: list[Decimal] | set[Decimal]
    ) -> None:
        mask = engine.is_in(decimal_sample_data, "d", values)
        assert self.evaluate_mask(mask, decimal_sample_data) == [True, False, False]

    def test_is_in_decimal_unrepresentable_values_match_nothing(
        self, engine: type[BaseMaskEngine], decimal_sample_data: Any
    ) -> None:
        mask = engine.is_in(decimal_sample_data, "d", [Decimal("12.345"), Decimal("99999999999.99")])
        assert self.evaluate_mask(mask, decimal_sample_data) == [False, False, False]

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
            "greater_equal-ratio-1",
            "greater_than-ratio-1",
            "is_in-ratio-none",
            "is_in-ratio-nan",
            "is_in-status-active-none",
        ],
    )
    def test_null_row_matches_only_none(
        self,
        engine: type[BaseMaskEngine],
        null_data: Any,
        method: str,
        column: str,
        arg: Any,
        expected_mask: list[bool],
    ) -> None:
        mask = getattr(engine, method)(null_data, column, arg)
        assert self.is_boolean_mask(mask, null_data)
        assert self.evaluate_mask(mask, null_data) == expected_mask
        expected_values = [v for v, keep in zip([10, 20, 30, 40], expected_mask) if keep]
        assert self.apply_mask(mask, null_data)["value"] == expected_values

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
        engine: type[BaseMaskEngine],
        decimal_sample_data: Any,
        method: str,
        expected_mask: list[bool],
    ) -> None:
        mask = getattr(engine, method)(decimal_sample_data, "d", Decimal("5.50"))
        assert self.is_boolean_mask(mask, decimal_sample_data)
        assert self.evaluate_mask(mask, decimal_sample_data) == expected_mask
