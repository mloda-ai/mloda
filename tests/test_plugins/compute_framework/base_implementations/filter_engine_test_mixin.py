"""Shared filter engine tests for BaseFilterEngine implementations.

Consumers set `filter_engine_class`, also read by `tests/test_plugins/test_mixin_consumer_coverage.py`, and
implement the abstract fixtures and methods their tests need, including `decimal_sample_data` and
`get_decimal_column_dtype`. A framework that cannot support a test overrides it and skips it with a reason.
"""

from abc import abstractmethod
from decimal import Decimal
from typing import Any, ClassVar

import pytest

from mloda.provider import BaseFilterEngine
from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from tests.test_plugins.compute_framework.base_implementations.mask_engine_test_mixin import (
    NON_COLLECTION_VALUES,
)


class FilterEngineTestMixin:
    """Shared tests for all BaseFilterEngine implementations."""

    filter_engine_class: ClassVar[type[BaseFilterEngine]]

    @pytest.fixture
    def filter_engine(self) -> type[BaseFilterEngine]:
        return self.filter_engine_class

    @pytest.fixture
    @abstractmethod
    def sample_data(self) -> Any:
        """Return framework-specific sample data.

        Override in framework-specific test class.
        Data should contain columns: id, age, name, category
        with values:
            id: [1, 2, 3, 4, 5]
            age: [25, 30, 35, 40, 45]
            name: ["Alice", "Bob", "Charlie", "David", "Eve"]
            category: ["A", "B", "A", "C", "B"]
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def nullable_category_sample_data(self) -> Any:
        """Return framework-specific sample data with null categories.

        Override in framework-specific test class.
        Data should contain columns:
            id: [1, 2, 3, 4, 5]
            category: ["A", None, "B", None, "C"]
            score: [1, None, 2, None, 3]
            ratio: [1.0, NaN, 2.0, None, 3.0]
        Nulls in category and score sit at ids 2 and 4.
        ratio has a NaN at id 2 and a null at id 4, so missing rows still sit at ids 2 and 4.
        """
        raise NotImplementedError

    @abstractmethod
    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Extract column values as a list from the result.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def result_row_count(self, result: Any) -> int:
        """Return the number of rows in a filter result.

        Defaults to ``len(result)`` (correct for row-shaped frames: pandas, polars,
        pyarrow). Columnar frameworks (PythonDict's ``dict[str, list]``) override this,
        since ``len`` on a columnar dict counts columns, not rows.
        """
        return len(result)

    def _assert_values_equal(self, actual: list[Any], expected: list[Any]) -> None:
        """Assert values are equal regardless of order."""
        assert sorted(actual) == sorted(expected)

    def test_do_range_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test range filter with min and max values."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"min": 30, "max": 40, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_range_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 3
        self._assert_values_equal(self.get_column_values(result, "age"), [30, 35, 40])
        self._assert_values_equal(self.get_column_values(result, "id"), [2, 3, 4])

    def test_do_range_filter_exclusive(self, filter_engine: Any, sample_data: Any) -> None:
        """Test range filter with exclusive max value."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"min": 30, "max": 40, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_range_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [30, 35])
        self._assert_values_equal(self.get_column_values(result, "id"), [2, 3])

    def test_do_min_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test min filter."""
        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"value": 40}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_min_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [40, 45])
        self._assert_values_equal(self.get_column_values(result, "id"), [4, 5])

    def test_do_min_filter_drops_null_or_nan_rows(self, filter_engine: Any, nullable_category_sample_data: Any) -> None:
        single_filter = SingleFilter(Feature("ratio"), FilterType.MIN, {"value": 2.0})

        result = filter_engine.do_min_filter(nullable_category_sample_data, single_filter)

        assert self.result_row_count(result) == 2
        self._assert_values_equal(self.get_column_values(result, "id"), [3, 5])

    def test_do_max_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test max filter."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_max_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [25, 30])
        self._assert_values_equal(self.get_column_values(result, "id"), [1, 2])

    def test_do_max_filter_with_tuple(self, filter_engine: Any, sample_data: Any) -> None:
        """Test max filter with tuple parameter."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"max": 35, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_max_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [25, 30])
        self._assert_values_equal(self.get_column_values(result, "id"), [1, 2])

    def test_do_equal_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test equal filter."""
        feature = Feature("age")
        filter_type = FilterType.EQUAL
        parameter = {"value": 35}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_equal_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 1
        assert self.get_column_values(result, "age")[0] == 35
        assert self.get_column_values(result, "id")[0] == 3

    def test_do_regex_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test regex filter."""
        feature = Feature("name")
        filter_type = FilterType.REGEX
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_regex_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 1
        assert self.get_column_values(result, "name")[0] == "Alice"
        assert self.get_column_values(result, "id")[0] == 1

    def test_do_regex_filter_is_unanchored(self, filter_engine: Any, sample_data: Any) -> None:
        """REGEX filter must use unanchored (substring / re.search) semantics.

        The pattern ``"li"`` appears inside ``"Alice"`` and ``"Charlie"`` but neither
        name STARTS with ``"li"``. With unanchored matching (the contract shared by all
        compute frameworks) exactly those two rows survive. Anchored matching
        (``Series.str.match`` / ``re.match``) would return 0 rows.
        """
        feature = Feature("name")
        filter_type = FilterType.REGEX
        parameter = {"value": "li"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_regex_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 2
        self._assert_values_equal(self.get_column_values(result, "name"), ["Alice", "Charlie"])
        self._assert_values_equal(self.get_column_values(result, "id"), [1, 3])

    def test_do_categorical_inclusion_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test categorical inclusion filter."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 4
        assert set(self.get_column_values(result, "category")) == {"A", "B"}
        assert set(self.get_column_values(result, "id")) == {1, 2, 3, 5}

    @pytest.mark.parametrize("make_values", NON_COLLECTION_VALUES)
    def test_do_categorical_inclusion_rejects_non_collection_values(
        self, filter_engine: Any, sample_data: Any, make_values: Any
    ) -> None:
        """Building the SingleFilter raises, so no engine ever filters on a non-collection values."""
        with pytest.raises(TypeError, match="list, tuple, set or frozenset"):
            single_filter = SingleFilter(
                Feature("category"), FilterType.CATEGORICAL_INCLUSION, {"values": make_values()}
            )
            filter_engine.do_categorical_inclusion_filter(sample_data, single_filter)

    def test_do_categorical_inclusion_empty_values(self, filter_engine: Any, sample_data: Any) -> None:
        """An empty allowed-values list must yield an empty result across all frameworks."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter: dict[str, Any] = {"values": []}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(sample_data, single_filter)

        assert self.result_row_count(result) == 0

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("category", ["A", None], id="string_category"),
            pytest.param("score", [1, None], id="numeric_score"),
            pytest.param("ratio", [1.0, None], id="float_ratio"),
            pytest.param("ratio", [1.0, float("nan")], id="float_ratio_nan"),
        ],
    )
    def test_do_categorical_inclusion_keeps_null_when_none_present(
        self, filter_engine: Any, nullable_category_sample_data: Any, column: str, values: list[Any]
    ) -> None:
        """When None is in the allowed-values list, null rows must be KEPT."""
        feature = Feature(column)
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": values}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(nullable_category_sample_data, single_filter)

        assert self.result_row_count(result) == 3
        self._assert_values_equal(self.get_column_values(result, "id"), [1, 2, 4])

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("category", ["A"], id="string_category"),
            pytest.param("score", [1], id="numeric_score"),
        ],
    )
    def test_do_categorical_inclusion_drops_null_when_none_absent(
        self, filter_engine: Any, nullable_category_sample_data: Any, column: str, values: list[Any]
    ) -> None:
        """When None is absent from the allowed-values list, null rows must be DROPPED."""
        feature = Feature(column)
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": values}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(nullable_category_sample_data, single_filter)

        assert self.result_row_count(result) == 1
        self._assert_values_equal(self.get_column_values(result, "id"), [1])

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("category", [None], id="string_category"),
            pytest.param("score", [None], id="numeric_score"),
            pytest.param("ratio", [None], id="float_ratio"),
        ],
    )
    def test_do_categorical_inclusion_only_none_keeps_only_nulls(
        self, filter_engine: Any, nullable_category_sample_data: Any, column: str, values: list[Any]
    ) -> None:
        """An allowed-values list of only [None] keeps exactly the null rows."""
        feature = Feature(column)
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": values}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(nullable_category_sample_data, single_filter)

        assert self.result_row_count(result) == 2
        self._assert_values_equal(self.get_column_values(result, "id"), [2, 4])

    def test_apply_filters(self, filter_engine: Any, sample_data: Any) -> None:
        """Test applying multiple filters."""
        feature = Feature("age")
        filters = [
            SingleFilter(feature, FilterType.MIN, {"value": 30}),
            SingleFilter(Feature("category"), FilterType.EQUAL, {"value": "A"}),
        ]

        class MockFeatureSet:
            def __init__(self, filters: Any) -> None:
                self.filters = filters

            def get_all_names(self) -> Any:
                return ["age", "category"]

        feature_set = MockFeatureSet(filters)

        result = filter_engine.apply_filters(sample_data, feature_set)

        assert self.result_row_count(result) == 1
        assert self.get_column_values(result, "age")[0] == 35
        assert self.get_column_values(result, "category")[0] == "A"
        assert self.get_column_values(result, "id")[0] == 3

    def test_final_filters(self, filter_engine: Any) -> None:
        """Test that final_filters returns True."""
        assert filter_engine.final_filters() is True

    def test_do_range_filter_missing_parameters(self, filter_engine: Any, sample_data: Any) -> None:
        """Test range filter with missing parameters."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"min": 30}  # Missing max parameter
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported"):
            filter_engine.do_range_filter(sample_data, single_filter)

    @pytest.fixture
    @abstractmethod
    def decimal_sample_data(self) -> Any:
        """Return a decimal column d, including a null, with precision 10 and scale 2."""
        raise NotImplementedError

    @abstractmethod
    def get_decimal_column_dtype(self, data: Any) -> Any:
        """Return the native dtype of d, or its Python value type for dictionary data."""
        raise NotImplementedError

    def test_min_filter_decimal(
        self,
        filter_engine: Any,
        decimal_sample_data: Any,
    ) -> None:
        single_filter = SingleFilter(Feature("d"), FilterType.MIN, {"value": Decimal("12.34")})

        result = filter_engine.do_min_filter(decimal_sample_data, single_filter)

        values = self.get_column_values(result, "d")
        assert values == [Decimal("12.34"), Decimal("99.99")]
        assert all(isinstance(value, Decimal) for value in values)
        assert self.get_decimal_column_dtype(result) == self.get_decimal_column_dtype(decimal_sample_data)

    def test_categorical_inclusion_decimal(
        self,
        filter_engine: Any,
        decimal_sample_data: Any,
    ) -> None:
        single_filter = SingleFilter(
            Feature("d"), FilterType.CATEGORICAL_INCLUSION, {"values": [Decimal("12.34"), Decimal("5.50")]}
        )

        result = filter_engine.do_categorical_inclusion_filter(decimal_sample_data, single_filter)

        values = self.get_column_values(result, "d")
        assert values == [Decimal("12.34"), Decimal("5.50")]
        assert all(isinstance(value, Decimal) for value in values)
        assert self.get_decimal_column_dtype(result) == self.get_decimal_column_dtype(decimal_sample_data)

    def test_categorical_inclusion_decimal_unrepresentable_values_match_nothing(
        self,
        filter_engine: Any,
        decimal_sample_data: Any,
    ) -> None:
        """Values that do not survive a round-trip cast to the column's precision/scale must match nothing."""
        single_filter = SingleFilter(
            Feature("d"),
            FilterType.CATEGORICAL_INCLUSION,
            {"values": [Decimal("12.345"), Decimal("99999999999.99")]},
        )

        result = filter_engine.do_categorical_inclusion_filter(decimal_sample_data, single_filter)

        assert self.get_column_values(result, "d") == []
        assert self.get_decimal_column_dtype(result) == self.get_decimal_column_dtype(decimal_sample_data)

    def test_categorical_inclusion_decimal_with_null(
        self,
        filter_engine: Any,
        decimal_sample_data: Any,
    ) -> None:
        single_filter = SingleFilter(
            Feature("d"), FilterType.CATEGORICAL_INCLUSION, {"values": [Decimal("12.34"), None]}
        )

        result = filter_engine.do_categorical_inclusion_filter(decimal_sample_data, single_filter)

        values = self.get_column_values(result, "d")
        assert values == [Decimal("12.34"), None]
        assert self.get_decimal_column_dtype(result) == self.get_decimal_column_dtype(decimal_sample_data)
