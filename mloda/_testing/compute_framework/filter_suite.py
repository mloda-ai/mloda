"""Conformance suite for BaseFilterEngine implementations, run through a framework adapter."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.optional_dependency import require
from mloda.user import Feature, FilterType, SingleFilter

if TYPE_CHECKING:
    import pyarrow as pa

# Zero-arg factories so a fresh generator is produced per test run.
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

SAMPLE_DATA = TypedTestData(
    {
        "id": (DataType.INT64, [1, 2, 3, 4, 5]),
        "age": (DataType.INT64, [25, 30, 35, 40, 45]),
        "name": (DataType.STRING, ["Alice", "Bob", "Charlie", "David", "Eve"]),
        "category": (DataType.STRING, ["A", "B", "A", "C", "B"]),
    }
)

NULLABLE_DATA = TypedTestData(
    {
        "id": (DataType.INT64, [1, 2, 3, 4, 5]),
        "category": (DataType.STRING, ["A", None, "B", None, "C"]),
        "score": (DataType.INT64, [1, None, 2, None, 3]),
        "ratio": (DataType.DOUBLE, [1.0, float("nan"), 2.0, None, 3.0]),
    }
)

DECIMAL_DATA = TypedTestData({"d": (DataType.DECIMAL, [Decimal("12.34"), Decimal("5.50"), Decimal("99.99"), None])})

TIME_DATA = TypedTestData(
    {
        "id": (DataType.INT64, [1, 2, 3, 4, 5, 6]),
        "ts": (DataType.TIMESTAMP_MICROS_TZ, [datetime(2023, 1, day, tzinfo=UTC) for day in range(1, 7)]),
    }
)

RANGE_MIN = datetime(2023, 1, 2, tzinfo=UTC)
RANGE_MAX = datetime(2023, 1, 5, tzinfo=UTC)
EXPECTED_IDS_RANGE_EXCLUSIVE: list[int] = [2, 3, 4]

MIN_BOUND = datetime(2023, 1, 4, tzinfo=UTC)
EXPECTED_IDS_MIN: list[int] = [4, 5, 6]

MAX_BOUND = datetime(2023, 1, 3, tzinfo=UTC)
EXPECTED_IDS_MAX: list[int] = [1, 2, 3]


class FilterEngineSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of the filter engine of the host's compute framework."""

    aspect = "filter"

    @pytest.fixture
    def filter_engine(self, framework: type[ComputeFramework]) -> Any:
        return framework.filter_engine()

    @pytest.fixture
    def sample_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "sample", SAMPLE_DATA)

    @pytest.fixture
    def nullable_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "nullable", NULLABLE_DATA)

    @pytest.fixture
    def decimal_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "decimal", DECIMAL_DATA)

    @pytest.fixture
    def time_data(self, adapter: FrameworkAdapter) -> Any:
        return self.frame(adapter, "time", TIME_DATA)

    def arrow_types(self, name: str) -> Mapping[str, pa.DataType] | None:
        if name == "decimal":
            return {"d": require("pyarrow", "building decimal test tables").decimal128(10, 2)}
        return None

    def per_filter_frame(self, adapter: FrameworkAdapter, frame: Any) -> Any:
        """The frame the do_* methods receive; identity unless the engine runs them on another frame."""
        return frame

    def apply_filters_names(self) -> list[str]:
        return ["age", "category"]

    def decimal_dtype(self, adapter: FrameworkAdapter, result: Any) -> Any:
        return adapter.column_type(result, "d")

    def _assert_values_equal(self, actual: list[Any], expected: list[Any]) -> None:
        """Assert values are equal regardless of order."""
        assert sorted(actual) == sorted(expected)

    def test_do_range_filter(self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        """Test range filter with min and max values."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"min": 30, "max": 40, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_range_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == 3
        self._assert_values_equal(self.column_values(adapter, result, "age"), [30, 35, 40])
        self._assert_values_equal(self.column_values(adapter, result, "id"), [2, 3, 4])

    def test_do_range_filter_exclusive(self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        """Test range filter with exclusive max value."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"min": 30, "max": 40, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_range_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == 2
        self._assert_values_equal(self.column_values(adapter, result, "age"), [30, 35])
        self._assert_values_equal(self.column_values(adapter, result, "id"), [2, 3])

    def test_do_min_filter(self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        """Test min filter."""
        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"value": 40}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_min_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == 2
        self._assert_values_equal(self.column_values(adapter, result, "age"), [40, 45])
        self._assert_values_equal(self.column_values(adapter, result, "id"), [4, 5])

    def test_do_min_filter_drops_null_or_nan_rows(
        self, filter_engine: Any, adapter: FrameworkAdapter, nullable_data: Any
    ) -> None:
        single_filter = SingleFilter(Feature("ratio"), FilterType.MIN, {"value": 2.0})

        result = filter_engine.do_min_filter(self.per_filter_frame(adapter, nullable_data), single_filter)

        assert self.row_count(adapter, result) == 2
        self._assert_values_equal(self.column_values(adapter, result, "id"), [3, 5])

    def test_do_max_filter(self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        """Test max filter."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_max_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == 2
        self._assert_values_equal(self.column_values(adapter, result, "age"), [25, 30])
        self._assert_values_equal(self.column_values(adapter, result, "id"), [1, 2])

    @pytest.mark.parametrize(
        ("max_exclusive", "expected_ages", "expected_ids"),
        [
            pytest.param(True, [25, 30], [1, 2], id="exclusive"),
            pytest.param(False, [25, 30, 35], [1, 2, 3], id="inclusive"),
        ],
    )
    def test_do_max_filter_with_tuple(
        self,
        filter_engine: Any,
        adapter: FrameworkAdapter,
        sample_data: Any,
        max_exclusive: bool,
        expected_ages: list[int],
        expected_ids: list[int],
    ) -> None:
        """Test max filter with tuple parameter."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"max": 35, "max_exclusive": max_exclusive}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_max_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == len(expected_ages)
        self._assert_values_equal(self.column_values(adapter, result, "age"), expected_ages)
        self._assert_values_equal(self.column_values(adapter, result, "id"), expected_ids)

    def test_do_equal_filter(self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        """Test equal filter."""
        feature = Feature("age")
        filter_type = FilterType.EQUAL
        parameter = {"value": 35}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_equal_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == 1
        assert self.column_values(adapter, result, "age")[0] == 35
        assert self.column_values(adapter, result, "id")[0] == 3

    def test_do_regex_filter(self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        """Test regex filter."""
        feature = Feature("name")
        filter_type = FilterType.REGEX
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_regex_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == 1
        assert self.column_values(adapter, result, "name")[0] == "Alice"
        assert self.column_values(adapter, result, "id")[0] == 1

    def test_do_regex_filter_is_unanchored(
        self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any
    ) -> None:
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

        result = filter_engine.do_regex_filter(self.per_filter_frame(adapter, sample_data), single_filter)

        assert self.row_count(adapter, result) == 2
        self._assert_values_equal(self.column_values(adapter, result, "name"), ["Alice", "Charlie"])
        self._assert_values_equal(self.column_values(adapter, result, "id"), [1, 3])

    def test_do_categorical_inclusion_filter(
        self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any
    ) -> None:
        """Test categorical inclusion filter."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, sample_data), single_filter
        )

        assert self.row_count(adapter, result) == 4
        assert set(self.column_values(adapter, result, "category")) == {"A", "B"}
        assert set(self.column_values(adapter, result, "id")) == {1, 2, 3, 5}

    @pytest.mark.parametrize("make_values", NON_COLLECTION_VALUES)
    def test_do_categorical_inclusion_rejects_non_collection_values(
        self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any, make_values: Any
    ) -> None:
        """Building the SingleFilter raises, so no engine ever filters on a non-collection values."""
        with pytest.raises(TypeError, match="list, tuple, set or frozenset"):
            single_filter = SingleFilter(
                Feature("category"), FilterType.CATEGORICAL_INCLUSION, {"values": make_values()}
            )
            filter_engine.do_categorical_inclusion_filter(self.per_filter_frame(adapter, sample_data), single_filter)

    def test_do_categorical_inclusion_empty_values(
        self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any
    ) -> None:
        """An empty allowed-values list must yield an empty result across all frameworks."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter: dict[str, Any] = {"values": []}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, sample_data), single_filter
        )

        assert self.row_count(adapter, result) == 0

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
        self, filter_engine: Any, adapter: FrameworkAdapter, nullable_data: Any, column: str, values: list[Any]
    ) -> None:
        """When None is in the allowed-values list, null rows must be KEPT."""
        feature = Feature(column)
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": values}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, nullable_data), single_filter
        )

        assert self.row_count(adapter, result) == 3
        self._assert_values_equal(self.column_values(adapter, result, "id"), [1, 2, 4])

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("category", ["A"], id="string_category"),
            pytest.param("score", [1], id="numeric_score"),
        ],
    )
    def test_do_categorical_inclusion_drops_null_when_none_absent(
        self, filter_engine: Any, adapter: FrameworkAdapter, nullable_data: Any, column: str, values: list[Any]
    ) -> None:
        """When None is absent from the allowed-values list, null rows must be DROPPED."""
        feature = Feature(column)
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": values}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, nullable_data), single_filter
        )

        assert self.row_count(adapter, result) == 1
        self._assert_values_equal(self.column_values(adapter, result, "id"), [1])

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("category", [None], id="string_category"),
            pytest.param("score", [None], id="numeric_score"),
            pytest.param("ratio", [None], id="float_ratio"),
        ],
    )
    def test_do_categorical_inclusion_only_none_keeps_only_nulls(
        self, filter_engine: Any, adapter: FrameworkAdapter, nullable_data: Any, column: str, values: list[Any]
    ) -> None:
        """An allowed-values list of only [None] keeps exactly the null rows."""
        feature = Feature(column)
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": values}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, nullable_data), single_filter
        )

        assert self.row_count(adapter, result) == 2
        self._assert_values_equal(self.column_values(adapter, result, "id"), [2, 4])

    def test_apply_filters(self, filter_engine: Any, adapter: FrameworkAdapter, sample_data: Any) -> None:
        """Test applying multiple filters."""
        feature = Feature("age")
        filters = [
            SingleFilter(feature, FilterType.MIN, {"value": 30}),
            SingleFilter(Feature("category"), FilterType.EQUAL, {"value": "A"}),
        ]

        names = self.apply_filters_names()

        class MockFeatureSet:
            def __init__(self, filters: Any) -> None:
                self.filters = filters
                self.link_index_columns: frozenset[str] = frozenset()

            def get_all_names(self) -> Any:
                return names

        feature_set = MockFeatureSet(filters)

        result = filter_engine.apply_filters(sample_data, feature_set)

        assert self.row_count(adapter, result) == 1
        assert self.column_values(adapter, result, "age")[0] == 35
        assert self.column_values(adapter, result, "category")[0] == "A"
        assert self.column_values(adapter, result, "id")[0] == 3

    def test_final_filters(self, filter_engine: Any) -> None:
        """Test that final_filters returns True."""
        assert filter_engine.final_filters() is True

    @pytest.mark.parametrize(
        ("filter_type", "column", "parameter", "match"),
        [
            pytest.param(
                FilterType.RANGE, "age", {"min": 30}, "Filter parameter .* not supported", id="range_missing_max"
            ),
            pytest.param(
                FilterType.MIN, "age", {"invalid": 30}, "Filter parameter 'value' not found", id="min_missing_value"
            ),
            pytest.param(
                FilterType.MIN, "age", {"min": 30}, "Filter parameter 'value' not found", id="min_rejects_min_key"
            ),
            pytest.param(
                FilterType.EQUAL, "age", {"invalid": 30}, "Filter parameter 'value' not found", id="equal_missing_value"
            ),
            pytest.param(
                FilterType.REGEX,
                "name",
                {"invalid": "^A"},
                "Filter parameter 'value' not found",
                id="regex_missing_value",
            ),
            pytest.param(
                FilterType.CATEGORICAL_INCLUSION,
                "category",
                {"invalid": ["A", "B"]},
                "Filter parameter 'values' not found",
                id="categorical_inclusion_missing_values",
            ),
            pytest.param(
                FilterType.MAX, "age", {"invalid": 30}, "No valid filter parameter found", id="max_invalid_parameters"
            ),
            pytest.param(
                FilterType.MAX,
                "age",
                {"min": 20, "max": 30},
                "Filter parameter .* not supported as max filter",
                id="max_with_min_parameter",
            ),
        ],
    )
    def test_do_filter_rejects_invalid_parameters(
        self,
        filter_engine: Any,
        adapter: FrameworkAdapter,
        sample_data: Any,
        filter_type: FilterType,
        column: str,
        parameter: dict[str, Any],
        match: str,
    ) -> None:
        single_filter = SingleFilter(Feature(column), filter_type, parameter)

        with pytest.raises(ValueError, match=match):
            filter_engine.do_filter(self.per_filter_frame(adapter, sample_data), single_filter)

    def test_min_filter_decimal(
        self,
        filter_engine: Any,
        adapter: FrameworkAdapter,
        decimal_data: Any,
    ) -> None:
        single_filter = SingleFilter(Feature("d"), FilterType.MIN, {"value": Decimal("12.34")})

        result = filter_engine.do_min_filter(self.per_filter_frame(adapter, decimal_data), single_filter)

        values = self.column_values(adapter, result, "d")
        assert values == [Decimal("12.34"), Decimal("99.99")]
        assert all(isinstance(value, Decimal) for value in values)
        assert self.decimal_dtype(adapter, result) == self.decimal_dtype(adapter, decimal_data)

    def test_categorical_inclusion_decimal(
        self,
        filter_engine: Any,
        adapter: FrameworkAdapter,
        decimal_data: Any,
    ) -> None:
        single_filter = SingleFilter(
            Feature("d"), FilterType.CATEGORICAL_INCLUSION, {"values": [Decimal("12.34"), Decimal("5.50")]}
        )

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, decimal_data), single_filter
        )

        values = self.column_values(adapter, result, "d")
        assert values == [Decimal("12.34"), Decimal("5.50")]
        assert all(isinstance(value, Decimal) for value in values)
        assert self.decimal_dtype(adapter, result) == self.decimal_dtype(adapter, decimal_data)

    def test_categorical_inclusion_decimal_unrepresentable_values_match_nothing(
        self,
        filter_engine: Any,
        adapter: FrameworkAdapter,
        decimal_data: Any,
    ) -> None:
        """Values that do not survive a round-trip cast to the column's precision/scale must match nothing."""
        single_filter = SingleFilter(
            Feature("d"),
            FilterType.CATEGORICAL_INCLUSION,
            {"values": [Decimal("12.345"), Decimal("99999999999.99")]},
        )

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, decimal_data), single_filter
        )

        assert self.column_values(adapter, result, "d") == []
        assert self.decimal_dtype(adapter, result) == self.decimal_dtype(adapter, decimal_data)

    def test_categorical_inclusion_decimal_with_null(
        self,
        filter_engine: Any,
        adapter: FrameworkAdapter,
        decimal_data: Any,
    ) -> None:
        single_filter = SingleFilter(
            Feature("d"), FilterType.CATEGORICAL_INCLUSION, {"values": [Decimal("12.34"), None]}
        )

        result = filter_engine.do_categorical_inclusion_filter(
            self.per_filter_frame(adapter, decimal_data), single_filter
        )

        values = self.column_values(adapter, result, "d")
        assert values == [Decimal("12.34"), None]
        assert self.decimal_dtype(adapter, result) == self.decimal_dtype(adapter, decimal_data)

    def test_do_range_filter_tz_aware_datetime_bounds(
        self, filter_engine: Any, adapter: FrameworkAdapter, time_data: Any
    ) -> None:
        """Range filter with tz-aware UTC datetime bounds against a tz-aware temporal column."""
        single_filter = SingleFilter(
            Feature("ts"),
            FilterType.RANGE,
            {"min": RANGE_MIN, "max": RANGE_MAX, "max_exclusive": True},
        )

        result = filter_engine.do_range_filter(self.per_filter_frame(adapter, time_data), single_filter)

        ids = sorted(self.column_values(adapter, result, "id"))
        assert ids == EXPECTED_IDS_RANGE_EXCLUSIVE

    def test_do_min_filter_tz_aware_datetime_bound(
        self, filter_engine: Any, adapter: FrameworkAdapter, time_data: Any
    ) -> None:
        """Min filter with a tz-aware UTC datetime bound against a tz-aware temporal column."""
        single_filter = SingleFilter(Feature("ts"), FilterType.MIN, {"value": MIN_BOUND})

        result = filter_engine.do_min_filter(self.per_filter_frame(adapter, time_data), single_filter)

        ids = sorted(self.column_values(adapter, result, "id"))
        assert ids == EXPECTED_IDS_MIN

    def test_do_max_filter_tz_aware_datetime_bound(
        self, filter_engine: Any, adapter: FrameworkAdapter, time_data: Any
    ) -> None:
        """Max filter with a tz-aware UTC datetime bound against a tz-aware temporal column."""
        single_filter = SingleFilter(Feature("ts"), FilterType.MAX, {"value": MAX_BOUND})

        result = filter_engine.do_max_filter(self.per_filter_frame(adapter, time_data), single_filter)

        ids = sorted(self.column_values(adapter, result, "id"))
        assert ids == EXPECTED_IDS_MAX
