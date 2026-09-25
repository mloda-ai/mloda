import datetime
from decimal import Decimal
from typing import Any
import pytest
import numpy as np
from unittest.mock import Mock, patch

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_filter_engine import IcebergFilterEngine
from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)

import logging

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        NestedField,
        LongType,
        StringType,
        DoubleType,
        DateType,
        TimestampType,
        StructType,
        FloatType,
        DecimalType,
        IntegerType,
        TimestamptzType,
        BooleanType,
    )
    from pyiceberg.expressions import (
        GreaterThan,
        LessThan,
        GreaterThanOrEqual,
        LessThanOrEqual,
        EqualTo,
        And,
        Or,
        In,
        IsNull,
        IsNaN,
        Reference,
        AlwaysTrue,
    )
    from pyiceberg.expressions.visitors import bind
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None  # type: ignore[assignment, unused-ignore]
    IcebergTable = None  # type: ignore
    Schema = None  # type: ignore
    NestedField = None  # type: ignore
    LongType = None  # type: ignore
    StringType = None  # type: ignore
    DoubleType = None  # type: ignore
    DateType = None  # type: ignore
    TimestampType = None  # type: ignore
    StructType = None  # type: ignore
    FloatType = None  # type: ignore
    DecimalType = None  # type: ignore
    IntegerType = None  # type: ignore
    TimestamptzType = None  # type: ignore
    BooleanType = None  # type: ignore
    GreaterThan = None  # type: ignore
    LessThan = None  # type: ignore
    GreaterThanOrEqual = None  # type: ignore
    LessThanOrEqual = None  # type: ignore
    EqualTo = None  # type: ignore
    And = None  # type: ignore
    Or = None  # type: ignore
    In = None  # type: ignore
    IsNull = None  # type: ignore
    IsNaN = None  # type: ignore
    Reference = None  # type: ignore
    AlwaysTrue = None  # type: ignore
    bind = None  # type: ignore


def build_mock_iceberg_table(data: Any, schema: Any) -> Mock:
    """Mock IcebergTable whose scan binds row_filter against schema (as a real scan would) then returns data unfiltered."""

    def _scan(*args: Any, **kwargs: Any) -> Mock:
        row_filter = kwargs.get("row_filter", AlwaysTrue())
        bind(schema, row_filter, case_sensitive=True)
        mock_scan = Mock()
        mock_scan.to_arrow.return_value = data
        return mock_scan

    mock_table = Mock(spec=IcebergTable)
    mock_table.schema.return_value = schema
    mock_table.scan.side_effect = _scan
    return mock_table


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergFilterEngine(FilterEngineTestMixin):
    """Unit tests for the IcebergFilterEngine class."""

    filter_engine_class = IcebergFilterEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        return pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

    @pytest.fixture
    def sample_schema(self) -> Any:
        return Schema(
            NestedField(1, "id", LongType(), required=False),
            NestedField(2, "age", LongType(), required=False),
            NestedField(3, "name", StringType(), required=False),
            NestedField(4, "category", StringType(), required=False),
        )

    @pytest.fixture
    def mock_iceberg_table(self, sample_data: Any, sample_schema: Any) -> Mock:
        """Mock Iceberg table whose scan returns sample_data unfiltered; filtering happens in the PyArrow re-pass."""
        return build_mock_iceberg_table(sample_data, sample_schema)

    @pytest.fixture
    def mock_feature_set(self) -> Mock:
        """Create a mock feature set for testing."""
        mock_feature_set = Mock()
        mock_feature_set.get_all_names.return_value = ["age", "name", "category"]
        return mock_feature_set

    @pytest.fixture
    def nullable_category_sample_data(self) -> Any:
        return pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", None, "B", None, "C"],
                "score": [1, None, 2, None, 3],
                "ratio": pa.array([1.0, float("nan"), 2.0, None, 3.0], type=pa.float64()),
            }
        )

    @pytest.fixture
    def decimal_sample_data(self) -> Any:
        values = [Decimal("12.34"), Decimal("5.50"), Decimal("99.99"), None]
        return pa.table({"d": pa.array(values, type=pa.decimal128(10, 2))})

    def get_decimal_column_dtype(self, data: Any) -> Any:
        return data["d"].type

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        return result[column].to_pylist()  # type: ignore[no-any-return]

    @pytest.mark.parametrize(
        "filter_type,parameter,expected_expression",
        [
            pytest.param(
                FilterType.EQUAL,
                {"value": 30},
                lambda: EqualTo(Reference("age"), 30),
                id="equal",
            ),
            pytest.param(
                FilterType.MIN,
                {"value": 25},
                lambda: GreaterThanOrEqual(Reference("age"), 25),
                id="min",
            ),
            pytest.param(
                FilterType.MAX,
                {"value": 50},
                lambda: LessThanOrEqual(Reference("age"), 50),
                id="max_simple",
            ),
            pytest.param(
                FilterType.MAX,
                {"max": 50, "max_exclusive": True},
                lambda: LessThanOrEqual(Reference("age"), 50),
                id="max_exclusive_pushed_inclusive",
            ),
            pytest.param(
                FilterType.RANGE,
                {"min": 25, "max": 50, "max_exclusive": False},
                lambda: And(GreaterThanOrEqual(Reference("age"), 25), LessThanOrEqual(Reference("age"), 50)),
                id="range",
            ),
            pytest.param(
                FilterType.RANGE,
                {"min": 25, "max": 50, "max_exclusive": True},
                lambda: And(GreaterThanOrEqual(Reference("age"), 25), LessThanOrEqual(Reference("age"), 50)),
                id="range_exclusive_pushed_inclusive",
            ),
            pytest.param(
                FilterType.CATEGORICAL_INCLUSION,
                {"values": [30, 40]},
                lambda: In(Reference("age"), {30, 40}),
                id="categorical_inclusion",
            ),
            pytest.param(
                FilterType.CATEGORICAL_INCLUSION,
                {"values": [30, None]},
                lambda: Or(In(Reference("age"), {30}), IsNull(Reference("age")), IsNaN(Reference("age"))),
                id="categorical_inclusion_with_none",
            ),
            pytest.param(
                FilterType.CATEGORICAL_INCLUSION,
                {"values": [30.0, float("nan")]},
                lambda: Or(In(Reference("age"), {30.0}), IsNull(Reference("age")), IsNaN(Reference("age"))),
                id="categorical_inclusion_with_nan",
            ),
        ],
    )
    def test_build_iceberg_expression(
        self, filter_type: FilterType, parameter: dict[str, Any], expected_expression: Any
    ) -> None:
        """pyiceberg unbound predicates compare by value, so the exact expression is asserted."""
        feature = Feature("age")
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)

        assert expression == expected_expression()

    def test_build_iceberg_expression_unsupported(self) -> None:
        """Test that an unsupported filter type raises NotImplementedError."""
        feature = Feature("name")
        filter_type = FilterType.REGEX
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(NotImplementedError):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_build_iceberg_expression_equal_missing_value_raises(self) -> None:
        """EQUAL with no value must raise ValueError, not silently return None."""
        feature = Feature("age")
        filter_type = FilterType.EQUAL
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_build_iceberg_expression_min_missing_value_raises(self) -> None:
        """MIN with no value must raise ValueError, not silently return None."""
        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_build_iceberg_expression_max_missing_value_raises(self) -> None:
        """MAX with neither value nor max must raise ValueError, matching the base engine's message."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="No valid filter parameter found"):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_build_iceberg_expression_max_with_min_raises(self) -> None:
        """MAX with both min and max present must raise ValueError instead of silently dropping min."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"min": 25, "max": 50}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="not supported as max filter"):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_build_iceberg_expression_range_missing_both_raises(self) -> None:
        """RANGE with neither min nor max must raise ValueError, not silently return None."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="not supported"):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_build_iceberg_expression_range_missing_max_raises(self) -> None:
        """RANGE requires both bounds; a min-only range must raise ValueError."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"min": 25}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="not supported"):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_build_iceberg_expression_range_missing_min_raises(self) -> None:
        """RANGE requires both bounds; a max-only range must raise ValueError."""
        feature = Feature("age")
        filter_type = FilterType.RANGE
        parameter = {"max": 50}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="not supported"):
            IcebergFilterEngine._build_iceberg_expression(single_filter)

    def test_extract_parameter_value(self) -> None:
        """Test extracting parameter values."""
        feature = Feature("age")
        filter_type = FilterType.EQUAL
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        value = IcebergFilterEngine._extract_parameter_value(single_filter, "value")
        assert value == 30

        # Test missing parameter
        missing_value = IcebergFilterEngine._extract_parameter_value(single_filter, "missing")
        assert missing_value is None

    def test_has_parameter(self) -> None:
        """Test checking if parameter exists."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"max": 50, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        assert IcebergFilterEngine._has_parameter(single_filter, "max") is True
        assert IcebergFilterEngine._has_parameter(single_filter, "max_exclusive") is True
        assert IcebergFilterEngine._has_parameter(single_filter, "missing") is False

    def test_apply_filters_iceberg_table(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """Test applying filters to Iceberg table."""
        # Create filters
        age_filter = SingleFilter(Feature("age"), FilterType.MIN, {"value": 25})
        mock_feature_set.filters = [age_filter]

        # Apply filters
        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        mock_iceberg_table.scan.assert_called_once()
        call_args = mock_iceberg_table.scan.call_args
        assert "row_filter" in call_args.kwargs
        assert call_args.kwargs["row_filter"] == GreaterThanOrEqual(Reference("age"), 25)

        # The PyArrow re-pass over the scan result keeps every row (age >= 25 matches all).
        assert self.get_column_values(result, "id") == [1, 2, 3, 4, 5]

    def test_apply_filters_non_iceberg_table(self, mock_feature_set: Mock) -> None:
        """Test applying filters to non-Iceberg data falls back to parent method."""
        non_iceberg_data = "not_iceberg_table"
        mock_feature_set.filters = []

        # This should fall back to the parent class method
        with patch.object(IcebergFilterEngine.__bases__[0], "apply_filters") as mock_parent:
            mock_parent.return_value = "filtered_data"
            result = IcebergFilterEngine.apply_filters(non_iceberg_data, mock_feature_set)

            mock_parent.assert_called_once_with(non_iceberg_data, mock_feature_set)
            assert result == "filtered_data"

    def test_apply_filters_no_filters(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """Test applying filters when no filters are present."""
        mock_feature_set.filters = None

        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        # Should return original data without calling scan
        assert result is mock_iceberg_table
        mock_iceberg_table.scan.assert_not_called()

    def test_apply_filters_empty_filters(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """Test applying filters when filter list is empty."""
        mock_feature_set.filters = []

        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        # Should return original data without calling scan
        assert result is mock_iceberg_table
        mock_iceberg_table.scan.assert_not_called()

    def test_apply_filters_multiple_filters(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """Test applying multiple filters."""
        # Create multiple filters
        age_filter = SingleFilter(Feature("age"), FilterType.MIN, {"value": 25})
        name_filter = SingleFilter(Feature("name"), FilterType.EQUAL, {"value": "Alice"})
        mock_feature_set.filters = [age_filter, name_filter]

        # Apply filters
        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        mock_iceberg_table.scan.assert_called_once()
        call_args = mock_iceberg_table.scan.call_args
        assert "row_filter" in call_args.kwargs
        assert call_args.kwargs["row_filter"] == And(
            GreaterThanOrEqual(Reference("age"), 25), EqualTo(Reference("name"), "Alice")
        )
        assert self.get_column_values(result, "id") == [1]

    def test_apply_filters_filtered_features(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """Test applying filters where some features are not in the feature set."""
        # Create filter for feature not in feature set
        unknown_filter = SingleFilter(Feature("unknown_column"), FilterType.EQUAL, {"value": "test"})
        age_filter = SingleFilter(Feature("age"), FilterType.MIN, {"value": 25})
        mock_feature_set.filters = [unknown_filter, age_filter]

        # Apply filters
        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        # Should only apply the age filter (unknown_column is not in get_all_names)
        mock_iceberg_table.scan.assert_called_once()
        call_args = mock_iceberg_table.scan.call_args
        assert call_args.kwargs["row_filter"] == GreaterThanOrEqual(Reference("age"), 25)
        assert self.get_column_values(result, "id") == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize(
        "filters,expected_row_filter,expected_ids",
        [
            pytest.param(
                [
                    SingleFilter(Feature("age"), FilterType.MIN, {"value": 30}),
                    SingleFilter(Feature("category"), FilterType.CATEGORICAL_INCLUSION, {"values": ["A", "B"]}),
                ],
                lambda: And(GreaterThanOrEqual(Reference("age"), 30), In(Reference("category"), {"A", "B"})),
                [2, 3, 5],
                id="min_pushed_plus_categorical_inclusion",
            ),
            pytest.param(
                [
                    SingleFilter(Feature("age"), FilterType.MIN, {"value": 30}),
                    SingleFilter(Feature("name"), FilterType.REGEX, {"value": "^[A-C]"}),
                ],
                lambda: GreaterThanOrEqual(Reference("age"), 30),
                [2, 3],
                id="min_pushed_plus_regex",
            ),
            pytest.param(
                [SingleFilter(Feature("category"), FilterType.CATEGORICAL_INCLUSION, {"values": ["A", "B"]})],
                lambda: In(Reference("category"), {"A", "B"}),
                [1, 2, 3, 5],
                id="categorical_inclusion_only_pushed",
            ),
        ],
    )
    def test_apply_filters_non_pushdown_types_via_pyarrow_pass(
        self,
        mock_iceberg_table: Mock,
        mock_feature_set: Mock,
        filters: list[SingleFilter],
        expected_row_filter: Any,
        expected_ids: list[int],
    ) -> None:
        """Categorical inclusion is pushed; regex is not but is applied on the scan result."""
        mock_feature_set.filters = filters

        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        mock_iceberg_table.scan.assert_called_once()
        call_args = mock_iceberg_table.scan.call_args
        assert call_args.kwargs["row_filter"] == expected_row_filter()
        assert self.get_column_values(result, "id") == expected_ids

    def test_apply_filters_custom_filter_type_raises(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """A filter type with no Iceberg pushdown and no PyArrow method reaches do_custom_filter."""
        custom_filter = SingleFilter(Feature("age"), "custom_op", {"value": 1})
        mock_feature_set.filters = [custom_filter]

        with pytest.raises(NotImplementedError, match="Custom filtering is not supported"):
            IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

    @pytest.mark.parametrize(
        "filter_feature,expected_match",
        [
            pytest.param(
                SingleFilter(Feature("age"), FilterType.EQUAL, {"invalid": 30}),
                "Filter parameter 'value' not found",
                id="equal",
            ),
            pytest.param(
                SingleFilter(Feature("category"), FilterType.CATEGORICAL_INCLUSION, {"invalid": ["A"]}),
                "Filter parameter 'values' not found",
                id="categorical_inclusion",
            ),
        ],
    )
    def test_apply_filters_missing_value_raises(
        self, mock_iceberg_table: Mock, mock_feature_set: Mock, filter_feature: SingleFilter, expected_match: str
    ) -> None:
        """apply_filters must not silently drop a malformed filter and return the unfiltered table."""
        mock_feature_set.filters = [filter_feature]

        with pytest.raises(ValueError, match=expected_match):
            IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        # An unfiltered scan would indicate the filter was silently dropped instead of raising.
        mock_iceberg_table.scan.assert_not_called()

    def test_unsupported_filter_methods(self) -> None:
        """Test that unsupported filter methods raise NotImplementedError."""
        mock_data = Mock()
        mock_filter = Mock()

        with pytest.raises(NotImplementedError, match="Custom filtering is not supported"):
            IcebergFilterEngine.do_custom_filter(mock_data, mock_filter)


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergFilterEngineStructAndTypePinning:
    """Type-exact pushdown decisions and nested-field survival, on one 4-row table."""

    @pytest.fixture
    def struct_schema(self) -> Any:
        return Schema(
            NestedField(1, "l", LongType(), required=False),
            NestedField(2, "x", DoubleType(), required=False),
            NestedField(3, "d", DateType(), required=False),
            NestedField(4, "ts", TimestampType(), required=False),
            NestedField(5, "b", StructType(NestedField(6, "c", LongType(), required=False)), required=False),
            NestedField(7, "f", FloatType(), required=False),
            NestedField(8, "dec", DecimalType(10, 2), required=False),
            NestedField(9, "i", IntegerType(), required=False),
            NestedField(10, "tz", TimestamptzType(), required=False),
            NestedField(11, "bo", BooleanType(), required=False),
        )

    @pytest.fixture
    def struct_data(self) -> Any:
        return pa.table(
            {
                "l": pa.array([1, 2, 3, 4], type=pa.int64()),
                "x": pa.array([1.0, 2.0, float("nan"), None], type=pa.float64()),
                "d": pa.array(
                    [
                        datetime.date(2024, 1, 1),
                        datetime.date(2024, 1, 2),
                        datetime.date(2024, 1, 3),
                        datetime.date(2024, 1, 4),
                    ],
                    type=pa.date32(),
                ),
                "ts": pa.array(
                    [
                        datetime.datetime(2024, 1, 1),
                        datetime.datetime(2024, 1, 2),
                        datetime.datetime(2024, 1, 3),
                        datetime.datetime(2024, 1, 4),
                    ],
                    type=pa.timestamp("us"),
                ),
                "b": pa.array(
                    [{"c": 100}, {"c": 200}, None, {"c": 400}],
                    type=pa.struct([("c", pa.int64())]),
                ),
                "f": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32()),
                "dec": pa.array(
                    [Decimal("1.00"), Decimal("2.00"), Decimal("3.00"), Decimal("4.00")],
                    type=pa.decimal128(10, 2),
                ),
                "i": pa.array([1, 2, 3, 4], type=pa.int32()),
                "tz": pa.array(
                    [
                        datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
                        datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc),
                        datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc),
                        datetime.datetime(2024, 1, 4, tzinfo=datetime.timezone.utc),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "bo": pa.array([True, False, True, False], type=pa.bool_()),
            }
        )

    @pytest.fixture
    def mock_struct_table(self, struct_data: Any, struct_schema: Any) -> Mock:
        return build_mock_iceberg_table(struct_data, struct_schema)

    @pytest.fixture
    def struct_feature_set(self) -> Mock:
        mock_feature_set = Mock()
        mock_feature_set.get_all_names.return_value = ["l", "x", "d", "ts", "b.c", "f", "dec", "i", "tz", "bo"]
        return mock_feature_set

    @pytest.mark.parametrize(
        "filter_feature,expected_row_filter,expected_l,expected_bc",
        [
            pytest.param(
                lambda: SingleFilter(Feature("l"), FilterType.MIN, {"value": 2.5}),
                lambda: AlwaysTrue(),
                [3, 4],
                [None, 400],
                id="min_float_on_long_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("x"), FilterType.EQUAL, {"value": float("nan")}),
                lambda: AlwaysTrue(),
                [],
                [],
                id="equal_nan_on_double_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("l"), FilterType.EQUAL, {"value": np.int64(2)}),
                lambda: AlwaysTrue(),
                [2],
                [200],
                id="equal_numpy_int64_on_long_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("d"), FilterType.EQUAL, {"value": datetime.datetime(2024, 1, 1)}),
                lambda: AlwaysTrue(),
                [1],
                [100],
                id="equal_datetime_on_date_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("l"), FilterType.CATEGORICAL_INCLUSION, {"values": [True]}),
                lambda: AlwaysTrue(),
                [1],
                [100],
                id="categorical_bool_on_long_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(
                    Feature("l"), FilterType.CATEGORICAL_INCLUSION, {"values": [np.int64(2), np.int64(3)]}
                ),
                lambda: AlwaysTrue(),
                [2, 3],
                [200, None],
                id="categorical_numpy_int64_on_long_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("x"), FilterType.CATEGORICAL_INCLUSION, {"values": [1.0, None]}),
                lambda: Or(In(Reference("x"), {1.0}), IsNull(Reference("x")), IsNaN(Reference("x"))),
                [1, 3, 4],
                [100, None, 400],
                id="categorical_with_null_on_double_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(
                    Feature("d"),
                    FilterType.RANGE,
                    {"min": datetime.date(2024, 1, 2), "max": datetime.date(2024, 1, 3), "max_exclusive": False},
                ),
                lambda: And(
                    GreaterThanOrEqual(Reference("d"), datetime.date(2024, 1, 2)),
                    LessThanOrEqual(Reference("d"), datetime.date(2024, 1, 3)),
                ),
                [2, 3],
                [200, None],
                id="range_date_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("ts"), FilterType.MIN, {"value": datetime.datetime(2024, 1, 3)}),
                lambda: GreaterThanOrEqual(Reference("ts"), datetime.datetime(2024, 1, 3)),
                [3, 4],
                [None, 400],
                id="min_naive_datetime_on_timestamp_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("b.c"), FilterType.EQUAL, {"value": 200}),
                lambda: EqualTo(Reference("b.c"), 200),
                [2],
                [200],
                id="equal_on_nested_struct_field_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("x"), FilterType.MAX, {"max": 2.0, "max_exclusive": True}),
                lambda: LessThanOrEqual(Reference("x"), 2.0),
                [1],
                [100],
                id="max_exclusive_on_double_pushed_inclusive",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("f"), FilterType.MIN, {"value": 1.5}),
                lambda: AlwaysTrue(),
                [2, 3, 4],
                [200, None, 400],
                id="min_float32_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("dec"), FilterType.MIN, {"value": Decimal("1.50")}),
                lambda: AlwaysTrue(),
                [2, 3, 4],
                [200, None, 400],
                id="min_decimal_not_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("i"), FilterType.EQUAL, {"value": 2}),
                lambda: EqualTo(Reference("i"), 2),
                [2],
                [200],
                id="equal_int32_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(
                    Feature("tz"),
                    FilterType.MIN,
                    {"value": datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc)},
                ),
                lambda: GreaterThanOrEqual(
                    Reference("tz"), datetime.datetime(2024, 1, 3, tzinfo=datetime.timezone.utc)
                ),
                [3, 4],
                [None, 400],
                id="min_aware_datetime_on_timestamptz_pushed",
            ),
            pytest.param(
                lambda: SingleFilter(Feature("bo"), FilterType.EQUAL, {"value": True}),
                lambda: EqualTo(Reference("bo"), True),
                [1, 3],
                [100, None],
                id="equal_bool_pushed",
            ),
        ],
    )
    def test_apply_filters_struct_field_and_type_pinning(
        self,
        mock_struct_table: Mock,
        struct_feature_set: Mock,
        filter_feature: Any,
        expected_row_filter: Any,
        expected_l: list[int],
        expected_bc: list[Any],
    ) -> None:
        """A filter is pushed only on exact-type match; b.c (nested, dropped by the plain scan) is kept when any filter applies."""
        struct_feature_set.filters = [filter_feature()]

        result = IcebergFilterEngine.apply_filters(mock_struct_table, struct_feature_set)

        mock_struct_table.scan.assert_called_once()
        call_args = mock_struct_table.scan.call_args
        assert call_args.kwargs["row_filter"] == expected_row_filter()
        assert result["l"].to_pylist() == expected_l
        assert result["b.c"].to_pylist() == expected_bc


@pytest.mark.skipif(
    pyiceberg is not None and pa is not None, reason="PyIceberg and PyArrow are installed. Skipping unavailable test."
)
class TestIcebergFilterEngineUnavailable:
    """Test behavior when PyIceberg expressions are not available."""

    def test_build_iceberg_expression_unavailable(self) -> None:
        """Test building expression when Iceberg expressions are not available."""
        with patch("mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_filter_engine.EqualTo", None):
            feature = Feature("age")
            filter_type = FilterType.EQUAL
            parameter = {"value": 30}
            single_filter = SingleFilter(feature, filter_type, parameter)

            expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
            assert expression is None
