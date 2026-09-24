from decimal import Decimal
from typing import Any
import pytest
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
    from pyiceberg.expressions import (
        GreaterThan,
        LessThan,
        GreaterThanOrEqual,
        LessThanOrEqual,
        EqualTo,
        And,
        Reference,
        AlwaysTrue,
    )
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None  # type: ignore[assignment, unused-ignore]
    IcebergTable = None  # type: ignore
    GreaterThan = None  # type: ignore
    LessThan = None  # type: ignore
    GreaterThanOrEqual = None  # type: ignore
    LessThanOrEqual = None  # type: ignore
    EqualTo = None  # type: ignore
    And = None  # type: ignore
    Reference = None  # type: ignore
    AlwaysTrue = None  # type: ignore


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
    def mock_iceberg_table(self, sample_data: Any) -> Mock:
        """Mock Iceberg table whose scan returns sample_data unfiltered; filtering happens in the PyArrow re-pass."""
        mock_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_scan.to_arrow.return_value = sample_data
        mock_table.scan.return_value = mock_scan
        return mock_table

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
                lambda: LessThan(Reference("age"), 50),
                id="max_complex_exclusive",
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
                lambda: And(GreaterThanOrEqual(Reference("age"), 25), LessThan(Reference("age"), 50)),
                id="range_exclusive",
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

    def test_build_iceberg_expression_categorical_inclusion_raises(self) -> None:
        """Unknown filter types must raise NotImplementedError, not silently fall through to None."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": ["A", "B"]}
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
                GreaterThanOrEqual(Reference("age"), 30) if GreaterThanOrEqual is not None else None,
                [2, 3, 5],
                id="min_pushed_plus_categorical_inclusion",
            ),
            pytest.param(
                [
                    SingleFilter(Feature("age"), FilterType.MIN, {"value": 30}),
                    SingleFilter(Feature("name"), FilterType.REGEX, {"value": "^[A-C]"}),
                ],
                GreaterThanOrEqual(Reference("age"), 30) if GreaterThanOrEqual is not None else None,
                [2, 3],
                id="min_pushed_plus_regex",
            ),
            pytest.param(
                [SingleFilter(Feature("category"), FilterType.CATEGORICAL_INCLUSION, {"values": ["A", "B"]})],
                AlwaysTrue() if AlwaysTrue is not None else None,
                [1, 2, 3, 5],
                id="categorical_inclusion_only_no_pushdown",
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
        """Regex and categorical inclusion are not pushed but are applied on the scan result."""
        mock_feature_set.filters = filters

        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        mock_iceberg_table.scan.assert_called_once()
        call_args = mock_iceberg_table.scan.call_args
        assert call_args.kwargs["row_filter"] == expected_row_filter
        assert self.get_column_values(result, "id") == expected_ids

    def test_apply_filters_custom_filter_type_raises(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """A filter type with no Iceberg pushdown and no PyArrow method reaches do_custom_filter."""
        custom_filter = SingleFilter(Feature("age"), "custom_op", {"value": 1})
        mock_feature_set.filters = [custom_filter]

        with pytest.raises(NotImplementedError, match="Custom filtering is not supported"):
            IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

    def test_apply_filters_equal_missing_value_raises(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """apply_filters must not silently drop a missing-value filter and return the unfiltered table."""
        age_filter = SingleFilter(Feature("age"), FilterType.EQUAL, {"invalid": 30})
        mock_feature_set.filters = [age_filter]

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
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
