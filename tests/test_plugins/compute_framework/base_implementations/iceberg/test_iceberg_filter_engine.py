import pytest
from unittest.mock import Mock, patch

from mloda import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_filter_engine import IcebergFilterEngine

import logging

logger = logging.getLogger(__name__)

try:
    import pyiceberg
    import pyarrow as pa
    from pyiceberg.table import Table as IcebergTable
    from pyiceberg.expressions import GreaterThan, LessThan, GreaterThanOrEqual, LessThanOrEqual, EqualTo, And
except ImportError:
    logger.warning("PyIceberg or PyArrow is not installed. Some tests will be skipped.")
    pyiceberg = None  # type: ignore
    pa = None
    IcebergTable = None  # type: ignore
    GreaterThan = None  # type: ignore
    LessThan = None  # type: ignore
    GreaterThanOrEqual = None  # type: ignore
    LessThanOrEqual = None  # type: ignore
    EqualTo = None  # type: ignore
    And = None  # type: ignore


@pytest.mark.skipif(
    pyiceberg is None or pa is None, reason="PyIceberg or PyArrow is not installed. Skipping this test."
)
class TestIcebergFilterEngine:
    """Unit tests for the IcebergFilterEngine class."""

    @pytest.fixture
    def mock_iceberg_table(self) -> Mock:
        """Create a mock Iceberg table for testing."""
        mock_table = Mock(spec=IcebergTable)
        mock_scan = Mock()
        mock_table.scan.return_value = mock_scan
        return mock_table

    @pytest.fixture
    def mock_feature_set(self) -> Mock:
        """Create a mock feature set for testing."""
        mock_feature_set = Mock()
        mock_feature_set.get_all_names.return_value = ["age", "name", "category"]
        return mock_feature_set

    def test_final_filters(self) -> None:
        """Test that final_filters returns False for Iceberg (predicate pushdown)."""
        assert IcebergFilterEngine.final_filters() is False

    def test_build_iceberg_expression_equal(self) -> None:
        """Test building equal filter expression."""
        feature = Feature("age")
        filter_type = FilterType.equal
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)

        # Since we can't easily test the actual Iceberg expression object,
        # we'll test that it's not None (meaning it was created successfully)
        assert expression is not None

    def test_build_iceberg_expression_min(self) -> None:
        """Test building min filter expression."""
        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 25}
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
        assert expression is not None

    def test_build_iceberg_expression_max_simple(self) -> None:
        """Test building max filter expression with simple parameter."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"value": 50}
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
        assert expression is not None

    def test_build_iceberg_expression_max_complex(self) -> None:
        """Test building max filter expression with complex parameter."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"max": 50, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
        assert expression is not None

    def test_build_iceberg_expression_range(self) -> None:
        """Test building range filter expression."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 25, "max": 50, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
        assert expression is not None

    def test_build_iceberg_expression_range_exclusive(self) -> None:
        """Test building range filter expression with exclusive max."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 25, "max": 50, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
        assert expression is not None

    def test_build_iceberg_expression_unsupported(self) -> None:
        """Test building expression for unsupported filter type."""
        feature = Feature("name")
        filter_type = FilterType.regex
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
        assert expression is None

    def test_extract_parameter_value(self) -> None:
        """Test extracting parameter values."""
        feature = Feature("age")
        filter_type = FilterType.equal
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
        filter_type = FilterType.max
        parameter = {"max": 50, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        assert IcebergFilterEngine._has_parameter(single_filter, "max") is True
        assert IcebergFilterEngine._has_parameter(single_filter, "max_exclusive") is True
        assert IcebergFilterEngine._has_parameter(single_filter, "missing") is False

    def test_apply_filters_iceberg_table(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """Test applying filters to Iceberg table."""
        # Create filters
        age_filter = SingleFilter(Feature("age"), FilterType.min, {"value": 25})
        mock_feature_set.filters = [age_filter]

        # Apply filters
        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        # Verify that scan was called with a filter
        mock_iceberg_table.scan.assert_called_once()
        call_args = mock_iceberg_table.scan.call_args
        assert "row_filter" in call_args.kwargs

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
        age_filter = SingleFilter(Feature("age"), FilterType.min, {"value": 25})
        name_filter = SingleFilter(Feature("name"), FilterType.equal, {"value": "Alice"})
        mock_feature_set.filters = [age_filter, name_filter]

        # Apply filters
        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        # Verify that scan was called with combined filter
        mock_iceberg_table.scan.assert_called_once()
        call_args = mock_iceberg_table.scan.call_args
        assert "row_filter" in call_args.kwargs

    def test_apply_filters_filtered_features(self, mock_iceberg_table: Mock, mock_feature_set: Mock) -> None:
        """Test applying filters where some features are not in the feature set."""
        # Create filter for feature not in feature set
        unknown_filter = SingleFilter(Feature("unknown_column"), FilterType.equal, {"value": "test"})
        age_filter = SingleFilter(Feature("age"), FilterType.min, {"value": 25})
        mock_feature_set.filters = [unknown_filter, age_filter]

        # Apply filters
        result = IcebergFilterEngine.apply_filters(mock_iceberg_table, mock_feature_set)

        # Should only apply the age filter (unknown_column is not in get_all_names)
        mock_iceberg_table.scan.assert_called_once()

    def test_standard_filter_methods_not_implemented(self) -> None:
        """Test that standard filter methods raise NotImplementedError."""
        mock_data = Mock()
        mock_filter = Mock()

        with pytest.raises(NotImplementedError, match="Use apply_filters method"):
            IcebergFilterEngine.do_range_filter(mock_data, mock_filter)

        with pytest.raises(NotImplementedError, match="Use apply_filters method"):
            IcebergFilterEngine.do_min_filter(mock_data, mock_filter)

        with pytest.raises(NotImplementedError, match="Use apply_filters method"):
            IcebergFilterEngine.do_max_filter(mock_data, mock_filter)

        with pytest.raises(NotImplementedError, match="Use apply_filters method"):
            IcebergFilterEngine.do_equal_filter(mock_data, mock_filter)

    def test_unsupported_filter_methods(self) -> None:
        """Test that unsupported filter methods raise NotImplementedError."""
        mock_data = Mock()
        mock_filter = Mock()

        with pytest.raises(NotImplementedError, match="Regex filtering is not supported"):
            IcebergFilterEngine.do_regex_filter(mock_data, mock_filter)

        with pytest.raises(NotImplementedError, match="Categorical inclusion filtering is not yet implemented"):
            IcebergFilterEngine.do_categorical_inclusion_filter(mock_data, mock_filter)

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
            filter_type = FilterType.equal
            parameter = {"value": 30}
            single_filter = SingleFilter(feature, filter_type, parameter)

            expression = IcebergFilterEngine._build_iceberg_expression(single_filter)
            assert expression is None
