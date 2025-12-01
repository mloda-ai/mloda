"""
Base test class for filter engine testing.

This module provides a reusable base class that implements common test logic
for filter operations across all compute frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import logging

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_core.filter.filter_engine import BaseFilterEngine

from ..multi_index.test_data_converter import DataConverter
from .test_scenarios import SCENARIOS, FilterScenario

logger = logging.getLogger(__name__)


class FilterEngineTestBase(ABC):
    """
    Base class for filter engine tests.

    Subclasses must implement:
    - filter_engine_class(): Return the FilterEngine class to test
    - framework_type(): Return the framework's data type
    - get_connection(): Return framework connection object (or None)

    This base class provides:
    - All standard filter test methods (range, min, max, equal, regex, categorical)
    - Framework-agnostic test logic using shared scenarios
    - Automatic conversion between dict and framework formats
    - Common assertion logic
    """

    def setup_method(self) -> None:
        """Initialize the test base with a data converter before each test method."""
        self.converter = DataConverter()

    # ==================== ABSTRACT METHODS (must implement) ====================

    @classmethod
    @abstractmethod
    def filter_engine_class(cls) -> Type[BaseFilterEngine]:
        """Return the filter engine class for this framework."""
        pass

    @classmethod
    @abstractmethod
    def framework_type(cls) -> Type[Any]:
        """Return the framework's expected data type (e.g., pd.DataFrame, pa.Table)."""
        pass

    @abstractmethod
    def get_connection(self) -> Optional[Any]:
        """Return framework connection object, or None if not needed."""
        pass

    # ==================== HELPER METHODS ====================

    def convert_dict_to_framework(self, data: List[Dict[str, Any]]) -> Any:
        """Convert List[Dict] to this framework's native format using the converter."""
        return self.converter.to_framework(data, self.framework_type(), self.get_connection())

    def convert_framework_to_dict(self, data: Any) -> List[Dict[str, Any]]:
        """Convert framework data back to List[Dict] for assertions."""
        return self.converter.from_framework(data, self.framework_type())

    def _run_filter_test(self, scenario_key: str, filter_method: str) -> None:
        """
        Run a filter test using a predefined scenario.

        Args:
            scenario_key: Key in SCENARIOS dict
            filter_method: Method name on filter engine (e.g., 'do_range_filter')
        """
        scenario = SCENARIOS[scenario_key]

        # Convert dict data to framework format
        data = self.convert_dict_to_framework(scenario["data"])

        # Create filter
        feature = Feature(scenario["feature_name"])
        filter_type = FilterTypeEnum[scenario["filter_type"]]
        single_filter = SingleFilter(feature, filter_type, scenario["parameter"])

        # Execute filter operation
        engine_class = self.filter_engine_class()
        result = getattr(engine_class, filter_method)(data, single_filter)

        # Convert result back to dict for assertions
        result_dict = self.convert_framework_to_dict(result)

        # Validate results
        self._assert_row_count(result_dict, scenario["expected_row_count"])
        self._assert_ids_match(result_dict, scenario["expected_ids"])

    def _assert_row_count(self, result: List[Dict[str, Any]], expected: int) -> None:
        """Assert that the result has the expected number of rows."""
        assert len(result) == expected, f"Expected {expected} rows, got {len(result)}"

    def _assert_ids_match(self, result: List[Dict[str, Any]], expected_ids: List[int]) -> None:
        """Assert that the result contains rows with expected IDs."""
        actual_ids = sorted([row["id"] for row in result])
        expected_ids_sorted = sorted(expected_ids)
        assert actual_ids == expected_ids_sorted, f"Expected IDs {expected_ids_sorted}, got {actual_ids}"

    def _get_column_values(self, result: List[Dict[str, Any]], column: str) -> List[Any]:
        """Extract values from a specific column."""
        return [row[column] for row in result]

    # ==================== TEST METHODS ====================

    def test_do_range_filter(self) -> None:
        """Test range filter with inclusive bounds."""
        self._run_filter_test("range_inclusive", "do_range_filter")

    def test_do_range_filter_exclusive(self) -> None:
        """Test range filter with exclusive max bound."""
        self._run_filter_test("range_exclusive", "do_range_filter")

    def test_do_min_filter(self) -> None:
        """Test min filter."""
        self._run_filter_test("min_basic", "do_min_filter")

    def test_do_max_filter(self) -> None:
        """Test max filter with simple parameter."""
        self._run_filter_test("max_basic", "do_max_filter")

    def test_do_max_filter_with_tuple(self) -> None:
        """Test max filter with exclusive bound."""
        self._run_filter_test("max_exclusive", "do_max_filter")

    def test_do_equal_filter(self) -> None:
        """Test equal filter on numeric column."""
        self._run_filter_test("equal_numeric", "do_equal_filter")

    def test_do_equal_filter_string(self) -> None:
        """Test equal filter on string column."""
        self._run_filter_test("equal_string", "do_equal_filter")

    def test_do_equal_filter_boolean(self) -> None:
        """Test equal filter on boolean column."""
        self._run_filter_test("equal_boolean", "do_equal_filter")

    def test_do_regex_filter(self) -> None:
        """Test regex filter."""
        self._run_filter_test("regex_start", "do_regex_filter")

    def test_do_regex_filter_complex(self) -> None:
        """Test regex filter with complex pattern."""
        self._run_filter_test("regex_complex", "do_regex_filter")

    def test_do_categorical_inclusion_filter(self) -> None:
        """Test categorical inclusion filter."""
        self._run_filter_test("categorical_basic", "do_categorical_inclusion_filter")

    def test_do_categorical_inclusion_filter_single(self) -> None:
        """Test categorical inclusion filter with single value."""
        self._run_filter_test("categorical_single", "do_categorical_inclusion_filter")

    def test_filter_with_null_values(self) -> None:
        """Test filtering with null values in data."""
        self._run_filter_test("min_with_nulls", "do_min_filter")

    def test_filter_empty_result(self) -> None:
        """Test filter that produces empty result."""
        self._run_filter_test("equal_no_match", "do_equal_filter")

    def test_filter_with_empty_data(self) -> None:
        """Test filtering empty dataset."""
        self._run_filter_test("min_empty_data", "do_min_filter")

    def test_apply_filters(self) -> None:
        """Test applying multiple filters."""
        # Use standard data
        data = self.convert_dict_to_framework(SCENARIOS["range_inclusive"]["data"])

        # Create multiple filters
        filters = [
            SingleFilter(Feature("age"), FilterTypeEnum.min, {"value": 30}),
            SingleFilter(Feature("category"), FilterTypeEnum.equal, {"value": "A"}),
        ]

        # Create a mock feature set
        class MockFeatureSet:
            def __init__(self, filters: List[SingleFilter]) -> None:
                self.filters = filters

            def get_all_names(self) -> List[str]:
                return ["age", "category"]

        feature_set = MockFeatureSet(filters)

        # Apply the filters
        engine_class = self.filter_engine_class()
        result = engine_class.apply_filters(data, feature_set)

        # Convert result back to dict for assertions
        result_dict = self.convert_framework_to_dict(result)

        # Should only have Charlie (id=3: age=35, category='A')
        self._assert_row_count(result_dict, 1)
        self._assert_ids_match(result_dict, [3])

    def test_final_filters(self) -> None:
        """Test that final_filters returns expected value."""
        engine_class = self.filter_engine_class()
        # Most engines return True (filters applied after computation)
        # Iceberg is an exception (predicate pushdown, returns False)
        result = engine_class.final_filters()
        assert isinstance(result, bool), "final_filters() must return a boolean"
