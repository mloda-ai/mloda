"""
Base test class for multi-index merge engine testing.

This module provides a reusable base class that implements common test logic
for multi-index merge operations across all compute frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import logging

import pytest

from mloda.user import Index
from mloda.provider import BaseMergeEngine
from mloda.core.abstract_plugins.components.link import JoinType

from .test_scenarios import SCENARIOS, MergeScenario
from .test_data_converter import DataConverter

logger = logging.getLogger(__name__)


class MultiIndexMergeEngineTestBase(ABC):
    """
    Base class for multi-index merge engine tests.

    Subclasses must implement:
    - merge_engine_class(): Return the MergeEngine class to test
    - convert_dict_to_framework(data): Convert dict data to framework format
    - get_connection(): Return framework connection object (or None)

    This base class provides:
    - All standard multi-index test methods (INNER, LEFT, OUTER, APPEND, UNION)
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
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        """Return the merge engine class for this framework."""
        pass

    @classmethod
    @abstractmethod
    def framework_type(cls) -> type[Any]:
        """Return the framework's expected data type (e.g., pd.DataFrame, pa.Table)."""
        pass

    @abstractmethod
    def get_connection(self) -> Optional[Any]:
        """Return framework connection object, or None if not needed."""
        pass

    # ==================== HELPER METHODS ====================

    def convert_dict_to_framework(self, data: list[dict[str, Any]]) -> Any:
        """Convert List[Dict] to this framework's native format using the converter."""
        return self.converter.to_framework(data, self.framework_type(), self.get_connection())

    def convert_framework_to_dict(self, data: Any) -> list[dict[str, Any]]:
        """Convert framework data back to List[Dict] for assertions."""
        return self.converter.from_framework(data, self.framework_type())

    def _run_merge_test(
        self,
        scenario_key: str,
        merge_method: str,
        additional_assertions: Optional[Callable[[list[dict[str, Any]], MergeScenario], None]] = None,
    ) -> None:
        """
        Run a merge test using a predefined scenario.

        Args:
            scenario_key: Key in SCENARIOS dict
            merge_method: Method name on merge engine (e.g., 'merge_inner')
            additional_assertions: Optional function for framework-specific assertions
        """
        scenario = SCENARIOS[scenario_key]

        # Convert dict data to framework format
        left_data = self.convert_dict_to_framework(scenario["left"])
        right_data = self.convert_dict_to_framework(scenario["right"])

        # Create multi-index
        index = Index(scenario["index"])

        # Initialize merge engine
        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        # Execute merge operation
        result = getattr(engine, merge_method)(left_data, right_data, index, index)

        # Convert result back to dict for assertions
        result_dicts = self.convert_framework_to_dict(result)

        # Run standard assertions
        self._assert_row_count(result_dicts, scenario["expected_rows"])
        self._assert_columns_exist(result_dicts, scenario["expected_columns"])

        # Run any additional custom assertions
        if additional_assertions:
            additional_assertions(result_dicts, scenario)

    def _assert_row_count(self, result: list[dict[str, Any]], expected: int) -> None:
        """Assert that result has expected number of rows."""
        actual = len(result)
        assert actual == expected, f"Expected {expected} rows, got {actual}"

    def _assert_columns_exist(self, result: list[dict[str, Any]], expected_columns: list[str]) -> None:
        """Assert that all expected columns exist in result."""
        if not result:
            return  # Empty result, skip column check

        actual_columns = set(result[0].keys())
        for col in expected_columns:
            assert col in actual_columns, f"Expected column '{col}' not found in result. Available: {actual_columns}"

    def _assert_has_null_values(self, result: list[dict[str, Any]], column: str) -> bool:
        """Check if any row has null value in specified column."""
        import math

        for row in result:
            value = row.get(column)
            if value is None:
                return True
            if isinstance(value, float) and math.isnan(value):
                return True
        return False

    # ==================== TEST METHODS ====================

    def test_merge_inner_with_multi_index(self) -> None:
        """Test INNER join with 2-column multi-index."""

        def additional_checks(result: list[dict[str, Any]], scenario: MergeScenario) -> None:
            # All rows should have values in both left and right columns
            for row in result:
                assert row.get("left_value") is not None, "INNER join should not have null left values"
                assert row.get("right_value") is not None, "INNER join should not have null right values"

        self._run_merge_test("inner_basic", "merge_inner", additional_checks)

    def test_merge_left_with_multi_index(self) -> None:
        """Test LEFT join with 2-column multi-index."""

        def additional_checks(result: list[dict[str, Any]], scenario: MergeScenario) -> None:
            # All rows should have left_value, but some may have null right_value
            for row in result:
                assert row.get("left_value") is not None, "LEFT join should have all left values"
            # Should have at least some null right values
            assert self._assert_has_null_values(result, "right_value"), "LEFT join should have some null right values"

        self._run_merge_test("left_with_unmatched", "merge_left", additional_checks)

    def test_merge_full_outer_with_multi_index(self) -> None:
        """Test OUTER join with 2-column multi-index."""

        def additional_checks(result: list[dict[str, Any]], scenario: MergeScenario) -> None:
            # Should have nulls on both sides
            assert self._assert_has_null_values(result, "left_value"), "OUTER join should have some null left values"
            assert self._assert_has_null_values(result, "right_value"), "OUTER join should have some null right values"

        self._run_merge_test("outer_both_sides", "merge_full_outer", additional_checks)

    def test_merge_append_with_multi_index(self) -> None:
        """Test APPEND with 2-column multi-index."""

        def additional_checks(result: list[dict[str, Any]], scenario: MergeScenario) -> None:
            # All rows should have values (no nulls in APPEND)
            for row in result:
                assert row.get("value") is not None, "APPEND should not have null values"

        self._run_merge_test("append_basic", "merge_append", additional_checks)

    def test_merge_union_with_multi_index(self) -> None:
        """Test UNION with 2-column multi-index."""

        def additional_checks(result: list[dict[str, Any]], scenario: MergeScenario) -> None:
            # All rows should have values (no nulls in UNION)
            for row in result:
                assert row.get("value") is not None, "UNION should not have null values"
            # Should have removed duplicates
            assert len(result) < len(scenario["left"]) + len(scenario["right"]), "UNION should remove duplicates"

        self._run_merge_test("union_with_duplicates", "merge_union", additional_checks)

    def test_mismatched_index_lengths_raises(self) -> None:
        """Test that mismatched index lengths raise ValueError in merge()."""
        left_index = Index(("a", "b"))
        right_index = Index(("x",))

        left_data = self.convert_dict_to_framework(
            [{"a": 1, "b": 2, "val": 10}],
        )
        right_data = self.convert_dict_to_framework(
            [{"x": 1, "val": 20}],
        )

        connection = self.get_connection()
        engine = self.merge_engine_class()(connection) if connection else self.merge_engine_class()()

        with pytest.raises(ValueError):
            engine.merge(left_data, right_data, JoinType.INNER, left_index, right_index)
