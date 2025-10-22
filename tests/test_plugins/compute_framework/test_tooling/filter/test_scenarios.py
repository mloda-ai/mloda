"""
Filter test scenarios for compute framework testing.

This module defines framework-agnostic test scenarios for filter operations.
Each scenario specifies input data, filter parameters, and expected outputs.

The scenarios are designed to be converted to framework-specific formats
using the DataConverter and executed across all compute framework implementations.
"""

from typing import Any, Dict, List, TypedDict


class FilterScenario(TypedDict):
    """Type definition for a filter test scenario.

    Attributes:
        data: Input data as list of dictionaries
        feature_name: Name of the feature/column to filter on
        filter_type: Type of filter (range, min, max, equal, regex, categorical_inclusion)
        parameter: Filter parameters (varies by filter type)
        expected_row_count: Expected number of rows after filtering
        expected_ids: Expected IDs of rows that pass the filter
        description: Human-readable description of the scenario
    """

    data: List[Dict[str, Any]]
    feature_name: str
    filter_type: str
    parameter: Dict[str, Any]
    expected_row_count: int
    expected_ids: List[int]
    description: str


# Standard test data used across multiple scenarios
STANDARD_DATA = [
    {"id": 1, "age": 25, "name": "Alice", "category": "A", "score": 85.5, "is_active": True},
    {"id": 2, "age": 30, "name": "Bob", "category": "B", "score": 92.0, "is_active": False},
    {"id": 3, "age": 35, "name": "Charlie", "category": "A", "score": 78.5, "is_active": True},
    {"id": 4, "age": 40, "name": "David", "category": "C", "score": 88.0, "is_active": False},
    {"id": 5, "age": 45, "name": "Eve", "category": "B", "score": 95.5, "is_active": True},
]

# Data with null values for edge case testing
DATA_WITH_NULLS = [
    {"id": 1, "age": 25, "name": "Alice", "category": "A"},
    {"id": 2, "age": 30, "name": "Bob", "category": "B"},
    {"id": 3, "age": 35, "name": "Charlie", "category": "A"},
    {"id": 4, "age": 40, "name": "David", "category": "C"},
    {"id": 5, "age": 45, "name": "Eve", "category": "B"},
    {"id": 6, "age": None, "name": "Frank", "category": "A"},
]

# Empty data for edge case testing (with schema defined by providing empty rows with keys)
EMPTY_DATA: List[Dict[str, Any]] = []

# Data with schema but no rows (for frameworks that need schema)
EMPTY_DATA_WITH_SCHEMA: List[Dict[str, Any]] = [
    # Empty list means no rows, but converter will create schema from STANDARD_DATA structure
]


# Filter test scenarios organized by filter type
SCENARIOS: Dict[str, FilterScenario] = {
    # Range filter scenarios
    "range_inclusive": {
        "data": STANDARD_DATA,
        "feature_name": "age",
        "filter_type": "range",
        "parameter": {"min": 30, "max": 40, "max_exclusive": False},
        "expected_row_count": 3,
        "expected_ids": [2, 3, 4],
        "description": "Range filter with inclusive bounds (age between 30 and 40 inclusive)",
    },
    "range_exclusive": {
        "data": STANDARD_DATA,
        "feature_name": "age",
        "filter_type": "range",
        "parameter": {"min": 30, "max": 40, "max_exclusive": True},
        "expected_row_count": 2,
        "expected_ids": [2, 3],
        "description": "Range filter with exclusive max bound (age between 30 and 40 exclusive)",
    },
    # Min filter scenarios
    "min_basic": {
        "data": STANDARD_DATA,
        "feature_name": "age",
        "filter_type": "min",
        "parameter": {"value": 40},
        "expected_row_count": 2,
        "expected_ids": [4, 5],
        "description": "Min filter (age >= 40)",
    },
    # Max filter scenarios
    "max_basic": {
        "data": STANDARD_DATA,
        "feature_name": "age",
        "filter_type": "max",
        "parameter": {"value": 30},
        "expected_row_count": 2,
        "expected_ids": [1, 2],
        "description": "Max filter (age <= 30)",
    },
    "max_exclusive": {
        "data": STANDARD_DATA,
        "feature_name": "age",
        "filter_type": "max",
        "parameter": {"max": 35, "max_exclusive": True},
        "expected_row_count": 2,
        "expected_ids": [1, 2],
        "description": "Max filter with exclusive bound (age < 35)",
    },
    # Equal filter scenarios
    "equal_numeric": {
        "data": STANDARD_DATA,
        "feature_name": "age",
        "filter_type": "equal",
        "parameter": {"value": 35},
        "expected_row_count": 1,
        "expected_ids": [3],
        "description": "Equal filter on numeric column (age == 35)",
    },
    "equal_string": {
        "data": STANDARD_DATA,
        "feature_name": "category",
        "filter_type": "equal",
        "parameter": {"value": "A"},
        "expected_row_count": 2,
        "expected_ids": [1, 3],
        "description": "Equal filter on string column (category == 'A')",
    },
    "equal_boolean": {
        "data": STANDARD_DATA,
        "feature_name": "is_active",
        "filter_type": "equal",
        "parameter": {"value": True},
        "expected_row_count": 3,
        "expected_ids": [1, 3, 5],
        "description": "Equal filter on boolean column (is_active == True)",
    },
    # Regex filter scenarios
    "regex_start": {
        "data": STANDARD_DATA,
        "feature_name": "name",
        "filter_type": "regex",
        "parameter": {"value": "^A"},
        "expected_row_count": 1,
        "expected_ids": [1],
        "description": "Regex filter for names starting with 'A'",
    },
    "regex_complex": {
        "data": STANDARD_DATA,
        "feature_name": "name",
        "filter_type": "regex",
        "parameter": {"value": "^[A-C]"},
        "expected_row_count": 3,
        "expected_ids": [1, 2, 3],  # Alice, Bob, Charlie
        "description": "Regex filter for names starting with A, B, or C",
    },
    # Categorical inclusion filter scenarios
    "categorical_basic": {
        "data": STANDARD_DATA,
        "feature_name": "category",
        "filter_type": "categorical_inclusion",
        "parameter": {"values": ["A", "B"]},
        "expected_row_count": 4,
        "expected_ids": [1, 2, 3, 5],
        "description": "Categorical inclusion filter (category in ['A', 'B'])",
    },
    "categorical_single": {
        "data": STANDARD_DATA,
        "feature_name": "category",
        "filter_type": "categorical_inclusion",
        "parameter": {"values": ["C"]},
        "expected_row_count": 1,
        "expected_ids": [4],
        "description": "Categorical inclusion filter with single value (category in ['C'])",
    },
    # Edge case: null values
    "min_with_nulls": {
        "data": DATA_WITH_NULLS,  # type: ignore[typeddict-item]
        "feature_name": "age",
        "filter_type": "min",
        "parameter": {"value": 30},
        "expected_row_count": 4,
        "expected_ids": [2, 3, 4, 5],
        "description": "Min filter with null values (should exclude nulls)",
    },
    # Edge case: empty result
    "equal_no_match": {
        "data": STANDARD_DATA,
        "feature_name": "age",
        "filter_type": "equal",
        "parameter": {"value": 999},
        "expected_row_count": 0,
        "expected_ids": [],
        "description": "Equal filter with no matching rows",
    },
    # Edge case: empty data
    "min_empty_data": {
        "data": EMPTY_DATA,
        "feature_name": "age",
        "filter_type": "min",
        "parameter": {"value": 30},
        "expected_row_count": 0,
        "expected_ids": [],
        "description": "Min filter on empty dataset",
    },
}
