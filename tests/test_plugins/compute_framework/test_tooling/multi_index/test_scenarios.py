"""
Reusable test scenarios for multi-index merge operations.

These scenarios are framework-agnostic and can be used by any compute framework
test by converting the dict format to the framework's native format.
"""

from typing import Any, Dict, List, TypedDict


class MergeScenario(TypedDict):
    """Type definition for a merge test scenario."""

    left: List[Dict[str, Any]]
    right: List[Dict[str, Any]]
    index: tuple[str, ...]
    expected_rows: int
    expected_columns: List[str]
    description: str


# Standard multi-index test scenarios
SCENARIOS: Dict[str, MergeScenario] = {
    "inner_basic": {
        "description": "Basic INNER join with all rows matching on 2-column index",
        "left": [
            {"id": 1, "category": "A", "left_value": 10},
            {"id": 2, "category": "B", "left_value": 20},
            {"id": 3, "category": "C", "left_value": 30},
        ],
        "right": [
            {"id": 1, "category": "A", "right_value": 100},
            {"id": 2, "category": "B", "right_value": 200},
            {"id": 3, "category": "C", "right_value": 300},
        ],
        "index": ("id", "category"),
        "expected_rows": 3,
        "expected_columns": ["id", "category", "left_value", "right_value"],
    },
    "left_with_unmatched": {
        "description": "LEFT join with some unmatched rows from left side",
        "left": [
            {"id": 1, "category": "A", "left_value": 10},
            {"id": 2, "category": "B", "left_value": 20},
            {"id": 3, "category": "C", "left_value": 30},
            {"id": 4, "category": "D", "left_value": 40},
        ],
        "right": [
            {"id": 1, "category": "A", "right_value": 100},
            {"id": 2, "category": "B", "right_value": 200},
            {"id": 5, "category": "E", "right_value": 500},
        ],
        "index": ("id", "category"),
        "expected_rows": 4,  # All 4 left rows
        "expected_columns": ["id", "category", "left_value", "right_value"],
    },
    "outer_both_sides": {
        "description": "OUTER join with unmatched rows on both sides",
        "left": [
            {"id": 1, "category": "A", "left_value": 10},
            {"id": 2, "category": "B", "left_value": 20},
            {"id": 3, "category": "C", "left_value": 30},
        ],
        "right": [
            {"id": 1, "category": "A", "right_value": 100},
            {"id": 4, "category": "D", "right_value": 400},
            {"id": 5, "category": "E", "right_value": 500},
        ],
        "index": ("id", "category"),
        "expected_rows": 5,  # 1 match + 2 left-only + 2 right-only
        "expected_columns": ["id", "category", "left_value", "right_value"],
    },
    "append_basic": {
        "description": "APPEND operation concatenating all rows",
        "left": [
            {"id": 1, "category": "A", "value": 10},
            {"id": 2, "category": "B", "value": 20},
        ],
        "right": [
            {"id": 3, "category": "C", "value": 30},
            {"id": 4, "category": "D", "value": 40},
        ],
        "index": ("id", "category"),
        "expected_rows": 4,  # All 4 rows concatenated
        "expected_columns": ["id", "category", "value"],
    },
    "union_with_duplicates": {
        "description": "UNION operation removing exact duplicate rows",
        "left": [
            {"id": 1, "category": "A", "value": 10},
            {"id": 2, "category": "B", "value": 20},
            {"id": 3, "category": "C", "value": 30},
        ],
        "right": [
            {"id": 1, "category": "A", "value": 10},  # Exact duplicate
            {"id": 4, "category": "D", "value": 40},
            {"id": 5, "category": "E", "value": 50},
        ],
        "index": ("id", "category"),
        "expected_rows": 5,  # 1 duplicate removed
        "expected_columns": ["id", "category", "value"],
    },
}
