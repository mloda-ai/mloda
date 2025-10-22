"""
PyArrow transformer test scenarios for compute framework testing.

This module defines framework-agnostic test scenarios for PyArrow transformation operations.
Each scenario specifies input data structure, expected schema properties, and data characteristics.

The scenarios are designed to validate PyArrow table transformations across different compute
framework implementations (DuckDB, Polars, Spark, etc.).
"""

from typing import Any, Dict, Set, TypedDict

import pyarrow as pa


class TransformerScenario(TypedDict):
    """Type definition for a transformer test scenario.

    Attributes:
        data: Input data as PyArrow-compatible dictionary
        description: Human-readable description of the scenario
        expected_rows: Expected number of rows in the table
        expected_columns: Expected number of columns in the table
        expected_column_names: Expected column names as a set
        has_nulls: Whether the data contains null values
        data_types: Dictionary mapping column names to their type descriptions
    """

    data: Dict[str, Any]
    description: str
    expected_rows: int
    expected_columns: int
    expected_column_names: Set[str]
    has_nulls: bool
    data_types: Dict[str, str]


# Standard test data for basic transformation scenarios
BASIC_DATA = {
    "int_col": [1, 2, 3, 4],
    "str_col": ["alice", "bob", "charlie", "david"],
    "float_col": [1.1, 2.2, 3.3, 4.4],
    "bool_col": [True, False, True, False],
}

# Empty table with schema definition
EMPTY_DATA_WITH_SCHEMA = {
    "int_col": pa.array([], type=pa.int64()),
    "str_col": pa.array([], type=pa.string()),
    "float_col": pa.array([], type=pa.float64()),
}

# Data with null values in multiple columns
DATA_WITH_NULLS = {
    "int_col": [1, None, 3, None, 5],
    "str_col": ["alice", None, "charlie", "david", None],
    "float_col": [1.1, 2.2, None, 4.4, 5.5],
    "bool_col": [True, None, False, None, True],
}

# Large dataset for performance testing
LARGE_DATA = {
    "id": list(range(1000)),
    "value": [f"value_{i}" for i in range(1000)],
    "score": [i * 0.1 for i in range(1000)],
    "active": [i % 2 == 0 for i in range(1000)],
}

# Schema preservation with specific typed columns
TYPED_SCHEMA_DATA = {
    "int8_col": pa.array([1, 2, 3], type=pa.int8()),
    "int32_col": pa.array([100, 200, 300], type=pa.int32()),
    "int64_col": pa.array([1000, 2000, 3000], type=pa.int64()),
    "float32_col": pa.array([1.1, 2.2, 3.3], type=pa.float32()),
    "float64_col": pa.array([10.1, 20.2, 30.3], type=pa.float64()),
    "string_col": pa.array(["x", "y", "z"], type=pa.string()),
    "bool_col": pa.array([True, False, True], type=pa.bool_()),
}


SCENARIOS: Dict[str, TransformerScenario] = {
    "basic_transformation": {
        "data": BASIC_DATA,
        "description": "Basic transformation with mixed types (int, str, float, bool)",
        "expected_rows": 4,
        "expected_columns": 4,
        "expected_column_names": {"int_col", "str_col", "float_col", "bool_col"},
        "has_nulls": False,
        "data_types": {
            "int_col": "int64",
            "str_col": "string",
            "float_col": "float64",
            "bool_col": "bool",
        },
    },
    "empty_table": {
        "data": EMPTY_DATA_WITH_SCHEMA,
        "description": "Empty table with schema definition",
        "expected_rows": 0,
        "expected_columns": 3,
        "expected_column_names": {"int_col", "str_col", "float_col"},
        "has_nulls": False,
        "data_types": {
            "int_col": "int64",
            "str_col": "string",
            "float_col": "float64",
        },
    },
    "null_values": {
        "data": DATA_WITH_NULLS,
        "description": "Table with NULL values in multiple columns",
        "expected_rows": 5,
        "expected_columns": 4,
        "expected_column_names": {"int_col", "str_col", "float_col", "bool_col"},
        "has_nulls": True,
        "data_types": {
            "int_col": "int64",
            "str_col": "string",
            "float_col": "float64",
            "bool_col": "bool",
        },
    },
    "large_dataset": {
        "data": LARGE_DATA,
        "description": "1000 rows for performance testing",
        "expected_rows": 1000,
        "expected_columns": 4,
        "expected_column_names": {"id", "value", "score", "active"},
        "has_nulls": False,
        "data_types": {
            "id": "int64",
            "value": "string",
            "score": "float64",
            "active": "bool",
        },
    },
    "schema_preservation": {
        "data": TYPED_SCHEMA_DATA,
        "description": "Specific typed columns to verify schema preservation",
        "expected_rows": 3,
        "expected_columns": 7,
        "expected_column_names": {
            "int8_col",
            "int32_col",
            "int64_col",
            "float32_col",
            "float64_col",
            "string_col",
            "bool_col",
        },
        "has_nulls": False,
        "data_types": {
            "int8_col": "int8",
            "int32_col": "int32",
            "int64_col": "int64",
            "float32_col": "float32",
            "float64_col": "float64",
            "string_col": "string",
            "bool_col": "bool",
        },
    },
}
