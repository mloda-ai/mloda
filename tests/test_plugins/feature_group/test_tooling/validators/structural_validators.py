"""
Business-agnostic structural validators.

Validate data structure properties: shape, columns, types, nulls.
NO business logic.
"""

from typing import Any, List, Optional, Dict, Type
import pandas as pd


def validate_row_count(result: Any, expected_count: int, message: Optional[str] = None) -> None:
    """
    Validate number of rows.

    Args:
        result: Result data (DataFrame, Table, etc.)
        expected_count: Expected row count
        message: Optional custom error message
    """
    if hasattr(result, "__len__"):
        actual = len(result)
    elif hasattr(result, "num_rows"):
        actual = result.num_rows
    else:
        raise ValueError(f"Cannot get row count from type: {type(result)}")

    if actual != expected_count:
        error_msg = message or f"Expected {expected_count} rows, got {actual}"
        raise AssertionError(error_msg)


def validate_columns_exist(result: Any, expected_columns: List[str], message: Optional[str] = None) -> None:
    """
    Validate that columns exist.

    Args:
        result: Result data
        expected_columns: Expected column names
        message: Optional custom error message
    """
    if hasattr(result, "columns"):
        actual_columns = set(result.columns)
    elif hasattr(result, "schema"):
        actual_columns = set(result.schema.names)
    else:
        raise ValueError(f"Cannot get columns from type: {type(result)}")

    missing = set(expected_columns) - actual_columns
    if missing:
        error_msg = message or f"Missing columns: {missing}. Available: {actual_columns}"
        raise AssertionError(error_msg)


def validate_column_count(result: Any, expected_count: int, message: Optional[str] = None) -> None:
    """
    Validate number of columns.

    Args:
        result: Result data
        expected_count: Expected column count
        message: Optional custom error message
    """
    if hasattr(result, "columns"):
        actual = len(result.columns)
    elif hasattr(result, "schema"):
        actual = len(result.schema.names)
    else:
        raise ValueError(f"Cannot get column count from type: {type(result)}")

    if actual != expected_count:
        error_msg = message or f"Expected {expected_count} columns, got {actual}"
        raise AssertionError(error_msg)


def validate_no_nulls(result: Any, columns: List[str], message: Optional[str] = None) -> None:
    """
    Validate no null values in specified columns.

    Args:
        result: Result data
        columns: Columns to check
        message: Optional custom error message
    """
    df = result.to_pandas() if hasattr(result, "to_pandas") else result

    for col in columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            error_msg = message or f"Column '{col}' has {null_count} null values"
            raise AssertionError(error_msg)


def validate_has_nulls(result: Any, columns: List[str], message: Optional[str] = None) -> None:
    """
    Validate that specified columns have some null values.

    Args:
        result: Result data
        columns: Columns that should have nulls
        message: Optional custom error message
    """
    df = result.to_pandas() if hasattr(result, "to_pandas") else result

    for col in columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            error_msg = message or f"Column '{col}' has no null values (expected some)"
            raise AssertionError(error_msg)


def validate_column_types(result: Any, expected_types: Dict[str, Type[Any]], message: Optional[str] = None) -> None:
    """
    Validate column data types.

    Args:
        result: Result data
        expected_types: Dict mapping column names to expected types
        message: Optional custom error message

    Example:
        >>> import numpy as np
        >>> validate_column_types(df, {"col1": np.float64, "col2": object})
    """
    df = result.to_pandas() if hasattr(result, "to_pandas") else result

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            raise AssertionError(f"Column '{col}' not found")

        actual_type = df[col].dtype
        if not pd.api.types.is_dtype_equal(actual_type, expected_type):
            error_msg = message or f"Column '{col}' has type {actual_type}, expected {expected_type}"
            raise AssertionError(error_msg)


def validate_shape(result: Any, expected_shape: tuple[int, int], message: Optional[str] = None) -> None:
    """
    Validate data shape (rows, columns).

    Args:
        result: Result data
        expected_shape: (n_rows, n_cols)
        message: Optional custom error message
    """
    if hasattr(result, "shape"):
        actual_shape = result.shape
    elif hasattr(result, "num_rows"):
        actual_shape = (result.num_rows, len(result.schema.names))
    else:
        raise ValueError(f"Cannot get shape from type: {type(result)}")

    if actual_shape != expected_shape:
        error_msg = message or f"Expected shape {expected_shape}, got {actual_shape}"
        raise AssertionError(error_msg)


def validate_not_empty(result: Any, message: Optional[str] = None) -> None:
    """
    Validate data is not empty.

    Args:
        result: Result data
        message: Optional custom error message
    """
    if hasattr(result, "__len__"):
        is_empty = len(result) == 0
    elif hasattr(result, "num_rows"):
        is_empty = result.num_rows == 0
    else:
        raise ValueError(f"Cannot check emptiness for type: {type(result)}")

    if is_empty:
        error_msg = message or "Data is empty"
        raise AssertionError(error_msg)


def validate_value_range(
    result: Any,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """
    Validate values fall within range.

    Args:
        result: Result data
        column: Column to check
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        message: Optional custom error message
    """
    df = result.to_pandas() if hasattr(result, "to_pandas") else result

    if column not in df.columns:
        raise AssertionError(f"Column '{column}' not found")

    if min_value is not None:
        actual_min = df[column].min()
        if actual_min < min_value:
            error_msg = message or f"Column '{column}' min {actual_min} < {min_value}"
            raise AssertionError(error_msg)

    if max_value is not None:
        actual_max = df[column].max()
        if actual_max > max_value:
            error_msg = message or f"Column '{column}' max {actual_max} > {max_value}"
            raise AssertionError(error_msg)
