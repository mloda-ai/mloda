"""
Examples of structural validator usage.

This module demonstrates how to use structural validators for testing.
"""

import pandas as pd
import numpy as np

from tests.test_plugins.feature_group.test_tooling.data_generation.generators import (
    DataGenerator,
    EdgeCaseGenerator,
)
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_row_count,
    validate_columns_exist,
    validate_column_count,
    validate_no_nulls,
    validate_has_nulls,
    validate_column_types,
    validate_shape,
    validate_not_empty,
    validate_value_range,
)


def example_basic_validation() -> None:
    """Example: Basic structural validation."""
    print("\n=== Basic Structural Validation ===")

    # Create test DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"], "col3": [1.1, 2.2, 3.3, 4.4, 5.5]})

    # Validate row count
    validate_row_count(df, 5)
    print("✓ Row count validation passed (5 rows)")

    # Validate columns exist
    validate_columns_exist(df, ["col1", "col2"])
    print("✓ Column existence validation passed")

    # Validate column count
    validate_column_count(df, 3)
    print("✓ Column count validation passed (3 columns)")

    # Validate shape
    validate_shape(df, (5, 3))
    print("✓ Shape validation passed (5, 3)")

    # Validate not empty
    validate_not_empty(df)
    print("✓ Not empty validation passed")


def example_null_validation() -> None:
    """Example: Validate null values."""
    print("\n=== Null Value Validation ===")

    # Data without nulls
    data_no_nulls = DataGenerator.generate_numeric_columns(n_rows=10, column_names=["col1", "col2"], seed=42)
    df_no_nulls = pd.DataFrame(data_no_nulls)

    validate_no_nulls(df_no_nulls, ["col1", "col2"])
    print("✓ No nulls validation passed")

    # Data with nulls
    data_with_nulls = EdgeCaseGenerator.with_nulls(data_no_nulls, columns=["col1"], null_percentage=0.3, seed=42)
    df_with_nulls = pd.DataFrame(data_with_nulls)

    validate_has_nulls(df_with_nulls, ["col1"])
    print("✓ Has nulls validation passed")

    # Validate col2 still has no nulls
    validate_no_nulls(df_with_nulls, ["col2"])
    print("✓ Column without nulls validation passed")


def example_type_validation() -> None:
    """Example: Validate column data types."""
    print("\n=== Column Type Validation ===")

    df = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3], "str_col": ["a", "b", "c"]})

    # Validate column types
    validate_column_types(df, {"int_col": np.int64, "float_col": np.float64, "str_col": object})
    print("✓ Column type validation passed")


def example_value_range_validation() -> None:
    """Example: Validate value ranges."""
    print("\n=== Value Range Validation ===")

    # Generate data with specific range
    data = DataGenerator.generate_numeric_columns(n_rows=100, column_names=["value"], value_range=(10, 50), seed=42)
    df = pd.DataFrame(data)

    # Validate values are within expected range
    validate_value_range(df, "value", min_value=10, max_value=50)
    print("✓ Value range validation passed (10-50)")

    # Validate only minimum
    validate_value_range(df, "value", min_value=0)
    print("✓ Minimum value validation passed (>= 0)")

    # Validate only maximum
    validate_value_range(df, "value", max_value=100)
    print("✓ Maximum value validation passed (<= 100)")


def example_validation_with_custom_messages() -> None:
    """Example: Use custom error messages."""
    print("\n=== Custom Error Messages ===")

    df = pd.DataFrame({"col1": [1, 2, 3]})

    # Validate with custom message
    validate_row_count(df, 3, message="Expected exactly 3 rows for test case")
    print("✓ Validation with custom message passed")

    # Show what a failure looks like
    print("\nExample of validation failure with custom message:")
    try:
        validate_row_count(df, 5, message="Custom error: Expected 5 rows but got different count")
    except AssertionError as e:
        print(f"  AssertionError: {e}")


def example_edge_case_validation() -> None:
    """Example: Validate edge cases."""
    print("\n=== Edge Case Validation ===")

    # Empty data
    empty_data = EdgeCaseGenerator.empty_data(["col1", "col2"])
    df_empty = pd.DataFrame(empty_data)

    validate_row_count(df_empty, 0)
    validate_columns_exist(df_empty, ["col1", "col2"])
    print("✓ Empty data validation passed")

    # Single row
    single_data = EdgeCaseGenerator.single_row(["col1", "col2"], value=42)
    df_single = pd.DataFrame(single_data)

    validate_row_count(df_single, 1)
    validate_shape(df_single, (1, 2))
    print("✓ Single row validation passed")

    # All nulls
    all_nulls_data = EdgeCaseGenerator.all_nulls(["col1"], n_rows=5)
    df_all_nulls = pd.DataFrame(all_nulls_data)

    validate_row_count(df_all_nulls, 5)
    validate_has_nulls(df_all_nulls, ["col1"])
    print("✓ All nulls validation passed")


def example_chained_validation() -> None:
    """Example: Chain multiple validations together."""
    print("\n=== Chained Validation Pattern ===")

    # Generate test data
    data = DataGenerator.generate_data(
        n_rows=50, numeric_cols=["value1", "value2"], categorical_cols=["category"], seed=42
    )
    df = pd.DataFrame(data)

    # Chain multiple validations
    validate_not_empty(df)
    validate_row_count(df, 50)
    validate_column_count(df, 3)
    validate_columns_exist(df, ["value1", "value2", "category"])
    validate_no_nulls(df, ["value1", "value2", "category"])
    validate_shape(df, (50, 3))

    print("✓ All chained validations passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Structural Validator Examples")
    print("=" * 60)

    example_basic_validation()
    example_null_validation()
    example_type_validation()
    example_value_range_validation()
    example_validation_with_custom_messages()
    example_edge_case_validation()
    example_chained_validation()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
