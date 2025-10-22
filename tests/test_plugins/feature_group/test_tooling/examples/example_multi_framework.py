"""
Examples of multi-framework testing patterns.

This module demonstrates how to test feature groups across multiple compute frameworks.
"""

import pandas as pd

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_row_count,
    validate_columns_exist,
)


def example_convert_to_frameworks() -> None:
    """Example: Convert test data to different compute frameworks."""
    print("\n=== Converting to Different Frameworks ===")

    # Generate test data
    test_data = DataGenerator.generate_data(
        n_rows=10, numeric_cols=["value1", "value2"], categorical_cols=["category"], seed=42
    )

    converter = DataConverter()

    # Convert to pandas
    pandas_df = converter.to_framework(test_data, pd.DataFrame)
    print(f"Pandas DataFrame: {type(pandas_df)} with shape {pandas_df.shape}")

    # Convert to PyArrow (if available)
    if PYARROW_AVAILABLE:
        pyarrow_table = converter.to_framework(test_data, pa.Table)
        print(f"PyArrow Table: {type(pyarrow_table)} with {pyarrow_table.num_rows} rows")


def example_convert_between_frameworks() -> None:
    """Example: Convert between different frameworks."""
    print("\n=== Converting Between Frameworks ===")

    converter = DataConverter()

    # Start with pandas
    pandas_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    print(f"Original Pandas DataFrame: {pandas_df.shape}")

    # Convert to list of dicts
    dict_data = converter.from_framework(pandas_df, pd.DataFrame)
    print(f"Converted to List[Dict]: {len(dict_data)} rows")

    # Convert to PyArrow
    if PYARROW_AVAILABLE:
        pyarrow_table = converter.to_framework(dict_data, pa.Table)
        print(f"Converted to PyArrow Table: {pyarrow_table.num_rows} rows")

        # Convert back to pandas
        back_to_pandas = converter.to_pandas(pyarrow_table)
        print(f"Converted back to Pandas: {back_to_pandas.shape}")


def example_validate_across_frameworks() -> None:
    """Example: Validate data structure across different frameworks."""
    print("\n=== Validating Across Frameworks ===")

    # Generate test data
    test_data = DataGenerator.generate_data(n_rows=20, numeric_cols=["value1", "value2"], seed=42)

    converter = DataConverter()

    # Test with pandas
    pandas_df = converter.to_framework(test_data, pd.DataFrame)
    validate_row_count(pandas_df, 20)
    validate_columns_exist(pandas_df, ["value1", "value2"])
    print("✓ Pandas validation passed")

    # Test with PyArrow
    if PYARROW_AVAILABLE:
        pyarrow_table = converter.to_framework(test_data, pa.Table)
        validate_row_count(pyarrow_table, 20)
        validate_columns_exist(pyarrow_table, ["value1", "value2"])
        print("✓ PyArrow validation passed")


def example_framework_agnostic_test() -> None:
    """Example: Write a framework-agnostic test using parametrization."""
    print("\n=== Framework-Agnostic Testing Pattern ===")

    # This pattern would be used with pytest.mark.parametrize in real tests
    frameworks = [pd.DataFrame]
    if PYARROW_AVAILABLE:
        frameworks.append(pa.Table)

    test_data = DataGenerator.generate_data(n_rows=15, numeric_cols=["col1", "col2"], seed=42)

    converter = DataConverter()

    for framework_type in frameworks:
        # Convert to framework
        data = converter.to_framework(test_data, framework_type)

        # Validate (validators work with any framework)
        validate_row_count(data, 15)
        validate_columns_exist(data, ["col1", "col2"])

        print(f"✓ Test passed for {framework_type.__name__}")


def example_pytest_parametrize_pattern() -> None:
    """Example: Show the pytest parametrize pattern for multi-framework tests."""
    print("\n=== Pytest Parametrize Pattern ===")

    example_code = """
import pytest
import pandas as pd
import pyarrow as pa

@pytest.mark.parametrize("framework_type", [
    pd.DataFrame,
    pa.Table,
])
def test_my_feature_group_multi_framework(framework_type):
    # Generate test data
    test_data = DataGenerator.generate_data(
        n_rows=100,
        numeric_cols=["feature1", "feature2"],
        categorical_cols=["category"]
    )

    # Convert to target framework
    converter = DataConverter()
    input_data = converter.to_framework(test_data, framework_type)

    # Run your feature group
    result = my_feature_group.transform(input_data)

    # Validate structure (works with any framework!)
    validate_row_count(result, 100)
    validate_columns_exist(result, ["output_col"])
"""

    print(example_code)


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Framework Testing Examples")
    print("=" * 60)

    example_convert_to_frameworks()
    example_convert_between_frameworks()
    example_validate_across_frameworks()
    example_framework_agnostic_test()
    example_pytest_parametrize_pattern()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
