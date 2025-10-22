"""
Examples of data generation patterns using DataGenerator and EdgeCaseGenerator.

This module demonstrates how to generate test data for feature group testing.
"""

from tests.test_plugins.feature_group.test_tooling.data_generation.generators import (
    DataGenerator,
    EdgeCaseGenerator,
)


def example_basic_data_generation() -> None:
    """Example: Generate basic test data with different column types."""
    print("\n=== Basic Data Generation ===")

    # Generate numeric columns
    numeric_data = DataGenerator.generate_numeric_columns(
        n_rows=10, column_names=["value1", "value2", "value3"], value_range=(0, 100), seed=42
    )
    print(f"Generated {len(numeric_data)} numeric columns with {len(numeric_data['value1'])} rows")

    # Generate categorical columns
    categorical_data = DataGenerator.generate_categorical_columns(
        n_rows=10, column_names=["category_a", "category_b"], n_categories=3, seed=42
    )
    print(f"Generated categorical data: {set(categorical_data['category_a'])}")

    # Generate temporal column
    temporal_data = DataGenerator.generate_temporal_column(n_rows=10, column_name="timestamp", start="2023-01-01")
    print(f"Generated temporal data from {temporal_data['timestamp'][0]} to {temporal_data['timestamp'][-1]}")


def example_complete_dataset() -> None:
    """Example: Generate a complete dataset with all column types."""
    print("\n=== Complete Dataset Generation ===")

    data = DataGenerator.generate_data(
        n_rows=100, numeric_cols=["feature1", "feature2", "feature3"], categorical_cols=["group"], temporal_col="date"
    )

    print(f"Generated dataset with {len(data)} columns:")
    for col, values in data.items():
        print(f"  - {col}: {len(values)} rows")


def example_edge_cases() -> None:
    """Example: Generate edge case test data."""
    print("\n=== Edge Case Data Generation ===")

    # Empty data
    empty = EdgeCaseGenerator.empty_data(["col1", "col2", "col3"])
    print(f"Empty data: {len(empty['col1'])} rows")

    # Single row
    single = EdgeCaseGenerator.single_row(["col1", "col2"], value=42)
    print(f"Single row data: {single}")

    # Data with nulls
    base_data = DataGenerator.generate_numeric_columns(n_rows=10, column_names=["col1", "col2"])
    with_nulls = EdgeCaseGenerator.with_nulls(base_data, columns=["col1"], null_percentage=0.3, seed=42)
    null_count = sum(1 for v in with_nulls["col1"] if v is None)
    print(f"Data with nulls: {null_count}/{len(with_nulls['col1'])} values are null")

    # All nulls
    all_nulls = EdgeCaseGenerator.all_nulls(["col1", "col2"], n_rows=5)
    print(f"All nulls data: {all_nulls['col1']}")

    # Duplicated rows
    base_data = {"col1": [1, 2], "col2": [3, 4]}
    duplicated = EdgeCaseGenerator.duplicate_rows(base_data, n_duplicates=3)
    print(f"Duplicated data: {len(duplicated['col1'])} rows (originally {len(base_data['col1'])})")


def example_reproducible_data() -> None:
    """Example: Generate reproducible test data using seeds."""
    print("\n=== Reproducible Data Generation ===")

    # Same seed produces same data
    data1 = DataGenerator.generate_numeric_columns(n_rows=5, column_names=["col1"], seed=42)
    data2 = DataGenerator.generate_numeric_columns(n_rows=5, column_names=["col1"], seed=42)

    print(f"Data 1: {data1['col1'][:3]}")
    print(f"Data 2: {data2['col1'][:3]}")
    print(f"Are they equal? {data1['col1'] == data2['col1']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Data Generation Examples")
    print("=" * 60)

    example_basic_data_generation()
    example_complete_dataset()
    example_edge_cases()
    example_reproducible_data()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
