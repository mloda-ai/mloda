"""
Business-agnostic data generation utilities.

These utilities generate data of various shapes, types, and edge cases
without any business meaning.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class DataGenerator:
    """
    Generate test data with specified structural properties.

    Focus: data shape, size, types - NOT business logic.
    """

    @staticmethod
    def generate_numeric_columns(
        n_rows: int,
        column_names: List[str],
        value_range: tuple[float, float] = (0, 100),
        dtype: str = "float64",
        seed: Optional[int] = 42,
    ) -> Dict[str, List[Any]]:
        """
        Generate numeric columns with random values.

        Args:
            n_rows: Number of rows
            column_names: Names for columns
            value_range: (min, max) for values
            dtype: Numpy dtype
            seed: Random seed for reproducibility

        Returns:
            Dict mapping column names to value lists

        Example:
            >>> data = DataGenerator.generate_numeric_columns(
            ...     n_rows=100,
            ...     column_names=["col1", "col2", "col3"]
            ... )
            >>> len(data["col1"])
            100
            >>> len(data)
            3
        """
        np.random.seed(seed)
        return {
            col: np.random.uniform(value_range[0], value_range[1], n_rows).astype(dtype).tolist()
            for col in column_names
        }

    @staticmethod
    def generate_categorical_columns(
        n_rows: int, column_names: List[str], n_categories: int = 5, seed: Optional[int] = 42
    ) -> Dict[str, List[Any]]:
        """
        Generate categorical columns.

        Args:
            n_rows: Number of rows
            column_names: Names for columns
            n_categories: Number of unique categories (labeled A, B, C, ...)
            seed: Random seed

        Returns:
            Dict mapping column names to category lists

        Example:
            >>> data = DataGenerator.generate_categorical_columns(
            ...     n_rows=100,
            ...     column_names=["cat1", "cat2"],
            ...     n_categories=3
            ... )
            >>> len(data["cat1"])
            100
            >>> set(data["cat1"]) <= {"A", "B", "C"}
            True
        """
        np.random.seed(seed)
        categories = [chr(65 + i) for i in range(n_categories)]  # A, B, C, ...
        return {col: np.random.choice(categories, n_rows).tolist() for col in column_names}

    @staticmethod
    def generate_temporal_column(
        n_rows: int, column_name: str = "timestamp", start: str = "2023-01-01", freq: str = "D"
    ) -> Dict[str, pd.DatetimeIndex]:
        """
        Generate temporal column.

        Args:
            n_rows: Number of rows
            column_name: Name for timestamp column
            start: Start date
            freq: Frequency (D=daily, H=hourly, etc.)

        Returns:
            Dict with timestamp column

        Example:
            >>> data = DataGenerator.generate_temporal_column(
            ...     n_rows=10,
            ...     column_name="timestamp"
            ... )
            >>> len(data["timestamp"])
            10
        """
        return {column_name: pd.date_range(start=start, periods=n_rows, freq=freq)}

    @staticmethod
    def generate_data(
        n_rows: int,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        temporal_col: Optional[str] = None,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Generate complete test dataset with mixed column types.

        Args:
            n_rows: Number of rows
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
            temporal_col: Name for temporal column (optional)
            seed: Random seed

        Returns:
            Dict with all columns

        Example:
            >>> data = DataGenerator.generate_data(
            ...     n_rows=100,
            ...     numeric_cols=["val1", "val2", "val3"],
            ...     categorical_cols=["cat1"],
            ...     temporal_col="timestamp"
            ... )
            >>> len(data["val1"])
            100
            >>> "cat1" in data
            True
            >>> "timestamp" in data
            True
        """
        result = {}

        if numeric_cols:
            result.update(DataGenerator.generate_numeric_columns(n_rows, numeric_cols, seed=seed))

        if categorical_cols:
            result.update(DataGenerator.generate_categorical_columns(n_rows, categorical_cols, seed=seed))

        if temporal_col:
            result.update(DataGenerator.generate_temporal_column(n_rows, temporal_col))

        return result


class EdgeCaseGenerator:
    """
    Generate edge case test data.

    Focus: structural edge cases (nulls, empty, single row, etc.)
    """

    @staticmethod
    def with_nulls(
        data: Dict[str, List[Any]], columns: List[str], null_percentage: float = 0.2, seed: Optional[int] = 42
    ) -> Dict[str, List[Any]]:
        """
        Add null values to specified columns.

        Args:
            data: Base data dict
            columns: Columns to add nulls to
            null_percentage: Fraction of values to make null (0.0-1.0)
            seed: Random seed

        Returns:
            Data dict with nulls added

        Example:
            >>> base_data = {"col1": [1, 2, 3, 4, 5]}
            >>> data_with_nulls = EdgeCaseGenerator.with_nulls(
            ...     base_data,
            ...     columns=["col1"],
            ...     null_percentage=0.4
            ... )
            >>> None in data_with_nulls["col1"]
            True
        """
        np.random.seed(seed)
        result = data.copy()

        for col in columns:
            if col not in result:
                continue

            values = list(result[col])
            n_nulls = int(len(values) * null_percentage)
            null_indices = np.random.choice(len(values), n_nulls, replace=False)

            for idx in null_indices:
                values[idx] = None

            result[col] = values

        return result

    @staticmethod
    def empty_data(columns: List[str]) -> Dict[str, List]:
        """
        Generate empty dataset with specified columns.

        Args:
            columns: Column names

        Returns:
            Dict with empty lists

        Example:
            >>> data = EdgeCaseGenerator.empty_data(["col1", "col2", "col3"])
            >>> len(data["col1"])
            0
            >>> len(data)
            3
        """
        return {col: [] for col in columns}

    @staticmethod
    def single_row(columns: List[str], value: Any = 1) -> Dict[str, List]:
        """
        Generate single-row dataset.

        Args:
            columns: Column names
            value: Value for all columns

        Returns:
            Dict with single-element lists

        Example:
            >>> data = EdgeCaseGenerator.single_row(["col1", "col2"], value=42)
            >>> data["col1"]
            [42]
            >>> len(data["col1"])
            1
        """
        return {col: [value] for col in columns}

    @staticmethod
    def all_nulls(columns: List[str], n_rows: int) -> Dict[str, List]:
        """
        Generate dataset where all values are null.

        Args:
            columns: Column names
            n_rows: Number of rows

        Returns:
            Dict with all None values

        Example:
            >>> data = EdgeCaseGenerator.all_nulls(["col1", "col2"], n_rows=10)
            >>> data["col1"][0] is None
            True
            >>> len(data["col1"])
            10
        """
        return {col: [None] * n_rows for col in columns}

    @staticmethod
    def duplicate_rows(data: Dict[str, List[Any]], n_duplicates: int = 2) -> Dict[str, List[Any]]:
        """
        Duplicate all rows in dataset.

        Args:
            data: Base data
            n_duplicates: How many times to duplicate each row

        Returns:
            Data with duplicated rows

        Example:
            >>> data = {"col1": [1, 2], "col2": [3, 4]}
            >>> dup_data = EdgeCaseGenerator.duplicate_rows(data, n_duplicates=2)
            >>> len(dup_data["col1"])
            4
        """
        result = {}
        for col, values in data.items():
            result[col] = values * n_duplicates
        return result
