# Feature Group Test Tooling - Design Document

## Document Status
- **Status**: Draft v2.0
- **Author**: Claude Code
- **Date**: 2025-10-22
- **Version**: 2.0 (Revised - Business-Agnostic)

## Executive Summary

### Goals
This design proposes a **business-agnostic** test tooling framework for feature groups, providing structural utilities for testing data shapes, types, framework compatibility, and edge cases - without any business logic.

### Key Objectives
1. **Reduce Boilerplate**: Minimize repetitive test setup code
2. **Framework Testing**: Ensure feature groups work correctly across Pandas, PyArrow, and future frameworks
3. **Structural Testing**: Test data shapes, types, nulls, empty data, single rows - purely technical concerns
4. **No Business Logic**: Zero business-specific scenarios or data
5. **Easy Adoption**: Simple utilities that don't constrain test authors

### What This Tooling IS:
- ✅ Base test classes to reduce boilerplate
- ✅ Data generation utilities (N rows, M columns, various types)
- ✅ Framework conversion utilities (dict ↔ pandas ↔ pyarrow)
- ✅ Generic validators (row count, column existence, null checks, type checks)
- ✅ Integration test helpers (mlodaAPI setup, result finding)

### What This Tooling IS NOT:
- ❌ Business-specific test scenarios
- ❌ Domain knowledge about features
- ❌ Prescriptive about what to test
- ❌ A replacement for feature-specific test logic

---

## Current State Analysis

### Compute Framework Testing Pattern

**Location**: `/workspace/tests/test_plugins/compute_framework/test_tooling/`

**Key Insight**: The compute framework test tooling is purely structural:
- Tests merge operations (INNER, LEFT, OUTER, etc.)
- Uses generic data (id, category, value columns)
- No business logic - just data structure operations

**Example**:
```python
# From test_scenarios.py - purely structural
SCENARIOS = {
    "inner_basic": {
        "left": [
            {"id": 1, "category": "A", "left_value": 10},
            {"id": 2, "category": "B", "left_value": 20},
        ],
        "right": [
            {"id": 1, "category": "A", "right_value": 100},
            {"id": 2, "category": "B", "right_value": 200},
        ],
        "index": ("id", "category"),
        "expected_rows": 2,
    }
}
```

**What makes it business-agnostic**:
- Generic column names (id, category, value)
- Focus on structure (row counts, column existence, nulls)
- Tests framework behavior, not business logic

### Current Feature Group Testing Pattern

**Problems**:
1. Each feature group creates its own test data (duplicated effort)
2. No utilities for framework conversion
3. Integration test patterns duplicated across feature groups
4. Validators scattered and duplicated

**What's needed**:
- Utilities to generate test data of various shapes
- Framework conversion helpers
- Integration test helpers
- Generic validators

---

## Proposed Architecture

### Design Philosophy

**Core Principle**: Provide **utilities, not scenarios**

The tooling should be like a toolkit:
- Tools for generating data
- Tools for converting between frameworks
- Tools for validation
- Tools for integration testing
- **Users bring their own test logic and data**

### Architecture Overview

```
tests/test_plugins/feature_group/test_tooling/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── feature_group_test_base.py      # Optional base class
│   ├── integration_test_mixin.py       # Integration test utilities
│   └── multi_framework_test_mixin.py   # Multi-framework utilities
├── data_generation/
│   ├── __init__.py
│   ├── generators.py                   # Data shape generators
│   ├── edge_cases.py                   # Edge case data (nulls, empty, etc.)
│   └── framework_data.py               # Framework-specific data helpers
├── converters/
│   ├── __init__.py
│   └── framework_converter.py          # Dict ↔ Framework conversion
├── validators/
│   ├── __init__.py
│   ├── structural_validators.py        # Row count, columns, types
│   └── result_validators.py            # Result validation helpers
└── integration/
    ├── __init__.py
    └── mloda_helpers.py                # mlodaAPI test helpers
```

---

## Core Components

### 1. Data Generation Utilities

**Purpose**: Generate test data of various shapes and types WITHOUT business logic.

**File**: `tests/test_plugins/feature_group/test_tooling/data_generation/generators.py`

```python
"""
Business-agnostic data generation utilities.

These utilities generate data of various shapes, types, and edge cases
without any business meaning.
"""

from typing import Any, Dict, List, Optional, Union
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
        seed: Optional[int] = 42
    ) -> Dict[str, List]:
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
            >>> # Returns: {"col1": [...], "col2": [...], "col3": [...]}
        """
        np.random.seed(seed)
        return {
            col: np.random.uniform(value_range[0], value_range[1], n_rows).astype(dtype).tolist()
            for col in column_names
        }

    @staticmethod
    def generate_categorical_columns(
        n_rows: int,
        column_names: List[str],
        n_categories: int = 5,
        seed: Optional[int] = 42
    ) -> Dict[str, List]:
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
            >>> # Returns: {"cat1": ["A", "B", "C", ...], "cat2": ["A", "C", ...]}
        """
        np.random.seed(seed)
        categories = [chr(65 + i) for i in range(n_categories)]  # A, B, C, ...
        return {
            col: np.random.choice(categories, n_rows).tolist()
            for col in column_names
        }

    @staticmethod
    def generate_temporal_column(
        n_rows: int,
        column_name: str = "timestamp",
        start: str = "2023-01-01",
        freq: str = "D"
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
            ...     n_rows=100,
            ...     column_name="timestamp"
            ... )
            >>> # Returns: {"timestamp": DatetimeIndex([...])}
        """
        return {
            column_name: pd.date_range(start=start, periods=n_rows, freq=freq)
        }

    @staticmethod
    def generate_data(
        n_rows: int,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        temporal_col: Optional[str] = None,
        seed: Optional[int] = 42
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
        """
        result = {}

        if numeric_cols:
            result.update(DataGenerator.generate_numeric_columns(
                n_rows, numeric_cols, seed=seed
            ))

        if categorical_cols:
            result.update(DataGenerator.generate_categorical_columns(
                n_rows, categorical_cols, seed=seed
            ))

        if temporal_col:
            result.update(DataGenerator.generate_temporal_column(
                n_rows, temporal_col
            ))

        return result


class EdgeCaseGenerator:
    """
    Generate edge case test data.

    Focus: structural edge cases (nulls, empty, single row, etc.)
    """

    @staticmethod
    def with_nulls(
        data: Dict[str, List],
        columns: List[str],
        null_percentage: float = 0.2,
        seed: Optional[int] = 42
    ) -> Dict[str, List]:
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
            >>> # Some values in col1 are now None
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
            >>> # Returns: {"col1": [], "col2": [], "col3": []}
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
            >>> # Returns: {"col1": [42], "col2": [42]}
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
            >>> # Returns: {"col1": [None, ...], "col2": [None, ...]}
        """
        return {col: [None] * n_rows for col in columns}

    @staticmethod
    def duplicate_rows(data: Dict[str, List], n_duplicates: int = 2) -> Dict[str, List]:
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
            >>> # Returns: {"col1": [1, 2, 1, 2], "col2": [3, 4, 3, 4]}
        """
        result = {}
        for col, values in data.items():
            result[col] = values * n_duplicates
        return result
```

---

### 2. Framework Conversion Utilities

**Purpose**: Convert data between dict and any compute framework.

**Approach**: **Reuse existing `ComputeFrameworkTransformer` infrastructure** (same as compute framework test tooling).

**File**: `tests/test_plugins/feature_group/test_tooling/converters/data_converter.py`

**Why this approach?**
- ✅ Reuses production transformation infrastructure
- ✅ Automatically supports ALL frameworks with PyArrow transformers
- ✅ Consistent with compute framework test tooling pattern
- ✅ No duplicate conversion logic to maintain
- ✅ New frameworks automatically supported when they add PyArrow transformers

```python
"""
Test data converter using the existing ComputeFrameworkTransformer system.

This module provides utilities to convert test data (List[Dict[str, Any]]) to and from
any compute framework format by leveraging the existing transformer infrastructure.
All conversions go through PyArrow as an intermediate format.

Based on: tests/test_plugins/compute_framework/test_tooling/multi_index/test_data_converter.py
"""

from typing import Any, Dict, List, Optional, Type

from mloda_core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

try:
    import pyarrow as pa
except ImportError:
    pa = None


# Load plugins once at module import time
PluginLoader().load_group("compute_framework")


class DataConverter:
    """
    Converts test data between List[Dict] format and any framework format.

    This class leverages the existing ComputeFrameworkTransformer system to automatically
    support all frameworks that have PyArrow transformers. New frameworks are automatically
    supported when they implement a PyArrow transformer.

    All conversions follow the path:
    - To framework: List[Dict] → PyArrow → Target Framework
    - From framework: Source Framework → PyArrow → List[Dict]

    Supported frameworks (automatically):
    - Pandas (pd.DataFrame)
    - PyArrow (pa.Table)
    - Polars (pl.DataFrame)
    - DuckDB
    - PythonDict (list)
    - Any future framework with PyArrow transformer
    """

    def __init__(self) -> None:
        """Initialize the converter with the transformer registry."""
        # Initialize transformer registry (will auto-discover all loaded transformer subclasses)
        self.transformer = ComputeFrameworkTransformer()

    def to_framework(
        self,
        data: List[Dict[str, Any]],
        target_framework_type: Type[Any],
        connection: Optional[Any] = None,
    ) -> Any:
        """
        Convert test data (List[Dict]) to target framework format.

        Args:
            data: Test data in List[Dict[str, Any]] format
            target_framework_type: The target framework type (e.g., pd.DataFrame, pa.Table)
            connection: Optional framework connection object (e.g., for DuckDB)

        Returns:
            Data in the target framework's native format

        Raises:
            KeyError: If no transformer exists for the target framework
            ImportError: If required framework is not installed

        Example:
            >>> converter = DataConverter()
            >>> data = [{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]
            >>> df = converter.to_framework(data, pd.DataFrame)
            >>> # Returns pandas DataFrame
            >>>
            >>> # Works with any framework!
            >>> table = converter.to_framework(data, pa.Table)
            >>> # Returns PyArrow Table
        """
        if pa is None:
            raise ImportError("PyArrow is required for test data conversion")

        # Special case: if target is already list, no conversion needed
        if target_framework_type == list:
            return data

        # Step 1: List[Dict] → PyArrow
        try:
            transformer_list_to_arrow = self.transformer.transformer_map[(list, pa.Table)]
            arrow_data = transformer_list_to_arrow.transform(list, pa.Table, data, None)
        except KeyError:
            raise KeyError(
                f"No transformer found for list → PyArrow. Ensure PythonDictPyarrowTransformer is available."
            )

        # Special case: if target is PyArrow, we're done
        if target_framework_type == pa.Table:
            return arrow_data

        # Step 2: PyArrow → Target Framework
        try:
            transformer_arrow_to_target = self.transformer.transformer_map[(pa.Table, target_framework_type)]
            return transformer_arrow_to_target.transform(pa.Table, target_framework_type, arrow_data, connection)
        except KeyError:
            raise KeyError(
                f"No transformer found for PyArrow → {target_framework_type}. "
                f"Ensure the framework has a PyArrow transformer."
            )

    def from_framework(self, data: Any, source_framework_type: Type[Any]) -> List[Dict[str, Any]]:
        """
        Convert framework data back to test data (List[Dict]) format.

        Args:
            data: Data in the source framework's native format
            source_framework_type: The source framework type (e.g., pd.DataFrame)

        Returns:
            List[Dict[str, Any]]: Data in test format

        Raises:
            KeyError: If no transformer exists for the source framework
            ImportError: If required framework is not installed

        Example:
            >>> converter = DataConverter()
            >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            >>> data = converter.from_framework(df, pd.DataFrame)
            >>> # Returns: [{"col1": 1, "col2": 3}, {"col1": 2, "col2": 4}]
        """
        if pa is None:
            raise ImportError("PyArrow is required for test data conversion")

        # Special case: if source is already list, no conversion needed
        if source_framework_type == list:
            assert isinstance(data, list), "Expected list data"
            return data

        # Special case: if source is PyArrow, skip the first step
        if source_framework_type == pa.Table:
            arrow_data = data
        else:
            # Step 1: Source Framework → PyArrow
            try:
                transformer_source_to_arrow = self.transformer.transformer_map[(source_framework_type, pa.Table)]
                arrow_data = transformer_source_to_arrow.transform(source_framework_type, pa.Table, data, None)
            except KeyError:
                raise KeyError(
                    f"No transformer found for {source_framework_type} → PyArrow. "
                    f"Ensure the framework has a PyArrow transformer."
                )

        # Step 2: PyArrow → List[Dict]
        try:
            transformer_arrow_to_list = self.transformer.transformer_map[(pa.Table, list)]
            result: List[Dict[str, Any]] = transformer_arrow_to_list.transform(pa.Table, list, arrow_data, None)
            return result
        except KeyError:
            raise KeyError(
                f"No transformer found for PyArrow → list. Ensure PythonDictPyarrowTransformer is available."
            )

    def to_pandas(self, data: Any) -> Any:
        """
        Convenience method: Convert any framework data to pandas.

        Args:
            data: Data in any framework format

        Returns:
            Pandas DataFrame

        Example:
            >>> converter = DataConverter()
            >>> table = pa.Table.from_pydict({"col1": [1, 2]})
            >>> df = converter.to_pandas(table)
            >>> # Returns pandas DataFrame
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, list):
            return pd.DataFrame(data)
        elif hasattr(data, 'to_pandas'):
            return data.to_pandas()
        else:
            # Use the transformer system
            dict_data = self.from_framework(data, type(data))
            return pd.DataFrame(dict_data)
```

**Key Differences from Custom Implementation:**
1. Uses `ComputeFrameworkTransformer` instead of hardcoded conversions
2. All conversions go through PyArrow as intermediate format
3. Automatically supports all frameworks (Pandas, PyArrow, Polars, DuckDB, Spark, Iceberg, etc.)
4. Consistent with compute framework test tooling pattern
5. No framework-specific code - works via transformer registry

---

### 3. Structural Validators

**Purpose**: Generic validation functions for structural properties (no business logic).

**File**: `tests/test_plugins/feature_group/test_tooling/validators/structural_validators.py`

```python
"""
Business-agnostic structural validators.

Validate data structure properties: shape, columns, types, nulls.
NO business logic.
"""

from typing import Any, List, Optional, Dict, Type
import pandas as pd


def validate_row_count(
    result: Any,
    expected_count: int,
    message: Optional[str] = None
) -> None:
    """
    Validate number of rows.

    Args:
        result: Result data (DataFrame, Table, etc.)
        expected_count: Expected row count
        message: Optional custom error message
    """
    if hasattr(result, '__len__'):
        actual = len(result)
    elif hasattr(result, 'num_rows'):
        actual = result.num_rows
    else:
        raise ValueError(f"Cannot get row count from type: {type(result)}")

    if actual != expected_count:
        error_msg = message or f"Expected {expected_count} rows, got {actual}"
        raise AssertionError(error_msg)


def validate_columns_exist(
    result: Any,
    expected_columns: List[str],
    message: Optional[str] = None
) -> None:
    """
    Validate that columns exist.

    Args:
        result: Result data
        expected_columns: Expected column names
        message: Optional custom error message
    """
    if hasattr(result, 'columns'):
        actual_columns = set(result.columns)
    elif hasattr(result, 'schema'):
        actual_columns = set(result.schema.names)
    else:
        raise ValueError(f"Cannot get columns from type: {type(result)}")

    missing = set(expected_columns) - actual_columns
    if missing:
        error_msg = message or f"Missing columns: {missing}. Available: {actual_columns}"
        raise AssertionError(error_msg)


def validate_column_count(
    result: Any,
    expected_count: int,
    message: Optional[str] = None
) -> None:
    """
    Validate number of columns.

    Args:
        result: Result data
        expected_count: Expected column count
        message: Optional custom error message
    """
    if hasattr(result, 'columns'):
        actual = len(result.columns)
    elif hasattr(result, 'schema'):
        actual = len(result.schema.names)
    else:
        raise ValueError(f"Cannot get column count from type: {type(result)}")

    if actual != expected_count:
        error_msg = message or f"Expected {expected_count} columns, got {actual}"
        raise AssertionError(error_msg)


def validate_no_nulls(
    result: Any,
    columns: List[str],
    message: Optional[str] = None
) -> None:
    """
    Validate no null values in specified columns.

    Args:
        result: Result data
        columns: Columns to check
        message: Optional custom error message
    """
    df = result.to_pandas() if hasattr(result, 'to_pandas') else result

    for col in columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            error_msg = message or f"Column '{col}' has {null_count} null values"
            raise AssertionError(error_msg)


def validate_has_nulls(
    result: Any,
    columns: List[str],
    message: Optional[str] = None
) -> None:
    """
    Validate that specified columns have some null values.

    Args:
        result: Result data
        columns: Columns that should have nulls
        message: Optional custom error message
    """
    df = result.to_pandas() if hasattr(result, 'to_pandas') else result

    for col in columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            error_msg = message or f"Column '{col}' has no null values (expected some)"
            raise AssertionError(error_msg)


def validate_column_types(
    result: Any,
    expected_types: Dict[str, Type],
    message: Optional[str] = None
) -> None:
    """
    Validate column data types.

    Args:
        result: Result data
        expected_types: Dict mapping column names to expected types
        message: Optional custom error message

    Example:
        >>> validate_column_types(df, {"col1": np.float64, "col2": object})
    """
    df = result.to_pandas() if hasattr(result, 'to_pandas') else result

    for col, expected_type in expected_types.items():
        if col not in df.columns:
            raise AssertionError(f"Column '{col}' not found")

        actual_type = df[col].dtype
        if not pd.api.types.is_dtype_equal(actual_type, expected_type):
            error_msg = message or \
                f"Column '{col}' has type {actual_type}, expected {expected_type}"
            raise AssertionError(error_msg)


def validate_shape(
    result: Any,
    expected_shape: tuple[int, int],
    message: Optional[str] = None
) -> None:
    """
    Validate data shape (rows, columns).

    Args:
        result: Result data
        expected_shape: (n_rows, n_cols)
        message: Optional custom error message
    """
    if hasattr(result, 'shape'):
        actual_shape = result.shape
    elif hasattr(result, 'num_rows'):
        actual_shape = (result.num_rows, len(result.schema.names))
    else:
        raise ValueError(f"Cannot get shape from type: {type(result)}")

    if actual_shape != expected_shape:
        error_msg = message or f"Expected shape {expected_shape}, got {actual_shape}"
        raise AssertionError(error_msg)


def validate_not_empty(
    result: Any,
    message: Optional[str] = None
) -> None:
    """
    Validate data is not empty.

    Args:
        result: Result data
        message: Optional custom error message
    """
    if hasattr(result, '__len__'):
        is_empty = len(result) == 0
    elif hasattr(result, 'num_rows'):
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
    message: Optional[str] = None
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
    df = result.to_pandas() if hasattr(result, 'to_pandas') else result

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
```

---

### 4. Integration Test Helpers

**Purpose**: Simplify mlodaAPI testing without business logic.

**File**: `tests/test_plugins/feature_group/test_tooling/integration/mloda_helpers.py`

```python
"""
Integration test helpers for mlodaAPI testing.

Utilities for setting up and validating integration tests.
NO business logic.
"""

from typing import Any, List, Optional, Set, Type

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI


class MlodaTestHelper:
    """
    Helper for mlodaAPI integration testing.

    Simplifies common integration test patterns.
    """

    @staticmethod
    def create_plugin_collector(
        feature_groups: Set[Type[AbstractFeatureGroup]],
        additional_plugins: Optional[Set] = None
    ) -> PlugInCollector:
        """
        Create plugin collector for testing.

        Args:
            feature_groups: Feature group classes to enable
            additional_plugins: Additional plugins (e.g., data creators)

        Returns:
            Configured PlugInCollector
        """
        plugins = feature_groups.copy()
        if additional_plugins:
            plugins.update(additional_plugins)
        return PlugInCollector.enabled_feature_groups(plugins)

    @staticmethod
    def run_integration_test(
        features: List[str],
        feature_groups: Set[Type[AbstractFeatureGroup]],
        compute_frameworks: Set[Type[ComputeFrameWork]],
        additional_plugins: Optional[Set] = None
    ) -> List[Any]:
        """
        Run mlodaAPI.run_all() with specified configuration.

        Args:
            features: Feature names to compute
            feature_groups: Feature groups to enable
            compute_frameworks: Compute frameworks to use
            additional_plugins: Additional plugins

        Returns:
            Results from mlodaAPI.run_all()

        Example:
            >>> results = MlodaTestHelper.run_integration_test(
            ...     features=["col1", "computed_col2"],
            ...     feature_groups={MyFeatureGroup},
            ...     compute_frameworks={PandasDataframe},
            ...     additional_plugins={MyTestDataCreator}
            ... )
        """
        plugin_collector = MlodaTestHelper.create_plugin_collector(
            feature_groups,
            additional_plugins
        )

        return mlodaAPI.run_all(
            features,
            compute_frameworks=compute_frameworks,
            plugin_collector=plugin_collector
        )

    @staticmethod
    def find_result_with_column(
        results: List[Any],
        column_name: str
    ) -> Optional[Any]:
        """
        Find result containing specified column.

        Args:
            results: List of results from mlodaAPI.run_all()
            column_name: Column to search for

        Returns:
            Result containing the column, or None
        """
        for result in results:
            if hasattr(result, 'columns'):  # Pandas
                if column_name in result.columns:
                    return result
            elif hasattr(result, 'schema'):  # PyArrow
                if column_name in result.schema.names:
                    return result
        return None

    @staticmethod
    def assert_result_found(
        results: List[Any],
        column_name: str
    ) -> Any:
        """
        Find result with column or raise assertion error.

        Args:
            results: Results from mlodaAPI.run_all()
            column_name: Column that should exist

        Returns:
            Result containing the column

        Raises:
            AssertionError: If no result contains the column
        """
        result = MlodaTestHelper.find_result_with_column(results, column_name)
        if result is None:
            raise AssertionError(
                f"No result found containing column '{column_name}'. "
                f"Got {len(results)} results."
            )
        return result

    @staticmethod
    def count_results(results: List[Any]) -> int:
        """Get number of results."""
        return len(results)

    @staticmethod
    def assert_result_count(
        results: List[Any],
        expected_count: int
    ) -> None:
        """
        Assert number of results.

        Args:
            results: Results from mlodaAPI.run_all()
            expected_count: Expected number of results
        """
        actual = len(results)
        if actual != expected_count:
            raise AssertionError(
                f"Expected {expected_count} results, got {actual}"
            )
```

---

### 5. Optional Base Classes

**Purpose**: Optional base classes to reduce boilerplate (NOT mandatory).

**File**: `tests/test_plugins/feature_group/test_tooling/base/feature_group_test_base.py`

```python
"""
Optional base test class for feature groups.

This is OPTIONAL - use only if it reduces boilerplate for your tests.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from ..converters.framework_converter import FrameworkConverter
from ..validators.structural_validators import (
    validate_columns_exist,
    validate_row_count,
    validate_no_nulls,
    validate_has_nulls,
)


class FeatureGroupTestBase(ABC):
    """
    Optional base class for feature group testing.

    Provides common utilities. Use if helpful, ignore if not.
    """

    # ==================== ABSTRACT METHODS ====================

    @classmethod
    @abstractmethod
    def feature_group_class(cls) -> Type[AbstractFeatureGroup]:
        """Return the feature group class to test."""
        pass

    # ==================== OPTIONAL METHODS ====================

    @classmethod
    def get_test_frameworks(cls) -> Set[Type[ComputeFrameWork]]:
        """
        Return frameworks to test.
        Override to customize.
        """
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
        return {PandasDataframe, PyarrowTable}

    # ==================== UTILITY METHODS ====================

    def to_pandas(self, data: Any) -> Any:
        """Convert data to pandas."""
        return FrameworkConverter.to_pandas(data)

    def to_framework(self, data: Dict[str, Any], framework: Type[ComputeFrameWork]) -> Any:
        """Convert dict to framework format."""
        return FrameworkConverter.to_framework(data, framework)

    def assert_columns_exist(self, result: Any, columns: List[str]) -> None:
        """Assert columns exist."""
        validate_columns_exist(result, columns)

    def assert_row_count(self, result: Any, count: int) -> None:
        """Assert row count."""
        validate_row_count(result, count)

    def assert_no_nulls(self, result: Any, columns: List[str]) -> None:
        """Assert no nulls."""
        validate_no_nulls(result, columns)

    def assert_has_nulls(self, result: Any, columns: List[str]) -> None:
        """Assert has nulls."""
        validate_has_nulls(result, columns)
```

---

## Usage Examples

### Example 1: Testing with Data Generation

```python
"""
Test time window feature group using test tooling.

NO business logic - just structural testing.
"""

from tests.test_plugins.feature_group.test_tooling.data_generation import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters import DataConverter
from tests.test_plugins.feature_group.test_tooling.validators import validate_columns_exist, validate_row_count

from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet


def test_time_window_basic():
    """Test time window with generated data - no business logic."""

    # Generate test data (structural only)
    data = DataGenerator.generate_data(
        n_rows=100,
        numeric_cols=["val1", "val2", "val3"],
        temporal_col="timestamp"
    )

    # Convert to pandas
    converter = DataConverter()
    df = converter.to_framework(data, pd.DataFrame)

    # Calculate feature (business logic is in the feature group, not the test)
    fg = PandasTimeWindowFeatureGroup()
    features = FeatureSet(["avg_3_day_window__val1"])
    result = fg.calculate_feature(df, features)

    # Validate structure (no business logic)
    validate_columns_exist(result, ["avg_3_day_window__val1"])
    validate_row_count(result, 100)


def test_time_window_with_nulls():
    """Test time window handles nulls correctly."""
    from tests.test_plugins.feature_group.test_tooling.data_generation import EdgeCaseGenerator

    # Generate data with nulls
    base_data = DataGenerator.generate_data(
        n_rows=50,
        numeric_cols=["val1"],
        temporal_col="timestamp"
    )

    data_with_nulls = EdgeCaseGenerator.with_nulls(
        base_data,
        columns=["val1"],
        null_percentage=0.3
    )

    converter = DataConverter()
    df = converter.to_framework(data_with_nulls, pd.DataFrame)

    # Test
    fg = PandasTimeWindowFeatureGroup()
    features = FeatureSet(["avg_3_day_window__val1"])
    result = fg.calculate_feature(df, features)

    # Validate
    validate_columns_exist(result, ["avg_3_day_window__val1"])
    validate_row_count(result, 50)


def test_time_window_empty_data():
    """Test time window with empty data."""
    from tests.test_plugins.feature_group.test_tooling.data_generation import EdgeCaseGenerator

    # Edge case: empty data
    empty_data = EdgeCaseGenerator.empty_data(["val1", "timestamp"])
    converter = DataConverter()
    df = converter.to_framework(empty_data, pd.DataFrame)

    fg = PandasTimeWindowFeatureGroup()
    features = FeatureSet(["avg_3_day_window__val1"])
    result = fg.calculate_feature(df, features)

    # Should handle gracefully
    validate_row_count(result, 0)
```

### Example 2: Multi-Framework Testing

```python
"""
Test feature group works across multiple frameworks.
"""

import pytest
from tests.test_plugins.feature_group.test_tooling.data_generation import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters import DataConverter
from tests.test_plugins.feature_group.test_tooling.validators import validate_columns_exist

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable


@pytest.mark.parametrize("framework", [PandasDataframe, PyarrowTable])
def test_feature_group_multi_framework(framework):
    """Test feature group works on pandas AND pyarrow."""

    # Generate data
    data = DataGenerator.generate_data(
        n_rows=100,
        numeric_cols=["val1", "val2"]
    )

    # Convert to framework (works with ANY framework!)
    converter = DataConverter()

    # Determine target framework type
    if framework == PandasDataframe:
        import pandas as pd
        framework_type = pd.DataFrame
    elif framework == PyarrowTable:
        import pyarrow as pa
        framework_type = pa.Table

    # Convert - List[Dict] → PyArrow → Target Framework
    framework_data = converter.to_framework(data, framework_type)

    # Test (your feature group logic here)
    # ...

    # Validate structure
    validate_columns_exist(framework_data, ["val1", "val2"])
```

### Example 3: Integration Testing

```python
"""
Integration test using mloda helpers.
"""

from tests.test_plugins.feature_group.test_tooling.integration import MlodaTestHelper
from tests.test_plugins.feature_group.test_tooling.validators import validate_columns_exist
from tests.test_plugins.feature_group.test_tooling.data_generation import DataGenerator
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group import BaseAggregatedFeatureGroup


class TestDataCreator(ATestDataCreator):
    """Simple data creator using tooling."""
    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls):
        # Use data generator (no business logic)
        return DataGenerator.generate_data(
            n_rows=100,
            numeric_cols=["val1", "val2", "val3"]
        )


def test_aggregation_integration():
    """Integration test for aggregation."""

    # Use helper
    results = MlodaTestHelper.run_integration_test(
        features=["sum_aggr__val1", "avg_aggr__val2"],
        feature_groups={BaseAggregatedFeatureGroup},
        compute_frameworks={PandasDataframe},
        additional_plugins={TestDataCreator}
    )

    # Validate
    MlodaTestHelper.assert_result_count(results, 2)
    result = MlodaTestHelper.assert_result_found(results, "sum_aggr__val1")
    validate_columns_exist(result, ["sum_aggr__val1", "avg_aggr__val2"])
```

---

## Implementation Plan

### Phase 1: Core Utilities (Week 1)
**Goal**: Build foundational utilities

**Tasks**:
1. Implement `DataGenerator` and `EdgeCaseGenerator`
2. Implement `FrameworkConverter`
3. Implement structural validators
4. Write tests for utilities themselves
5. Documentation with examples

**Deliverables**:
- Working data generation utilities
- Working framework converters
- Working validators
- README with examples

### Phase 2: Integration Helpers (Week 2)
**Goal**: Add integration testing support

**Tasks**:
1. Implement `MlodaTestHelper`
2. Implement optional base classes
3. Integration examples
4. Documentation updates

**Deliverables**:
- Working integration helpers
- Optional base classes
- Complete documentation

### Phase 3: Pilot Adoption (Week 3-4)
**Goal**: Test with real feature groups

**Tasks**:
1. Pick 2-3 feature groups for pilot
2. Write new tests using tooling
3. Compare with old tests (code reduction, clarity)
4. Gather feedback and iterate

**Deliverables**:
- Pilot tests written with tooling
- Feedback document
- Refined utilities

### Phase 4: Rollout (Week 5-6)
**Goal**: Make tooling available to all

**Tasks**:
1. Publish documentation
2. Add examples for common patterns
3. Provide migration examples
4. Support adoption

**Deliverables**:
- Complete documentation
- Migration guide
- Example repository

---

## Benefits

1. **Zero Business Logic**: Tooling is purely structural
2. **Reusable Utilities**: Generate data, convert frameworks, validate structure
3. **Reduced Boilerplate**: Common patterns centralized
4. **Multi-Framework Support**: Easy to test across pandas, pyarrow, etc.
5. **Flexible**: Use what you need, ignore what you don't
6. **Not Prescriptive**: Doesn't dictate what to test, just provides tools

---

## Non-Goals

1. **NOT** providing business-specific test scenarios
2. **NOT** dictating what feature groups should test
3. **NOT** mandatory base classes (optional only)
4. **NOT** replacing feature-specific test logic

---

## Success Metrics

1. **Adoption**: % of feature group tests using at least one utility
2. **Code Reduction**: Reduction in test boilerplate
3. **Framework Coverage**: Tests easily cover pandas + pyarrow
4. **Developer Feedback**: Utility helpfulness ratings

---

## Appendix: Complete Directory Structure

```
tests/test_plugins/feature_group/test_tooling/
├── __init__.py
├── README.md                               # Usage documentation
├── data_generation/
│   ├── __init__.py
│   ├── generators.py                       # DataGenerator, EdgeCaseGenerator
│   └── examples.py                         # Usage examples
├── converters/
│   ├── __init__.py
│   └── framework_converter.py              # FrameworkConverter
├── validators/
│   ├── __init__.py
│   └── structural_validators.py            # All validators
├── integration/
│   ├── __init__.py
│   └── mloda_helpers.py                    # MlodaTestHelper
├── base/
│   ├── __init__.py
│   └── feature_group_test_base.py          # Optional base class
└── examples/
    ├── __init__.py
    ├── example_data_generation.py          # Example: using data generators
    ├── example_multi_framework.py          # Example: multi-framework testing
    └── example_integration.py              # Example: integration testing
```

---

**End of Design Document v2.0**
