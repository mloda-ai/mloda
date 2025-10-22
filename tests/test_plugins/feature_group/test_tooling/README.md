# Feature Group Test Tooling

Business-agnostic test utilities for feature group testing. Provides data generation, framework conversion, and structural validation tools.

**Design Philosophy**: Utilities, not scenarios. Zero business logic. Purely structural testing tools.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Modules](#modules)
  - [Data Generation](#data-generation)
  - [Framework Conversion](#framework-conversion)
  - [Structural Validators](#structural-validators)
- [Common Patterns](#common-patterns)
- [Examples](#examples)

---

## Quick Start

```python
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_row_count,
    validate_columns_exist
)

# Generate test data
data = DataGenerator.generate_data(
    n_rows=100,
    numeric_cols=["value1", "value2"],
    categorical_cols=["category"],
    temporal_col="timestamp"
)

# Convert to pandas
converter = DataConverter()
df = converter.to_framework(data, pd.DataFrame)

# Validate structure
validate_row_count(df, 100)
validate_columns_exist(df, ["value1", "value2", "category"])
```

---

## Modules

### Data Generation

**Location**: `data_generation/generators.py`

Generate test data with specified structural properties (shape, types, edge cases).

#### DataGenerator

```python
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator

# Generate numeric columns
numeric_data = DataGenerator.generate_numeric_columns(
    n_rows=100,
    column_names=["col1", "col2", "col3"],
    value_range=(0, 100),
    dtype="float64",
    seed=42
)

# Generate categorical columns
categorical_data = DataGenerator.generate_categorical_columns(
    n_rows=100,
    column_names=["cat1", "cat2"],
    n_categories=5,  # A, B, C, D, E
    seed=42
)

# Generate temporal column
temporal_data = DataGenerator.generate_temporal_column(
    n_rows=100,
    column_name="timestamp",
    start="2023-01-01",
    freq="D"  # Daily
)

# Generate complete dataset
data = DataGenerator.generate_data(
    n_rows=100,
    numeric_cols=["val1", "val2"],
    categorical_cols=["cat1"],
    temporal_col="timestamp",
    seed=42
)
```

#### EdgeCaseGenerator

```python
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import (
    DataGenerator,
    EdgeCaseGenerator
)

# Base data
base_data = DataGenerator.generate_numeric_columns(
    n_rows=100,
    column_names=["col1", "col2"]
)

# Add nulls to data
data_with_nulls = EdgeCaseGenerator.with_nulls(
    base_data,
    columns=["col1"],
    null_percentage=0.2,  # 20% nulls
    seed=42
)

# Empty dataset
empty_data = EdgeCaseGenerator.empty_data(["col1", "col2", "col3"])

# Single row dataset
single_row = EdgeCaseGenerator.single_row(["col1", "col2"], value=42)

# All null values
all_nulls = EdgeCaseGenerator.all_nulls(["col1", "col2"], n_rows=10)

# Duplicate rows
duplicated = EdgeCaseGenerator.duplicate_rows(base_data, n_duplicates=3)
```

---

### Framework Conversion

**Location**: `converters/data_converter.py`

Convert test data between `List[Dict]` format and any compute framework using the existing `ComputeFrameworkTransformer` system.

**Conversion Path**:
- To framework: `List[Dict] → PyArrow → Target Framework`
- From framework: `Source Framework → PyArrow → List[Dict]`

**Supported Frameworks** (automatic):
- Pandas (`pd.DataFrame`)
- PyArrow (`pa.Table`)
- Polars (`pl.DataFrame`)
- DuckDB
- PythonDict (`list`)
- Any future framework with PyArrow transformer

#### DataConverter

```python
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter
import pandas as pd
import pyarrow as pa

converter = DataConverter()

# Test data
test_data = [
    {"col1": 1, "col2": "a"},
    {"col1": 2, "col2": "b"}
]

# Convert to pandas
df = converter.to_framework(test_data, pd.DataFrame)

# Convert to PyArrow
table = converter.to_framework(test_data, pa.Table)

# Convert back from pandas to List[Dict]
data = converter.from_framework(df, pd.DataFrame)

# Convenience method: convert any framework to pandas
df = converter.to_pandas(table)
```

**Benefits**:
- ✅ Reuses production infrastructure
- ✅ Automatic support for new frameworks
- ✅ Consistent with compute framework test tooling
- ✅ Less code to maintain

---

### Structural Validators

**Location**: `validators/structural_validators.py`

Validate structural properties of data: shape, columns, types, nulls. NO business logic.

```python
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_row_count,
    validate_columns_exist,
    validate_column_count,
    validate_no_nulls,
    validate_has_nulls,
    validate_column_types,
    validate_shape,
    validate_not_empty,
    validate_value_range
)
import pandas as pd
import numpy as np

df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

# Validate row count
validate_row_count(df, 3)

# Validate columns exist
validate_columns_exist(df, ["col1", "col2"])

# Validate column count
validate_column_count(df, 2)

# Validate no nulls in columns
validate_no_nulls(df, ["col1", "col2"])

# Validate some nulls exist (for edge case tests)
# validate_has_nulls(df_with_nulls, ["col1"])

# Validate column types
validate_column_types(df, {"col1": np.int64, "col2": np.int64})

# Validate shape (rows, cols)
validate_shape(df, (3, 2))

# Validate not empty
validate_not_empty(df)

# Validate value range
validate_value_range(df, "col1", min_value=0, max_value=10)
```

**Works with**: Pandas DataFrames, PyArrow Tables, and any framework with `.to_pandas()` method.

---

## Common Patterns

### Pattern 1: Multi-Framework Testing

Test a feature group with multiple compute frameworks.

```python
import pytest
import pandas as pd
import pyarrow as pa
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_row_count,
    validate_columns_exist
)

# Test data
test_data = DataGenerator.generate_data(
    n_rows=100,
    numeric_cols=["feature1", "feature2"],
    categorical_cols=["category"]
)

# Test with multiple frameworks
@pytest.mark.parametrize("framework_type", [pd.DataFrame, pa.Table])
def test_my_feature_group_multi_framework(framework_type):
    converter = DataConverter()

    # Convert to target framework
    input_data = converter.to_framework(test_data, framework_type)

    # Run your feature group
    result = my_feature_group.transform(input_data)

    # Validate structure
    validate_row_count(result, 100)
    validate_columns_exist(result, ["output_col1", "output_col2"])
```

### Pattern 2: Edge Case Testing

Test with edge cases (nulls, empty data, single row, etc.)

```python
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import (
    DataGenerator,
    EdgeCaseGenerator
)
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter

def test_feature_group_handles_nulls():
    # Generate base data
    base_data = DataGenerator.generate_data(
        n_rows=100,
        numeric_cols=["col1", "col2"]
    )

    # Add nulls
    data_with_nulls = EdgeCaseGenerator.with_nulls(
        base_data,
        columns=["col1"],
        null_percentage=0.3
    )

    # Convert to pandas
    converter = DataConverter()
    df = converter.to_framework(data_with_nulls, pd.DataFrame)

    # Test feature group with nulls
    result = my_feature_group.transform(df)

    # Validate handling (depends on your requirements)
    # Either no nulls in output, or nulls handled appropriately
    validate_not_empty(result)

def test_feature_group_empty_data():
    empty_data = EdgeCaseGenerator.empty_data(["col1", "col2"])

    converter = DataConverter()
    df = converter.to_framework(empty_data, pd.DataFrame)

    # Test with empty data
    result = my_feature_group.transform(df)
    validate_row_count(result, 0)
```

### Pattern 3: Integration Testing

Test feature groups with mlodaAPI using MlodaTestHelper.

```python
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
from tests.test_plugins.feature_group.test_tooling.integration.mloda_test_helper import MlodaTestHelper

def test_feature_group_integration():
    # Generate test data
    test_data = DataGenerator.generate_data(
        n_rows=100,
        numeric_cols=["value1", "value2"],
        categorical_cols=["category"]
    )

    # Configure mloda
    config = {
        "features": [
            {
                "name": "my_feature",
                "feature_group": "MyFeatureGroup",
                # ... feature group config
            }
        ]
    }

    # Run integration test using helper
    helper = MlodaTestHelper()
    results = helper.run_integration_test(config, test_data)

    # Validate results using helper methods
    helper.assert_result_found(results, "my_feature")
    assert helper.count_results(results) > 0

    # Find specific result
    result = helper.find_result_with_column(results, "my_feature")
    assert result is not None
```

### Pattern 4: Using Optional Base Class

Simplify common test patterns with FeatureGroupTestBase.

```python
from tests.test_plugins.feature_group.test_tooling.base.feature_group_test_base import FeatureGroupTestBase
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
import pandas as pd

class TestMyFeatureGroup(FeatureGroupTestBase):
    """Test suite for MyFeatureGroup using base class utilities."""

    def feature_group_class(self):
        """Return the feature group class being tested."""
        return MyFeatureGroup

    def test_basic_functionality(self):
        # Generate test data
        data = DataGenerator.generate_data(
            n_rows=50,
            numeric_cols=["col1", "col2"]
        )

        # Use base class utility to convert to pandas
        df = self.to_pandas(data)

        # Run feature group
        result = self.feature_group_class().transform(df)

        # Use base class assertion helpers
        self.assert_row_count(result, 50)
        self.assert_columns_exist(result, ["output_col"])

    def test_multi_framework(self):
        # Generate test data
        data = DataGenerator.generate_data(n_rows=20, numeric_cols=["col1"])

        # Test with pandas
        df = self.to_framework(data, pd.DataFrame)
        result = self.feature_group_class().transform(df)
        self.assert_row_count(result, 20)
```

---

---

## Integration Helpers

**Location**: `integration/mloda_test_helper.py`

Simplify integration testing with mlodaAPI.

### MlodaTestHelper

```python
from tests.test_plugins.feature_group.test_tooling.integration.mloda_test_helper import MlodaTestHelper

helper = MlodaTestHelper()

# Create plugin collector for tests
collector = helper.create_plugin_collector()

# Run integration test
config = {"features": [...]}
test_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
results = helper.run_integration_test(config, test_data)

# Find result with specific column
result = helper.find_result_with_column(results, "output_column")

# Assert result exists
helper.assert_result_found(results, "output_column")

# Count results
count = helper.count_results(results)
```

**Methods**:
- `create_plugin_collector()` - Create PluginLoader for tests
- `run_integration_test(config, data)` - Run mlodaAPI with config and data
- `find_result_with_column(results, column_name)` - Find result containing column
- `assert_result_found(results, column_name)` - Assert result with column exists
- `count_results(results)` - Count number of results

---

## Optional Base Classes

**Location**: `base/feature_group_test_base.py`

Optional base class to reduce boilerplate in feature group tests.

### FeatureGroupTestBase

```python
from tests.test_plugins.feature_group.test_tooling.base.feature_group_test_base import FeatureGroupTestBase

class TestMyFeatureGroup(FeatureGroupTestBase):
    def feature_group_class(self):
        """Required: Return the feature group class being tested."""
        return MyFeatureGroup

    def test_something(self):
        # Use inherited utility methods
        df = self.to_pandas(data)
        result = self.to_framework(data, pd.DataFrame)
        self.assert_columns_exist(result, ["col1"])
        self.assert_row_count(result, 10)
```

**Abstract Methods**:
- `feature_group_class()` - Return the feature group class (required)

**Utility Methods**:
- `to_pandas(data)` - Convert data to pandas DataFrame
- `to_framework(data, framework_type)` - Convert data to target framework
- `assert_columns_exist(result, columns)` - Assert columns exist
- `assert_row_count(result, count)` - Assert row count matches

**When to use**:
- ✅ When you have multiple tests for a single feature group
- ✅ When you want to reduce boilerplate
- ✅ When you want consistent patterns across test suites

**When NOT to use**:
- ❌ For simple, one-off tests
- ❌ When you need more flexibility
- ❌ When inheritance doesn't fit your test structure

---

## Examples

See the `examples/` directory for complete runnable examples:

- `example_data_generation.py` - Data generation patterns
- `example_multi_framework.py` - Multi-framework testing
- `example_validators.py` - Validator usage

Run examples:
```bash
python -m tests.test_plugins.feature_group.test_tooling.examples.example_data_generation
python -m tests.test_plugins.feature_group.test_tooling.examples.example_multi_framework
python -m tests.test_plugins.feature_group.test_tooling.examples.example_validators
```

---

## Best Practices

1. **Keep tests business-agnostic**: Focus on structure, not semantics
2. **Test multiple frameworks**: Use parametrize with different framework types
3. **Test edge cases**: Always test empty data, nulls, single row
4. **Use descriptive names**: Test names should describe what is being validated
5. **Reuse utilities**: Don't write custom data generation when these utilities exist
6. **Clear assertions**: Use validators with clear error messages
7. **Seed random data**: Always use seed parameter for reproducibility

---

## Migration from Old Patterns

If you have existing tests with custom data generation:

**Before**:
```python
# Custom data generation
df = pd.DataFrame({
    "col1": np.random.rand(100),
    "col2": np.random.choice(["A", "B", "C"], 100),
    "timestamp": pd.date_range("2023-01-01", periods=100)
})
```

**After**:
```python
# Using test tooling
data = DataGenerator.generate_data(
    n_rows=100,
    numeric_cols=["col1"],
    categorical_cols=["col2"],
    temporal_col="timestamp",
    seed=42  # Reproducible!
)
converter = DataConverter()
df = converter.to_framework(data, pd.DataFrame)
```

---

## Troubleshooting

### ImportError: PyArrow is required

**Solution**: Install PyArrow: `pip install pyarrow`

### KeyError: No transformer found

**Solution**: Ensure plugins are loaded:
```python
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
PluginLoader().load_group("compute_framework")
```

### AssertionError: Missing columns

Check the actual columns returned:
```python
print(f"Available columns: {result.columns}")
```

---

## Contributing

When adding new utilities:

1. Keep them business-agnostic
2. Add unit tests
3. Update this README with examples
4. Document parameters and return types
5. Include docstring examples

---

## Design Document

For detailed design rationale, see: `/workspace/docs/design/feature_group_test_tooling.md`
