# Phase 3: Pilot Adoption - Migration Plan

**Created**: 2025-10-22
**Status**: Ready to Execute

---

## Overview

Phase 3 involves migrating existing feature group tests to use the new test tooling infrastructure built in Phases 1 & 2. This plan provides a step-by-step approach for pilot adoption.

---

## Selected Pilot Feature Groups

Based on the todo.md priorities:

1. **Time Window Feature Group** (High Priority)
   - Location: `tests/test_plugins/feature_group/experimental/test_time_window_feature_group/`
   - ~1100 lines of test code across 7 test files
   - Has custom data creators and validators (good candidates for replacement)

2. **Aggregated Feature Group** (High Priority)
   - Location: `tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/`
   - Similar patterns to Time Window
   - Good candidate for demonstrating reusability

---

## Migration Strategy

### Step 1: Analyze Current Tests

For each pilot feature group:

1. **Identify patterns to replace:**
   - Custom data generation (replace with `DataGenerator`)
   - Manual DataFrame creation (replace with `DataConverter`)
   - Custom validators (replace with structural validators)
   - Repeated test setup (consolidate with `FeatureGroupTestBase`)

2. **Measure baseline:**
   - Count lines of code
   - Count number of tests
   - Identify duplicated code

3. **Document findings:**
   - What patterns are most common?
   - What pain points exist?
   - What edge cases are tested?

### Step 2: Create Migration Template

Before migrating, create a template showing:

**Before (Old Pattern):**
```python
# Custom data creation
dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
data = {
    "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24],
    "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55],
    "reference_time": dates,
}
df = pd.DataFrame(data)

# Custom validation
assert "feature_name" in result.columns
assert len(result) == 10
```

**After (New Pattern):**
```python
# Using test tooling
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_columns_exist,
    validate_row_count
)

test_data = DataGenerator.generate_data(
    n_rows=10,
    numeric_cols=["temperature", "humidity"],
    temporal_col="reference_time",
    seed=42
)

converter = DataConverter()
df = converter.to_framework(test_data, pd.DataFrame)

# Structural validation
validate_columns_exist(result, ["feature_name"])
validate_row_count(result, 10)
```

### Step 3: Incremental Migration Approach

**DO NOT** migrate all tests at once. Instead:

1. **Start with utilities file:**
   - Migrate `test_time_window_utils.py` first
   - Replace `TimeWindowTestDataCreator` with `DataGenerator`
   - Replace `validate_time_window_features` with structural validators
   - Run tests to ensure no regressions

2. **Migrate one test file:**
   - Choose the simplest test file (e.g., `test_time_window_integration.py`)
   - Refactor to use new tooling
   - Run `pytest` on that file only
   - Document lines of code saved

3. **Migrate remaining files one by one:**
   - After each file migration, run full test suite
   - Document improvements and pain points
   - Adjust tooling if needed

4. **Consolidate if beneficial:**
   - Consider using `FeatureGroupTestBase` if multiple tests share setup
   - Only consolidate if it genuinely reduces complexity

### Step 4: Validation Checklist

After each migration:

- [ ] All tests still pass (`pytest <file>`)
- [ ] No reduction in test coverage
- [ ] Code is more readable (subjective but important)
- [ ] Less duplicated code
- [ ] Run full test suite (`tox`) to ensure no regressions
- [ ] Count lines of code before/after
- [ ] Document any issues encountered

### Step 5: Metrics Collection

Track for each migrated feature group:

**Quantitative:**
- Lines of code: Before vs After
- Number of custom utilities removed
- Number of duplicated patterns eliminated
- Test execution time: Before vs After

**Qualitative:**
- Readability improvements
- Ease of adding new tests
- Pain points encountered
- Missing utilities (add to Phase 5 enhancements)

---

## Migration Order

### Phase 3.1: Time Window Feature Group

**Priority**: High
**Estimated Effort**: 2-3 days

**Files to migrate (in order):**

1. **`test_time_window_utils.py`** (56 lines)
   - Replace: `TimeWindowTestDataCreator` → `DataGenerator`
   - Replace: `validate_time_window_features` → `validate_columns_exist`
   - Expected reduction: ~30 lines → ~15 lines

2. **`test_time_window_integration.py`** (138 lines)
   - Use: `MlodaTestHelper` for integration tests
   - Use: `DataGenerator` for test data
   - Expected reduction: ~138 lines → ~80 lines

3. **`test_base_time_window_feature_group.py`** (200+ lines)
   - Consider: Using `FeatureGroupTestBase` if beneficial
   - Use: `DataGenerator`, validators, `DataConverter`
   - Expected reduction: ~200 lines → ~120 lines

4. **`test_pandas_time_window_feature_group.py`** (~190 lines)
   - Use: Multi-framework testing patterns
   - Use: `DataConverter` for framework conversion
   - Expected reduction: ~190 lines → ~100 lines

5. **`test_pyarrow_time_window_feature_group.py`** (~210 lines)
   - Similar to pandas migration
   - Consolidate with pandas tests using `@pytest.mark.parametrize` if possible

6. **`test_time_window_feature_parser_integration.py`** (~160 lines)
   - Use: `MlodaTestHelper`
   - Use: `DataGenerator`

7. **`test_time_window_with_global_filter.py`** (~330 lines)
   - Largest file - migrate last
   - Apply all learned patterns

**After each file:**
- Run: `pytest tests/test_plugins/feature_group/experimental/test_time_window_feature_group/`
- Validate: All tests pass
- Run: `tox` to ensure no global regressions

### Phase 3.2: Aggregated Feature Group

**Priority**: High
**Estimated Effort**: 2-3 days

Apply same approach as Time Window:
1. Analyze test structure
2. Identify patterns to replace
3. Migrate incrementally
4. Collect metrics

---

## Common Migration Patterns

### Pattern 1: Replace Custom Data Creation

**Before:**
```python
data = {
    "col1": [1, 2, 3, 4, 5],
    "col2": [10, 20, 30, 40, 50],
    "col3": ["A", "B", "C", "A", "B"],
}
df = pd.DataFrame(data)
```

**After:**
```python
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter

data = DataGenerator.generate_data(
    n_rows=5,
    numeric_cols=["col1", "col2"],
    categorical_cols=["col3"],
    seed=42  # Reproducible!
)
converter = DataConverter()
df = converter.to_framework(data, pd.DataFrame)
```

**Benefits:**
- Reproducible (seed parameter)
- Less verbose
- Standard pattern across all tests

### Pattern 2: Replace Custom Validators

**Before:**
```python
assert "feature_name" in result.columns, "Feature not found"
assert len(result) == expected_rows, f"Expected {expected_rows} rows"
assert result["feature_name"].notna().all(), "Found nulls"
```

**After:**
```python
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_columns_exist,
    validate_row_count,
    validate_no_nulls
)

validate_columns_exist(result, ["feature_name"])
validate_row_count(result, expected_rows)
validate_no_nulls(result, ["feature_name"])
```

**Benefits:**
- Works with any framework (pandas, pyarrow, etc.)
- Clear error messages
- Reusable across all tests

### Pattern 3: Multi-Framework Testing

**Before (Separate test files for pandas and pyarrow):**
```python
# test_pandas_feature_group.py
def test_feature_with_pandas():
    df_pandas = pd.DataFrame(...)
    result = feature_group.transform(df_pandas)
    assert ...

# test_pyarrow_feature_group.py
def test_feature_with_pyarrow():
    table = pa.table(...)
    result = feature_group.transform(table)
    assert ...
```

**After (Single parametrized test):**
```python
import pytest
import pandas as pd
import pyarrow as pa

from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter

@pytest.mark.parametrize("framework_type", [pd.DataFrame, pa.Table])
def test_feature_multi_framework(framework_type):
    # Generate test data once
    test_data = DataGenerator.generate_data(n_rows=10, numeric_cols=["col1"])

    # Convert to target framework
    converter = DataConverter()
    data = converter.to_framework(test_data, framework_type)

    # Test (validators work with any framework!)
    result = feature_group.transform(data)
    validate_row_count(result, 10)
```

**Benefits:**
- Single test covers multiple frameworks
- Reduces code duplication
- Easier to maintain

### Pattern 4: Integration Testing

**Before:**
```python
from mloda_core.mloda_api import mlodaAPI

def test_integration():
    df = pd.DataFrame(...)
    config = {...}
    results = mlodaAPI.run_all(config, df)

    # Manual result finding
    found = False
    for result in results:
        if "output_col" in result.columns:
            found = True
            break
    assert found
```

**After:**
```python
from tests.test_plugins.feature_group.test_tooling.integration.mloda_test_helper import MlodaTestHelper
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator

def test_integration():
    test_data = DataGenerator.generate_data(n_rows=10, numeric_cols=["col1"])
    config = {...}

    helper = MlodaTestHelper()
    results = helper.run_integration_test(config, test_data)

    # Use helper methods
    helper.assert_result_found(results, "output_col")
```

**Benefits:**
- Cleaner integration test setup
- Helper methods for common operations
- Consistent pattern across all integration tests

---

## Migration Checklist Template

Use this checklist for each file migration:

### Pre-Migration
- [ ] Read file and understand test structure
- [ ] Count lines of code: _____ lines
- [ ] Count number of tests: _____ tests
- [ ] Identify custom utilities used
- [ ] Run tests to establish baseline: `pytest <file> -v`

### During Migration
- [ ] Replace custom data generation with `DataGenerator`
- [ ] Replace manual DataFrame creation with `DataConverter`
- [ ] Replace custom validators with structural validators
- [ ] Consider using `FeatureGroupTestBase` if beneficial
- [ ] Consider consolidating duplicate setup code
- [ ] Update imports

### Post-Migration
- [ ] Run migrated file tests: `pytest <file> -v`
- [ ] All tests pass: YES / NO
- [ ] Count lines of code after: _____ lines
- [ ] Reduction: _____ lines (_____ %)
- [ ] Run full test suite: `tox`
- [ ] No regressions: YES / NO
- [ ] Code more readable: YES / NO / NEUTRAL
- [ ] Document any issues or missing features

### Metrics
- **Lines of code reduction**: _____ %
- **Number of custom utilities removed**: _____
- **Time spent on migration**: _____ hours
- **Issues encountered**: _____

---

## Risk Mitigation

### Risk 1: Breaking Tests During Migration
**Mitigation:**
- Migrate one file at a time
- Run tests after each file
- Keep git commits granular
- Easy to rollback if needed

### Risk 2: New Tooling Missing Features
**Mitigation:**
- Document missing features immediately
- Add to Phase 5 enhancement backlog
- Consider temporary workarounds
- Update tooling if critical

### Risk 3: Time-Consuming Migration
**Mitigation:**
- Set time limits per file (e.g., 1-2 hours max)
- If migration takes too long, document why
- May indicate tooling gaps or complex test structure
- Pause and reassess if needed

### Risk 4: Reduced Readability
**Mitigation:**
- Compare before/after for each file
- If new version is less clear, document why
- May need to adjust patterns or tooling
- Readability is more important than line count

---

## Success Criteria

### Phase 3 Complete When:

1. **Time Window Feature Group migrated:**
   - All 7 test files updated
   - All tests pass
   - Code reduction achieved (target: 30%+)
   - Metrics documented

2. **Aggregated Feature Group migrated:**
   - All test files updated
   - All tests pass
   - Code reduction achieved (target: 30%+)
   - Metrics documented

3. **Feedback collected:**
   - Developer experience survey
   - Pain points documented
   - Missing features identified
   - Recommendations for Phase 4/5

4. **Documentation updated:**
   - Migration guide created
   - Best practices documented
   - Common pitfalls noted
   - Examples added to README

---

## Next Steps After Phase 3

Based on pilot results:

1. **Gather Feedback:**
   - What worked well?
   - What was difficult?
   - What's missing from tooling?
   - Would you use this for new tests?

2. **Refine Tooling (Phase 5):**
   - Add missing utilities
   - Improve documentation
   - Fix discovered issues
   - Add more examples

3. **Plan Full Rollout (Phase 4):**
   - Prioritize remaining feature groups
   - Estimate effort based on pilot
   - Create rollout schedule
   - Assign tasks

---

## Appendix: Quick Reference

### Key Test Tooling Imports

```python
# Data generation
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import (
    DataGenerator,
    EdgeCaseGenerator,
)

# Framework conversion
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter

# Validators
from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_row_count,
    validate_columns_exist,
    validate_column_count,
    validate_no_nulls,
    validate_has_nulls,
    validate_shape,
    validate_not_empty,
    validate_value_range,
)

# Integration helpers
from tests.test_plugins.feature_group.test_tooling.integration.mloda_test_helper import MlodaTestHelper

# Optional base class
from tests.test_plugins.feature_group.test_tooling.base.feature_group_test_base import FeatureGroupTestBase
```

### Common Patterns

See: `/workspace/tests/test_plugins/feature_group/test_tooling/README.md`

### Examples

Run examples:
```bash
python -m tests.test_plugins.feature_group.test_tooling.examples.example_data_generation
python -m tests.test_plugins.feature_group.test_tooling.examples.example_multi_framework
python -m tests.test_plugins.feature_group.test_tooling.examples.example_validators
```

---

**Ready to Execute**: This plan is ready for implementation. Start with Time Window Feature Group, `test_time_window_utils.py` file.
