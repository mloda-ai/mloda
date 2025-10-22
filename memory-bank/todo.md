# Todo List - Feature Group Test Tooling Implementation

## Overview
Implementing business-agnostic test tooling for feature groups based on the design document at `/workspace/docs/design/feature_group_test_tooling.md`.

**Design Philosophy**: Provide utilities (not scenarios). Zero business logic. Purely structural testing tools.

---

## Phase 1: Core Utilities (Week 1)

### 1.1 Data Generation Module ✅
**Location**: `tests/test_plugins/feature_group/test_tooling/data_generation/`

- [x] Create directory structure
- [x] Implement `DataGenerator` class
  - [x] `generate_numeric_columns()` - Generate N numeric columns with random values
  - [x] `generate_categorical_columns()` - Generate categorical columns (A, B, C, ...)
  - [x] `generate_temporal_column()` - Generate timestamp columns
  - [x] `generate_data()` - Master method combining all column types
- [x] Implement `EdgeCaseGenerator` class
  - [x] `with_nulls()` - Add null values to existing data
  - [x] `empty_data()` - Generate empty datasets
  - [x] `single_row()` - Generate single-row datasets
  - [x] `all_nulls()` - Generate datasets with all nulls
  - [x] `duplicate_rows()` - Duplicate rows in dataset
- [x] Write unit tests for data generators
- [x] Document with examples

**Acceptance Criteria**:
- Can generate data with any shape (N rows, M columns)
- Supports numeric, categorical, temporal types
- No business-specific logic
- All generators have unit tests
- Documentation includes usage examples

---

### 1.2 Framework Conversion Module ✅
**Location**: `tests/test_plugins/feature_group/test_tooling/converters/`

**Approach**: Reuse existing `ComputeFrameworkTransformer` infrastructure (like compute framework test tooling does)

- [x] Create directory structure
- [x] Implement `DataConverter` class (based on compute framework pattern)
  - [x] `__init__()` - Initialize with ComputeFrameworkTransformer
  - [x] `to_framework()` - Convert List[Dict] → target framework via PyArrow
    - Path: List[Dict] → PyArrow → Target Framework
    - Uses existing transformer_map
  - [x] `from_framework()` - Convert framework → List[Dict] via PyArrow
    - Path: Source Framework → PyArrow → List[Dict]
    - Uses existing transformer_map
  - [x] Handle special cases (list, PyArrow)
- [x] Write unit tests for converter
- [x] Test with all supported frameworks (Pandas, PyArrow, Polars, DuckDB, etc.)
- [x] Document conversion approach

**Acceptance Criteria**:
- Reuses existing ComputeFrameworkTransformer system
- Automatically supports ALL frameworks with PyArrow transformers
- No duplicate conversion logic
- Handles edge cases (empty data, nulls, special types)
- Tests cover multiple frameworks
- Clear error messages for unsupported frameworks

**Benefits of this approach**:
- ✅ Reuses production infrastructure
- ✅ Automatic support for new frameworks
- ✅ Consistent with compute framework test tooling
- ✅ Less code to maintain

---

### 1.3 Structural Validators Module ✅
**Location**: `tests/test_plugins/feature_group/test_tooling/validators/`

- [x] Create directory structure
- [x] Implement structural validators
  - [x] `validate_row_count()` - Check number of rows
  - [x] `validate_columns_exist()` - Check columns exist
  - [x] `validate_column_count()` - Check number of columns
  - [x] `validate_no_nulls()` - Check columns have no nulls
  - [x] `validate_has_nulls()` - Check columns have some nulls
  - [x] `validate_column_types()` - Check column data types
  - [x] `validate_shape()` - Check (rows, cols) shape
  - [x] `validate_not_empty()` - Check data not empty
  - [x] `validate_value_range()` - Check values within range
- [x] Write unit tests for all validators
- [x] Test with pandas and pyarrow data
- [x] Document validator usage

**Acceptance Criteria**:
- All validators work with pandas and pyarrow
- Clear, helpful error messages
- No business logic in validators
- Comprehensive test coverage

---

### 1.4 Documentation ✅
- [x] Create `tests/test_plugins/feature_group/test_tooling/README.md`
- [x] Document DataGenerator with examples
- [x] Document FrameworkConverter with examples
- [x] Document validators with examples
- [x] Include common patterns and recipes

---

## Phase 2: Integration Helpers (Week 2)

### 2.1 Integration Test Helpers
**Location**: `tests/test_plugins/feature_group/test_tooling/integration/`

- [ ] Create directory structure
- [ ] Implement `MlodaTestHelper` class
  - [ ] `create_plugin_collector()` - Create PlugInCollector for tests
  - [ ] `run_integration_test()` - Run mlodaAPI.run_all() with config
  - [ ] `find_result_with_column()` - Find result containing column
  - [ ] `assert_result_found()` - Assert result with column exists
  - [ ] `count_results()` - Count results
  - [ ] `assert_result_count()` - Assert number of results
- [ ] Write unit tests for integration helpers
- [ ] Write integration tests using the helper
- [ ] Document integration patterns

**Acceptance Criteria**:
- Simplifies mlodaAPI test setup
- Works with all feature groups
- Clear error messages
- Examples show real usage

---

### 2.2 Optional Base Classes
**Location**: `tests/test_plugins/feature_group/test_tooling/base/`

- [ ] Create directory structure
- [ ] Implement `FeatureGroupTestBase` (optional base class)
  - [ ] Abstract method: `feature_group_class()`
  - [ ] Optional method: `get_test_frameworks()`
  - [ ] Utility method: `to_pandas()`
  - [ ] Utility method: `to_framework()`
  - [ ] Utility method: `assert_columns_exist()`
  - [ ] Utility method: `assert_row_count()`
  - [ ] Utility method: `assert_no_nulls()`
  - [ ] Utility method: `assert_has_nulls()`
- [ ] Write example tests using base class
- [ ] Document when to use (and when not to use) base class

**Acceptance Criteria**:
- Optional, not mandatory
- Reduces boilerplate for common patterns
- Clear documentation on when to use
- Examples show both with/without base class

---

### 2.3 Examples Module
**Location**: `tests/test_plugins/feature_group/test_tooling/examples/`

- [ ] Create `example_data_generation.py` - Show data generation patterns
- [ ] Create `example_multi_framework.py` - Show multi-framework testing
- [ ] Create `example_integration.py` - Show integration test patterns
- [ ] Create `example_edge_cases.py` - Show edge case testing
- [ ] Create `example_validators.py` - Show validator usage

**Acceptance Criteria**:
- Examples are runnable
- Cover common use cases
- Show best practices
- Include comments explaining patterns

---

### 2.4 Complete Documentation
- [ ] Update README with integration helpers
- [ ] Document MlodaTestHelper usage
- [ ] Document optional base classes
- [ ] Add migration guide from old patterns
- [ ] Include troubleshooting section

---

## Phase 3: Pilot Adoption (Week 3-4)

### 3.1 Choose Pilot Feature Groups
- [ ] Select 2-3 feature groups for pilot
  - [ ] Time Window Feature Group (high priority)
  - [ ] Aggregated Feature Group (high priority)
  - [ ] One additional: Clustering OR Geo Distance

---

### 3.2 Migrate Time Window Tests
**Location**: `tests/test_plugins/feature_group/experimental/test_time_window_feature_group/`

- [ ] Identify tests to migrate
- [ ] Rewrite using DataGenerator
- [ ] Rewrite using FrameworkConverter
- [ ] Rewrite using validators
- [ ] Rewrite integration tests using MlodaTestHelper
- [ ] Compare old vs new (lines of code, clarity)
- [ ] Run `tox` to ensure all tests pass
- [ ] Document migration experience

---

### 3.3 Migrate Aggregated Feature Group Tests
**Location**: `tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/`

- [ ] Identify tests to migrate
- [ ] Rewrite using tooling utilities
- [ ] Compare old vs new
- [ ] Run `tox` to ensure all tests pass
- [ ] Document migration experience

---

### 3.4 Migrate Third Pilot Feature Group
- [ ] Identify tests to migrate
- [ ] Rewrite using tooling utilities
- [ ] Compare old vs new
- [ ] Run `tox` to ensure all tests pass
- [ ] Document migration experience

---

### 3.5 Gather Feedback
- [ ] Code reduction metrics (% lines saved)
- [ ] Clarity improvements
- [ ] Pain points encountered
- [ ] Missing utilities/features
- [ ] Developer feedback survey
- [ ] Create feedback document

---

### 3.6 Refine Tooling
- [ ] Address feedback issues
- [ ] Add missing utilities
- [ ] Improve documentation
- [ ] Fix bugs discovered
- [ ] Update examples

---

## Phase 4: Full Rollout (Week 5-10)

### 4.1 Migration Plan
- [ ] Create prioritized list of remaining feature groups
- [ ] Estimate effort per feature group
- [ ] Create migration schedule
- [ ] Assign migration tasks

---

### 4.2 Migrate Remaining Feature Groups

#### Clustering
- [ ] Migrate clustering feature group tests
- [ ] Run `tox`
- [ ] Document changes

#### Geo Distance
- [ ] Migrate geo distance feature group tests
- [ ] Run `tox`
- [ ] Document changes

#### Dimensionality Reduction
- [ ] Migrate dimensionality reduction tests
- [ ] Run `tox`
- [ ] Document changes

#### Node Centrality
- [ ] Migrate node centrality tests
- [ ] Run `tox`
- [ ] Document changes

#### Forecasting
- [ ] Migrate forecasting tests
- [ ] Run `tox`
- [ ] Document changes

#### Sklearn Groups
- [ ] Migrate sklearn encoding tests
- [ ] Migrate sklearn scaling tests
- [ ] Migrate sklearn pipeline tests
- [ ] Run `tox`
- [ ] Document changes

#### Text Cleaning
- [ ] Migrate text cleaning tests
- [ ] Run `tox`
- [ ] Document changes

#### Data Quality / Missing Value
- [ ] Migrate missing value tests
- [ ] Run `tox`
- [ ] Document changes

---

### 4.3 Cleanup
- [ ] Review all migrated tests
- [ ] Remove old/duplicate test utilities
- [ ] Consolidate scattered validators
- [ ] Clean up conftest.py files
- [ ] Update all test documentation
- [ ] Run final `tox` on entire test suite

---

### 4.4 Final Documentation
- [ ] Complete README with all patterns
- [ ] Migration guide for future feature groups
- [ ] Best practices document
- [ ] API reference
- [ ] Troubleshooting guide
- [ ] Add to project documentation

---

## Phase 5: Continuous Improvement (Ongoing)

### 5.1 Monitoring
- [ ] Track adoption metrics
- [ ] Monitor test execution times
- [ ] Collect developer feedback
- [ ] Track bugs/issues

---

### 5.2 Enhancements
- [ ] Add utilities as needed
- [ ] Performance optimizations
- [ ] Support for new compute frameworks
- [ ] Enhanced validators
- [ ] Additional examples

---

### 5.3 Maintenance
- [ ] Keep documentation updated
- [ ] Update examples for new patterns
- [ ] Refactor as needed
- [ ] Bug fixes
- [ ] Version updates

---

## Success Metrics

### Quantitative
- [ ] **Code Reduction**: Measure % reduction in test code lines
  - Target: 30-50% reduction
  - Track per feature group
- [ ] **Test Coverage**: Maintain or improve
  - Target: 100% coverage maintained
- [ ] **Test Execution Time**: Should not increase
  - Target: Maintain or improve by 10%
- [ ] **Adoption Rate**: % of feature groups using tooling
  - Target: 100% by end of Phase 4

### Qualitative
- [ ] Developer satisfaction survey
- [ ] Code review feedback
- [ ] Onboarding time improvements
- [ ] Bug discovery improvements

---

## Current Status

**Last Updated**: 2025-10-22

**Current Phase**: Phase 1 - Core Utilities COMPLETE ✅

**Completed in Phase 1**:
1. ✅ Data Generation Module - DataGenerator and EdgeCaseGenerator with 10 passing tests
2. ✅ Framework Conversion Module - DataConverter with 4 passing tests
3. ✅ Structural Validators - 9 validators with 5 passing tests
4. ✅ README documentation with examples and patterns
5. ✅ All tests pass (19/19)
6. ✅ Type annotations fixed (mypy clean)

**Next Steps**:
1. Begin Phase 2.1 - Integration Test Helpers
2. Implement MlodaTestHelper class
3. Create optional base classes

**Blockers**: None

**Notes**:
- Design document: `/workspace/docs/design/feature_group_test_tooling.md`
- Design philosophy: Business-agnostic, utility-focused, optional base classes
- All tooling is purely structural with zero business logic
- Test tooling location: `tests/test_plugins/feature_group/test_tooling/`
