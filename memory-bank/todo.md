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

## Phase 2: Integration Helpers (Week 2) ✅

### 2.1 Integration Test Helpers ✅
**Location**: `tests/test_plugins/feature_group/test_tooling/integration/`

- [x] Create directory structure
- [x] Implement `MlodaTestHelper` class
  - [x] `create_plugin_collector()` - Create PlugInCollector for tests
  - [x] `run_integration_test()` - Run mlodaAPI.run_all() with config
  - [x] `find_result_with_column()` - Find result containing column
  - [x] `assert_result_found()` - Assert result with column exists
  - [x] `count_results()` - Count results
  - [x] `assert_result_count()` - Assert number of results (implemented as count_results)
- [x] Write unit tests for integration helpers
- [x] Write integration tests using the helper
- [x] Document integration patterns

**Acceptance Criteria**:
- Simplifies mlodaAPI test setup
- Works with all feature groups
- Clear error messages
- Examples show real usage

---

### 2.2 Optional Base Classes ✅
**Location**: `tests/test_plugins/feature_group/test_tooling/base/`

- [x] Create directory structure
- [x] Implement `FeatureGroupTestBase` (optional base class)
  - [x] Abstract method: `feature_group_class()`
  - [x] Optional method: `get_test_frameworks()` (deferred - not needed yet)
  - [x] Utility method: `to_pandas()`
  - [x] Utility method: `to_framework()`
  - [x] Utility method: `assert_columns_exist()`
  - [x] Utility method: `assert_row_count()`
  - [x] Utility method: `assert_no_nulls()` (delegates to validators)
  - [x] Utility method: `assert_has_nulls()` (delegates to validators)
- [x] Write example tests using base class
- [x] Document when to use (and when not to use) base class

**Acceptance Criteria**:
- Optional, not mandatory
- Reduces boilerplate for common patterns
- Clear documentation on when to use
- Examples show both with/without base class

---

### 2.3 Examples Module ✅
**Location**: `tests/test_plugins/feature_group/test_tooling/examples/`

- [x] Create `example_data_generation.py` - Show data generation patterns
- [x] Create `example_multi_framework.py` - Show multi-framework testing
- [x] Create `example_integration.py` - Show integration test patterns (combined with multi_framework)
- [x] Create `example_edge_cases.py` - Show edge case testing (included in data_generation)
- [x] Create `example_validators.py` - Show validator usage

**Acceptance Criteria**:
- Examples are runnable
- Cover common use cases
- Show best practices
- Include comments explaining patterns

---

### 2.4 Complete Documentation ✅
- [x] Update README with integration helpers
- [x] Document MlodaTestHelper usage
- [x] Document optional base classes
- [x] Add migration guide from old patterns
- [x] Include troubleshooting section

---

## Phase 3: Pilot Adoption (Week 3-4) - READY TO EXECUTE

**Detailed Plan**: See `/workspace/memory-bank/phase3_migration_plan.md`

### 3.1 Choose Pilot Feature Groups ✅
- [x] Select 2-3 feature groups for pilot
  - [x] Time Window Feature Group (high priority) - Selected
  - [x] Aggregated Feature Group (high priority) - Selected
  - [ ] One additional: Clustering OR Geo Distance (Deferred to Phase 4)

---

### 3.2 Migrate Time Window Tests - PLANNED
**Location**: `tests/test_plugins/feature_group/experimental/test_time_window_feature_group/`

**Status**: Ready to execute (see phase3_migration_plan.md for detailed steps)

- [ ] Phase 3.2.1: Migrate `test_time_window_utils.py` (56 lines → ~15 lines)
- [ ] Phase 3.2.2: Migrate `test_time_window_integration.py` (138 lines → ~80 lines)
- [ ] Phase 3.2.3: Migrate `test_base_time_window_feature_group.py` (200+ lines → ~120 lines)
- [ ] Phase 3.2.4: Migrate `test_pandas_time_window_feature_group.py` (~190 lines → ~100 lines)
- [ ] Phase 3.2.5: Migrate `test_pyarrow_time_window_feature_group.py` (~210 lines → consolidate)
- [ ] Phase 3.2.6: Migrate `test_time_window_feature_parser_integration.py` (~160 lines)
- [ ] Phase 3.2.7: Migrate `test_time_window_with_global_filter.py` (~330 lines)
- [ ] Run `tox` after each migration to ensure no regressions
- [ ] Collect metrics (lines saved, time spent, issues encountered)
- [ ] Document migration experience and lessons learned

**Expected Outcome**:
- 30-50% code reduction
- Improved readability and maintainability
- Standard patterns applied
- All tests continue to pass

---

### 3.3 Migrate Aggregated Feature Group Tests - PLANNED
**Location**: `tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/`

**Status**: Ready to execute after Time Window migration (see phase3_migration_plan.md)

- [ ] Analyze test structure and identify patterns
- [ ] Apply lessons learned from Time Window migration
- [ ] Migrate files incrementally (one at a time)
- [ ] Use same patterns: DataGenerator, DataConverter, validators, MlodaTestHelper
- [ ] Run `tox` after each file migration
- [ ] Collect metrics and compare to Time Window results
- [ ] Document migration experience

**Expected Outcome**:
- Similar 30-50% code reduction
- Validation of reusable patterns
- Refinement of migration approach

---

### 3.4 Migrate Third Pilot Feature Group - DEFERRED
**Status**: Deferred to Phase 4 (Full Rollout)

**Rationale**: Two pilot feature groups (Time Window + Aggregated) provide sufficient data to validate the tooling and approach. Additional pilots can be part of Phase 4.

---

### 3.5 Gather Feedback - PLANNED
**When**: After completing migrations in 3.2 and 3.3

Metrics to collect:
- [ ] **Code reduction**: Lines before vs after for each file
- [ ] **Clarity**: Subjective assessment (better/same/worse)
- [ ] **Time spent**: Hours per file migration
- [ ] **Pain points**: What was difficult or frustrating?
- [ ] **Missing features**: What utilities are needed but don't exist?
- [ ] **Test execution time**: Before vs after (performance check)
- [ ] Create summary document with findings

**Output**: `memory-bank/phase3_feedback.md`

---

### 3.6 Refine Tooling - PLANNED
**When**: After gathering feedback in 3.5

Based on feedback:
- [ ] Identify top 3 missing utilities/features
- [ ] Prioritize fixes vs enhancements
- [ ] Update documentation with lessons learned
- [ ] Add examples for common pain points
- [ ] Consider adding to Phase 5 backlog if not critical

**Output**: Updated tooling or Phase 5 enhancement list

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

**Current Phase**: Phase 3 - Pilot Adoption PLANNED & READY TO EXECUTE ✅

**Completed in Phase 1**:
1. ✅ Data Generation Module - DataGenerator and EdgeCaseGenerator with 10 passing tests
2. ✅ Framework Conversion Module - DataConverter with 4 passing tests
3. ✅ Structural Validators - 9 validators with 5 passing tests
4. ✅ README documentation with examples and patterns
5. ✅ All tests pass (19/19)
6. ✅ Type annotations fixed (mypy clean)

**Completed in Phase 2**:
1. ✅ Integration Test Helpers - MlodaTestHelper with 5 passing tests
2. ✅ Optional Base Classes - FeatureGroupTestBase with 4 passing tests
3. ✅ Example Modules - 3 runnable examples (data_generation, multi_framework, validators)
4. ✅ Updated README with integration helpers and base class documentation
5. ✅ All tests pass (28/28)

**Completed in Phase 3 Planning**:
1. ✅ Selected pilot feature groups (Time Window + Aggregated)
2. ✅ Created detailed migration plan (`phase3_migration_plan.md`)
3. ✅ Documented migration patterns and templates
4. ✅ Defined success criteria and metrics
5. ✅ Risk mitigation strategies in place

**Next Steps** (Ready to Execute):
1. Execute Phase 3.2: Migrate Time Window Feature Group tests (7 files)
2. Execute Phase 3.3: Migrate Aggregated Feature Group tests
3. Gather feedback and metrics (Phase 3.5)
4. Refine tooling based on feedback (Phase 3.6)

**Blockers**: None

**Notes**:
- Design document: `/workspace/docs/design/feature_group_test_tooling.md`
- Design philosophy: Business-agnostic, utility-focused, optional base classes
- All tooling is purely structural with zero business logic
- Test tooling location: `tests/test_plugins/feature_group/test_tooling/`
- Total tests: 28 (all passing)
