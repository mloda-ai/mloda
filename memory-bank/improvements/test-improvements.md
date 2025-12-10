# Test Code Improvements

This document identifies 10 targeted improvements for the test codebase. Each improvement may break backwards compatibility but will result in cleaner, more maintainable tests.

---

## 1. Migrate unittest.TestCase Classes to Pure Pytest

- [x] Identify all `unittest.TestCase` classes (17 found, not 37)
- [x] Convert `setUp`/`tearDown` to `setup_method`/`teardown_method`
- [x] Convert `setUpClass`/`tearDownClass` to `setup_class`/`teardown_class`
- [x] Replace `self.assertEqual` with `assert ==`
- [x] Replace `self.assertRaises` with `pytest.raises`
- [x] Replace `self.assertAlmostEqual` with `pytest.approx`
- [x] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Migrated 13 unittest.TestCase classes to pure pytest style (keeping class structure, removing inheritance):

| Difficulty | Files Migrated |
|------------|----------------|
| Low | `test_python_version.py`, `test_compute_framework_availability.py`, `test_plugin_collector.py`, `test_feature_group_version.py` |
| Medium | `test_single_filter.py`, `test_global_filter.py`, `test_read_db.py`, `test_forecasting_feature_group.py` |
| Higher | `test_geo_distance.py`, `test_implementations.py`, `test_add_index.py`, `test_read.py`, `test_flight_server.py` |

Key transformations applied:
- `class TestX(unittest.TestCase):` → `class TestX:`
- `setUp` → `setup_method`
- `tearDown` → `teardown_method`
- `setUpClass` → `setup_class`
- `tearDownClass` → `teardown_class`
- `self.assertEqual(a, b)` → `assert a == b`
- `self.assertTrue(x)` → `assert x`
- `self.assertFalse(x)` → `assert not x`
- `self.assertIn(a, b)` → `assert a in b`
- `self.assertIsInstance(a, T)` → `assert isinstance(a, T)`
- `self.assertRaises(E)` → `pytest.raises(E)`
- `self.assertAlmostEqual(a, b, delta=d)` → `assert a == pytest.approx(b, abs=d)`
- `self.skipTest()` → `pytest.skip()`

**Why:** The codebase mixed unittest and pytest styles. This created inconsistency and prevented leveraging pytest's superior fixtures, parametrization, and assertion introspection. Pure pytest tests are more readable (`assert x == y` vs `self.assertEqual(x, y)`) and provide better failure messages without extra code.

| Pros | Cons |
|------|------|
| Consistent test style across codebase | Requires touching test classes |
| Better assertion failure messages | Developers familiar with unittest need adjustment |
| Can use pytest fixtures everywhere | Some complex setUp patterns may need rethinking |
| Removes unittest import dependency | One-time migration effort |

**Files migrated:** 13 test files across `tests/test_core/` and `tests/test_plugins/`

---

## 2. Convert Class-Level Test Data to Fixtures

- [x] Identify classes with shared class-level state
- [x] Create pytest fixtures for `left_data`, `right_data`, `expected_data`
- [x] Remove class attributes, inject via fixture parameters
- [x] Ensure each test gets fresh data instances
- [x] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Converted 4 compute framework test classes from class-level attributes to pytest fixtures:

| Test Class | File |
|------------|------|
| `TestPandasDataFrameComputeFramework` | `pandas/test_pandas_dataframe.py` |
| `TestPolarsDataFrameComputeFramework` | `polars/test_polars_dataframe.py` |
| `TestPyArrowTableComputeFramework` | `pyarrow/test_pyarrow_table.py` |
| `TestDuckDBFrameworkComputeFramework` | `duckdb/test_duckdb_framework.py` |

Key transformations applied:
- Class-level `framework_instance` → `@pytest.fixture` returning fresh instance
- Class-level `dict_data` → `@pytest.fixture` returning fresh dictionary
- Class-level `expected_data` → `@pytest.fixture` using dict_data fixture
- Removed unused `left_data`, `right_data`, `idx` attributes (merge tests use `DataFrameTestBase`)
- Test methods now receive fixtures as parameters
- Removed unused `assert_series_equal` import from pandas tests

**Note:** Spark framework tests were already using fixtures via `spark_session` parameter.

**Why:** Tests like `TestPandasDataFrameComputeFramework` define shared class attributes (`left_data`, `right_data`) that persist across test methods. If any test accidentally mutates this data, subsequent tests receive polluted state, causing intermittent failures that are difficult to debug. Fixtures create fresh instances per test, guaranteeing isolation.

| Pros | Cons |
|------|------|
| True test isolation | Slightly more verbose test setup |
| No hidden state pollution | May increase test runtime marginally |
| Easier to debug failures | Requires restructuring test classes |
| Tests can run in any order | - |

**Files converted:** `test_pandas_dataframe.py`, `test_polars_dataframe.py`, `test_pyarrow_table.py`, `test_duckdb_framework.py`

---

## 3. Create Abstract Base Test Classes for Compute Frameworks

- [x] Create `BaseComputeFrameworkTest` abstract class
- [x] Define abstract methods for framework-specific setup
- [x] Move merge tests (inner, left, right, outer, append, union) to base class
- [x] Have Pandas, Polars, PyArrow, DuckDB tests inherit from base
- [x] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Created `DataFrameTestBase` in `tests/test_plugins/compute_framework/test_tooling/dataframe_test_base.py`:
- Abstract base class with `framework_class()`, `create_dataframe()`, `get_connection()` abstract methods
- `setup_method()` creates fresh test data per test (addresses isolation concerns from #2)
- Common merge tests: `test_merge_inner`, `test_merge_left`, `test_merge_right`, `test_merge_full_outer`, `test_merge_append`, `test_merge_union`
- Helper methods: `_create_test_framework()`, `_get_merge_engine()`, `_assert_row_count()`, `_assert_result_equals()`

Framework implementations:
- `TestPandasDataFrameMerge(DataFrameTestBase)` - pandas/test_pandas_dataframe.py
- `TestPolarsDataFrameMerge(DataFrameTestBase)` - polars/test_polars_dataframe.py
- `TestPyArrowTableMerge(DataFrameTestBase)` - pyarrow/test_pyarrow_table.py (skips append/union due to PyArrow limitations)
- `TestDuckDBFrameworkMerge(DataFrameTestBase)` - duckdb/test_duckdb_framework.py (custom connection handling)

Tests for the base class itself: `test_tooling/test_dataframe_test_base.py`

**Why:** The merge operation tests (`test_merge_inner`, `test_merge_left`, etc.) are nearly identical across pandas, polars, pyarrow, and duckdb test files—over 400 lines of duplicated code. An abstract base class with concrete framework subclasses would reduce this to ~100 lines, making maintenance easier and ensuring consistent test coverage across all frameworks.

| Pros | Cons |
|------|------|
| Eliminates ~300 lines of duplication | Adds inheritance complexity |
| Single place to add new merge test scenarios | Abstract patterns can be harder to debug |
| Ensures consistent coverage across frameworks | Requires understanding of test class hierarchy |
| Pattern already exists in `multi_index_test_base.py` | - |

**Files consolidated:** `test_pandas_dataframe.py`, `test_polars_dataframe.py`, `test_pyarrow_table.py`, `test_duckdb_framework.py`

---

## 4. Parametrize Import Availability Tests

- [x] Create shared helper for import mocking pattern
- [x] Update framework availability tests to use shared helper
- [x] Remove duplicate mocking logic from test files
- [x] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Created `tests/test_plugins/compute_framework/test_tooling/availability_test_helper.py` with `assert_unavailable_when_import_blocked()` helper function. Each test file now calls this helper instead of duplicating the mocking logic.

**Approach taken**: Instead of moving all tests to a central module (which would hurt discoverability and cause import issues), we created a shared helper that each test file calls. Tests remain in their respective files but delegate to the helper for the mocking logic.

**Files updated:**
- `polars/test_polars_dataframe.py` - uses helper with `["polars"]`
- `polars/test_polars_lazy_dataframe.py` - uses helper with `["polars"]`
- `duckdb/test_duckdb_framework.py` - uses helper with `["duckdb"]`
- `pyarrow/test_pyarrow_table.py` - uses helper with `["pyarrow"]`
- `spark/test_spark_framework.py` - uses helper with `["pyspark.sql"]`
- `iceberg/test_iceberg_framework.py` - uses helper with `["pyiceberg"]` (standardized from different pattern)

**Why:** Each compute framework has nearly identical tests for checking availability when dependencies aren't installed. The pattern `@patch("builtins.__import__")` with `side_effect` raising `ImportError` is repeated 6+ times. A single parametrized test would cover all frameworks in one place, making it trivial to add new framework availability tests.

| Pros | Cons |
|------|------|
| Tests remain discoverable in their files | - |
| Easy to add new frameworks | - |
| Reduces boilerplate significantly (~10 lines → 1 line per test) | - |
| Centralizes import mock logic | - |
| No import issues (each file imports only its own framework) | - |

---

## 5. Rename Numbered Test Methods to Descriptive Names

- [x] Audit tests with `_1`, `_2`, `_3` suffixes
- [x] Determine what each numbered variant tests differently
- [x] Rename to describe the specific scenario being tested
- [x] Update any documentation referencing old names
- [x] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Renamed 5 numbered test methods in `tests/test_core/test_integration/test_core/test_runner_one_compute_framework.py`:

| Old Name | New Descriptive Name |
|----------|---------------------|
| `test_runner_dependent_feature_config_given_1` | `test_runner_single_feature_with_config_modifies_output` |
| `test_runner_dependent_feature_config_given_2` | `test_runner_multiple_features_with_same_config` |
| `test_runner_dependent_feature_config_given_3` | `test_runner_same_feature_with_different_configs` |
| `test_runner_dependent_feature_config_given_4` | `test_runner_same_feature_different_configs_custom_naming` |
| `test_runner_dependent_feature_config_given_5` | `test_runner_dependency_chain_with_config_propagation` |

The new names clearly describe what each test validates:
- Test 1: Single feature with config affects output name and values
- Test 2: Multiple features requested with same config applied to both
- Test 3: Same feature requested twice with different configs produces distinct results
- Test 4: Same feature with custom naming logic, different configs
- Test 5: Dependency chain where config propagates through Test1 → Test2 → Test3

**Why:** Test methods like `test_runner_dependent_feature_config_given_1` through `_5` don't communicate what differentiates them. When a test fails, developers must read the implementation to understand what's being tested. Descriptive names like `test_runner_with_config_returns_modified_values` and `test_runner_multiple_features_with_config` make failures immediately actionable.

| Pros | Cons |
|------|------|
| Self-documenting test names | Longer method names |
| Failures are immediately understandable | Requires analysis to determine proper names |
| Easier code review | - |
| Better test coverage visibility | - |

**Example file:** `tests/test_core/test_integration/test_core/test_runner_one_compute_framework.py:182-247`

---

## 6. Extract Test Feature Groups to Shared Fixtures Module

- [x] Create shared module for compute framework classes
- [x] Move `SecondCfw`, `ThirdCfw`, `FourthCfw` to shared module
- [x] Import classes from shared location in test files
- [x] Remove duplicate class definitions
- [x] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Created `tests/test_plugins/compute_framework/test_tooling/shared_compute_frameworks.py` with shared custom compute framework classes.

**Analysis performed:**
- Explored 93 feature group classes across the test codebase
- Identified that most feature groups are context-specific (used locally for specific test scenarios)
- Found exact duplicates: `SecondCfw`, `ThirdCfw`, `FourthCfw` classes in two files

**Classes extracted:**

| Class | Description | Previously in |
|-------|-------------|---------------|
| `SecondCfw` | Custom CFW for multi-framework testing | 2 files (duplicate) |
| `ThirdCfw` | Custom CFW for multi-framework testing | 2 files (duplicate) |
| `FourthCfw` | Custom CFW for join testing | 1 file |

**Files updated:**
- `test_runner_multiple_compute_framework.py` - imports `SecondCfw`, `ThirdCfw` from shared module
- `test_runner_join_multiple_compute_framework.py` - imports `SecondCfw`, `ThirdCfw`, `FourthCfw` from shared module

**Why not extract more?** After thorough analysis:
- `EngineRunnerTest*` classes are context-specific with interdependent test scenarios
- `NonRootJoinTestFeature*` classes differ between single-CFW and multi-CFW tests (not exact duplicates)
- `features_for_testing.py` already serves as a shared module for its domain

**Why:** Test feature groups like `EngineRunnerTest`, `NonRootJoinTestFeature`, and similar classes are defined inline in multiple test files. This creates duplication and risks inconsistency if the same logical feature group is defined differently in different files. Centralizing these in a shared fixtures module ensures consistency and reduces the test file size.

| Pros | Cons |
|------|------|
| Single source of truth for test feature groups | Adds import dependencies |
| Smaller, focused test files | Shared fixtures need careful design |
| Reusable across test modules | Initial refactoring effort |
| Easier to maintain test infrastructure | - |

**New shared module:** `tests/test_plugins/compute_framework/test_tooling/shared_compute_frameworks.py`

---

## 7. Standardize Assertion Style to Pytest Native

- [x] Replace `pandas.testing.assert_series_equal` with pytest patterns where possible
- [x] Use `pytest.approx()` for floating-point comparisons
- [ ] Create custom assertion helpers for complex comparisons (not needed currently)
- [ ] Document assertion patterns in test README (not needed currently)
- [x] Run `tox` to verify all tests pass

**Status: MOSTLY ADDRESSED (Low Priority Remaining)**

After investigation, this improvement is largely outdated:

1. **`assert_series_equal` in pandas tests**: Was imported but **never used**. Removed the unused import during improvement #2.

2. **`pdt.assert_frame_equal` in `test_base_merge_engine.py`**: This is the only remaining pandas.testing usage (1 occurrence). It's appropriate here since the test specifically validates pandas DataFrame equality with proper NaN handling.

3. **`self.assertEqual` style**: Already addressed in improvement #1 (unittest migration).

**Remaining work** (very low priority):
- The single `pdt.assert_frame_equal` usage is intentional and appropriate for pandas-specific testing
- No widespread assertion style inconsistency exists in the current codebase

**Why:** The codebase uses multiple assertion styles: plain `assert`, `self.assertEqual`, `pandas.testing.assert_series_equal`, and framework-specific matchers. While some framework assertions are necessary (e.g., for DataFrame equality with NaN handling), many uses could be standardized. Consistent assertions improve readability and make it easier to understand what's being verified.

| Pros | Cons |
|------|------|
| Consistent assertion patterns | Framework-specific assertions sometimes necessary |
| Better pytest integration and output | Migration effort |
| Simpler test code | Some precision may require framework assertions |
| Reduced cognitive load | - |

**Note:** `test_pandas_dataframe.py` now uses standard pytest `assert` statements and DataFrame `.equals()` methods.

---

## 8. Remove sys.path Manipulation from Tests

- [x] Identify all `sys.path.insert` calls in test files
- [x] Fix underlying import issues via proper package structure
- [x] Ensure `conftest.py` and pytest configuration handle paths
- [x] Remove sys.path manipulation code
- [x] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Removed all `sys.path.insert` manipulation from 5 Spark test files by fixing the underlying package structure issue.

**Root cause:** The `tests/test_plugins/compute_framework/base_implementations/` directory and its subdirectories were missing `__init__.py` files, which prevented Python's import system from treating them as packages. This caused the full package import path to fail, requiring a fallback to sys.path manipulation.

**Solution:**
1. Added `__init__.py` files to:
   - `tests/test_plugins/compute_framework/base_implementations/`
   - `tests/test_plugins/compute_framework/base_implementations/spark/`
   - `tests/test_plugins/compute_framework/base_implementations/pandas/`
   - `tests/test_plugins/compute_framework/base_implementations/polars/`
   - `tests/test_plugins/compute_framework/base_implementations/pyarrow/`
   - `tests/test_plugins/compute_framework/base_implementations/duckdb/`
   - `tests/test_plugins/compute_framework/base_implementations/iceberg/`
   - `tests/test_plugins/compute_framework/base_implementations/python_dict/`

2. Replaced try/except import blocks with direct imports in all 5 Spark test files:
   - `test_spark_framework.py`
   - `test_spark_integration.py`
   - `test_spark_pyarrow_transformer.py`
   - `test_spark_filter_engine.py`
   - `test_spark_merge_engine.py`

**Why:** Tests like `test_spark_integration.py` contained `sys.path.insert(0, ...)` blocks, which is a code smell indicating import issues. Pytest should handle test discovery and imports automatically when the project is properly structured. Path manipulation can cause inconsistent behavior between local runs and CI, and makes tests dependent on execution context.

| Pros | Cons |
|------|------|
| Tests work consistently everywhere | - |
| No hidden import side effects | - |
| Cleaner test code | - |
| Better IDE support | - |
| Full package imports work reliably | - |

**Files modified:** 5 Spark test files in `tests/test_plugins/compute_framework/base_implementations/spark/`
**Files added:** 8 `__init__.py` files in `base_implementations/` and subdirectories

---

## 9. Consolidate conftest.py Fixtures

- [ ] Audit all conftest.py files for duplicated fixtures
- [ ] Move shared fixtures to appropriate conftest scope
- [ ] Create fixture documentation in docstrings
- [ ] Remove orphaned or unused fixtures
- [ ] Run `tox` to verify all tests pass

**Why:** There are conftest.py files at multiple levels with some fixture duplication. The root conftest has global fixtures like `flight_server`, while subdirectory conftest files have local fixtures that could potentially be shared. Consolidating fixtures reduces duplication and makes it clearer what fixtures are available at each test level.

| Pros | Cons |
|------|------|
| Clearer fixture hierarchy | Requires careful scoping decisions |
| No duplicate fixture definitions | May affect test isolation if scoping wrong |
| Better fixture documentation | - |
| Easier to find available fixtures | - |

**Conftest files:** `tests/conftest.py`, `tests/test_plugins/compute_framework/base_implementations/spark/conftest.py`, `tests/test_plugins/feature_group/experimental/test_time_window_feature_group/conftest.py`

---

## 10. Add Type Hints to All Test Functions and Fixtures

- [ ] Audit test functions missing return type hints
- [ ] Add `-> None` to all test functions
- [ ] Add type hints to fixture parameters
- [ ] Configure mypy for test directory
- [ ] Run `tox` and `mypy tests/` to verify

**Why:** While most test functions have `-> None` type hints, some are missing them. Complete type hints enable mypy to catch errors in test code, ensure fixtures return expected types, and improve IDE support. Type-checked tests are more reliable and serve as documentation for fixture contracts.

| Pros | Cons |
|------|------|
| Catch type errors in tests | Additional annotation overhead |
| Better IDE autocomplete | Some dynamic test patterns hard to type |
| Fixture contracts are explicit | mypy configuration needed |
| Consistent with production code standards | - |

**Example:** Some fixtures use `-> Any` where specific types like `-> pd.DataFrame` would be better

---

## Priority Order

1. **Housekeeping:** #9 (Consolidate conftest), #10 (Type hints)

**Completed:**
- ✅ #1 (Migrate unittest) - 13 classes migrated
- ✅ #2 (Class-level state to fixtures) - 4 test classes converted to pytest fixtures
- ✅ #3 (Abstract base tests) - `DataFrameTestBase` implemented
- ✅ #4 (Parametrize availability) - `availability_test_helper.py` with shared helper
- ✅ #5 (Rename numbered tests) - 5 methods renamed to descriptive names
- ✅ #6 (Extract feature groups) - `shared_compute_frameworks.py` with `SecondCfw`, `ThirdCfw`, `FourthCfw`
- ✅ #7 (Standardize assertions) - Mostly addressed; unused imports removed, only 1 appropriate pandas.testing usage remains
- ✅ #8 (Remove sys.path) - Added missing `__init__.py` files, removed sys.path manipulation from 5 Spark test files
