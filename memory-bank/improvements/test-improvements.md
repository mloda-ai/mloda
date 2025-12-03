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

- [ ] Identify classes with shared class-level state
- [ ] Create pytest fixtures for `left_data`, `right_data`, `expected_data`
- [ ] Remove class attributes, inject via fixture parameters
- [ ] Ensure each test gets fresh data instances
- [ ] Run `tox` to verify all tests pass

**Partial Progress:** The `DataFrameTestBase` class (from improvement #3) uses `setup_method()` to create fresh `left_data` and `right_data` per test method, addressing isolation for merge tests. However, the original `TestPandasDataframeComputeFramework` (and similar) still use class-level attributes.

**Why:** Tests like `TestPandasDataframeComputeFramework` define shared class attributes (`left_data`, `right_data`) that persist across test methods. If any test accidentally mutates this data, subsequent tests receive polluted state, causing intermittent failures that are difficult to debug. Fixtures create fresh instances per test, guaranteeing isolation.

| Pros | Cons |
|------|------|
| True test isolation | Slightly more verbose test setup |
| No hidden state pollution | May increase test runtime marginally |
| Easier to debug failures | Requires restructuring test classes |
| Tests can run in any order | - |

**Example file:** `tests/test_plugins/compute_framework/base_implementations/pandas/test_pandas_dataframe.py:24-29`

---

## 3. Create Abstract Base Test Classes for Compute Frameworks

- [x] Create `BaseComputeFrameworkTest` abstract class
- [x] Define abstract methods for framework-specific setup
- [x] Move merge tests (inner, left, right, outer, append, union) to base class
- [x] Have Pandas, Polars, PyArrow, DuckDB tests inherit from base
- [ ] Run `tox` to verify all tests pass

**Status: IMPLEMENTED**

Created `DataFrameTestBase` in `tests/test_plugins/compute_framework/test_tooling/dataframe_test_base.py`:
- Abstract base class with `framework_class()`, `create_dataframe()`, `get_connection()` abstract methods
- `setup_method()` creates fresh test data per test (addresses isolation concerns from #2)
- Common merge tests: `test_merge_inner`, `test_merge_left`, `test_merge_right`, `test_merge_full_outer`, `test_merge_append`, `test_merge_union`
- Helper methods: `_create_test_framework()`, `_get_merge_engine()`, `_assert_row_count()`, `_assert_result_equals()`

Framework implementations:
- `TestPandasDataframeMerge(DataFrameTestBase)` - pandas/test_pandas_dataframe.py
- `TestPolarsDataframeMerge(DataFrameTestBase)` - polars/test_polars_dataframe.py
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

- [ ] Create shared fixture for import mocking pattern
- [ ] Parametrize framework availability tests by framework name
- [ ] Remove duplicate `test_is_available_when_X_not_installed` methods
- [ ] Move to shared test module `tests/test_plugins/compute_framework/test_availability.py`
- [ ] Run `tox` to verify all tests pass

**Why:** Each compute framework has nearly identical tests for checking availability when dependencies aren't installed. The pattern `@patch("builtins.__import__")` with `side_effect` raising `ImportError` is repeated 6+ times. A single parametrized test would cover all frameworks in one place, making it trivial to add new framework availability tests.

| Pros | Cons |
|------|------|
| Single test covers all frameworks | Parametrized test failures can be harder to isolate |
| Easy to add new frameworks | Requires understanding parametrize patterns |
| Reduces boilerplate significantly | - |
| Centralizes import mock logic | - |

**Current duplication in:** `test_pandas_dataframe.py`, `test_polars_dataframe.py`, `test_duckdb_framework.py`, `test_pyarrow_table.py`

---

## 5. Rename Numbered Test Methods to Descriptive Names

- [ ] Audit tests with `_1`, `_2`, `_3` suffixes
- [ ] Determine what each numbered variant tests differently
- [ ] Rename to describe the specific scenario being tested
- [ ] Update any documentation referencing old names
- [ ] Run `tox` to verify all tests pass

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

- [ ] Create `tests/fixtures/feature_groups.py` module
- [ ] Move `EngineRunnerTest`, `EngineRunnerTest2`, etc. to shared module
- [ ] Import feature groups from shared location in test files
- [ ] Remove duplicate feature group definitions
- [ ] Run `tox` to verify all tests pass

**Why:** Test feature groups like `EngineRunnerTest`, `NonRootJoinTestFeature`, and similar classes are defined inline in multiple test files. This creates duplication and risks inconsistency if the same logical feature group is defined differently in different files. Centralizing these in a shared fixtures module ensures consistency and reduces the test file size.

| Pros | Cons |
|------|------|
| Single source of truth for test feature groups | Adds import dependencies |
| Smaller, focused test files | Shared fixtures need careful design |
| Reusable across test modules | Initial refactoring effort |
| Easier to maintain test infrastructure | - |

**Files with duplicate definitions:** `test_runner_one_compute_framework.py`, `test_non_root_merges_one_cfw.py`, `features_for_testing.py`

---

## 7. Standardize Assertion Style to Pytest Native

- [ ] Replace `pandas.testing.assert_series_equal` with pytest patterns where possible
- [ ] Use `pytest.approx()` for floating-point comparisons
- [ ] Create custom assertion helpers for complex comparisons
- [ ] Document assertion patterns in test README
- [ ] Run `tox` to verify all tests pass

**Why:** The codebase uses multiple assertion styles: plain `assert`, `self.assertEqual`, `pandas.testing.assert_series_equal`, and framework-specific matchers. While some framework assertions are necessary (e.g., for DataFrame equality with NaN handling), many uses could be standardized. Consistent assertions improve readability and make it easier to understand what's being verified.

| Pros | Cons |
|------|------|
| Consistent assertion patterns | Framework-specific assertions sometimes necessary |
| Better pytest integration and output | Migration effort |
| Simpler test code | Some precision may require framework assertions |
| Reduced cognitive load | - |

**Mixed styles in:** `test_pandas_dataframe.py` (uses `assert`, `assert_series_equal`, `equals()` method)

---

## 8. Remove sys.path Manipulation from Tests

- [ ] Identify all `sys.path.insert` calls in test files
- [ ] Fix underlying import issues via proper package structure
- [ ] Ensure `conftest.py` and pytest configuration handle paths
- [ ] Remove sys.path manipulation code
- [ ] Run `tox` to verify all tests pass

**Why:** Tests like `test_spark_integration.py` contain `sys.path.insert(0, ...)` blocks, which is a code smell indicating import issues. Pytest should handle test discovery and imports automatically when the project is properly structured. Path manipulation can cause inconsistent behavior between local runs and CI, and makes tests dependent on execution context.

| Pros | Cons |
|------|------|
| Tests work consistently everywhere | May require pyproject.toml/setup.py changes |
| No hidden import side effects | Need to understand pytest import system |
| Cleaner test code | - |
| Better IDE support | - |

**Files affected:** `tests/test_plugins/compute_framework/base_implementations/spark/test_spark_integration.py`

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

1. **High Impact, Lower Effort:** #2 (Class-level state), #5 (Rename numbered tests)
2. **High Impact, Medium Effort:** #6 (Extract feature groups)
3. **Medium Impact:** #4 (Parametrize availability), #7 (Standardize assertions)
4. **Housekeeping:** #8 (Remove sys.path), #9 (Consolidate conftest), #10 (Type hints)

**Completed:**
- ✅ #1 (Migrate unittest) - 13 classes migrated
- ✅ #3 (Abstract base tests) - `DataFrameTestBase` implemented
