# Duplicate Tests Analysis

This document identifies duplicate and redundant tests in the codebase that are candidates for consolidation.

## Executive Summary

| Category | Files Affected | Duplicate Methods | Priority |
|----------|---------------|-------------------|----------|
| Filter Engine Tests | 6 | 11 per file | High |
| Merge Engine Tests | 6 | 11+ per file | High |
| Feature Group Base Tests | 15+ | 3-4 per file | Medium |
| Common Method Tests | Various | 6+ occurrences | Low |

---

## 1. Filter Engine Tests (High Priority)

Nearly identical tests across 6 compute framework implementations. Each file tests the same filter behavior with identical logic, only differing in data structure.

### Affected Files

```
tests/test_plugins/compute_framework/base_implementations/python_dict/test_python_dict_filter_engine.py
tests/test_plugins/compute_framework/base_implementations/polars/test_polars_filter_engine.py
tests/test_plugins/compute_framework/base_implementations/pyarrow/test_pyarrow_filter_engine.py
tests/test_plugins/compute_framework/base_implementations/duckdb/test_duckdb_filter_engine.py
tests/test_plugins/compute_framework/base_implementations/iceberg/test_iceberg_filter_engine.py
tests/test_plugins/compute_framework/base_implementations/spark/test_spark_filter_engine.py
```

### Duplicate Test Methods (11 per file)

| Test Method | Description |
|-------------|-------------|
| `test_do_equal_filter` | Tests filter equality checks |
| `test_do_range_filter` | Tests range filtering with min/max |
| `test_do_range_filter_exclusive` | Tests range filtering with exclusive bounds |
| `test_do_min_filter` | Tests minimum value filtering |
| `test_do_max_filter` | Tests maximum value filtering |
| `test_do_max_filter_with_tuple` | Tests tuple-based max filtering |
| `test_do_regex_filter` | Tests regex pattern filtering |
| `test_do_categorical_inclusion_filter` | Tests categorical filtering |
| `test_apply_filters` | Tests applying multiple filters |
| `test_final_filters` | Tests final filter status |
| `test_do_range_filter_missing_parameters` | Tests error handling |

### Consolidation Recommendation

Create a parameterized base test class:
```python
# tests/test_plugins/compute_framework/base_implementations/test_filter_engine_base.py
@pytest.mark.parametrize("framework", ["python_dict", "polars", "pyarrow", "duckdb", "iceberg", "spark"])
class TestFilterEngineBase:
    # Shared test logic with framework-specific fixtures
```

---

## 2. Merge Engine Tests (High Priority)

Duplicate merge operation tests across compute frameworks with identical semantics.

### Affected Files

```
tests/test_plugins/compute_framework/base_implementations/pandas/test_pandas_merge_engine.py
tests/test_plugins/compute_framework/base_implementations/polars/test_polars_merge_engine.py
tests/test_plugins/compute_framework/base_implementations/polars/test_polars_lazy_merge_engine.py
tests/test_plugins/compute_framework/base_implementations/pyarrow/test_pyarrow_merge_engine.py
tests/test_plugins/compute_framework/base_implementations/duckdb/test_duckdb_merge_engine.py
tests/test_plugins/compute_framework/base_implementations/spark/test_spark_merge_engine.py
```

### Duplicate Test Methods

| Test Method | Occurrences |
|-------------|-------------|
| `test_merge_inner` | 3+ |
| `test_merge_left` | 3+ |
| `test_merge_append` | 3+ |
| `test_merge_union` | 3+ |
| `test_merge_right` | 2+ |
| `test_merge_full_outer` | 2+ |
| `test_merge_with_different_join_columns` | 2+ |
| `test_merge_with_empty_datasets` | 2+ |
| `test_merge_with_complex_data` | 2+ |
| `test_merge_with_null_values` | 2+ |
| `test_merge_method_integration` | 2+ |

### Consolidation Recommendation

Similar parameterized approach as filter engine tests.

---

## 3. Feature Group Base Tests (Medium Priority)

Inherited test methods from base classes repeated across concrete implementations.

### Pattern: `test_input_features` (9+ occurrences)

Tests parsing of feature names to extract input features. Identical logic across:

```
tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/test_aggregated_feature_group.py:130
tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/test_modernized_aggregated_feature_group.py:146
tests/test_plugins/feature_group/experimental/test_missing_value_feature_group/test_base_missing_value_feature_group.py:89
tests/test_plugins/feature_group/experimental/test_missing_value_feature_group/test_pandas_missing_value_feature_group.py:84
tests/test_plugins/feature_group/experimental/test_missing_value_feature_group/test_python_dict_missing_value_feature_group.py:121
tests/test_plugins/feature_group/experimental/test_missing_value_feature_group/test_pyarrow_missing_value_feature_group.py:80
tests/test_plugins/feature_group/experimental/test_clustering_feature_group/test_base_clustering_feature_group.py:78
tests/test_plugins/feature_group/experimental/test_dimensionality_reduction_feature_group/test_base_dimensionality_reduction_feature_group.py:58
tests/test_plugins/feature_group/experimental/test_forecasting/test_forecasting_feature_group.py:59
tests/test_plugins/feature_group/experimental/test_geo_distance_feature_group/test_geo_distance.py:45
tests/test_plugins/feature_group/experimental/test_node_centrality_feature_group/test_base_node_centrality_feature_group.py:59
tests/test_plugins/feature_group/experimental/test_time_window_feature_group/test_base_time_window_feature_group.py:103
tests/test_plugins/feature_group/experimental/sklearn/test_scaling_feature_group/test_base_scaling_feature_group.py:52
tests/test_plugins/feature_group/experimental/sklearn/test_encoding_feature_group/test_base_encoding_feature_group.py:145
tests/test_plugins/feature_group/experimental/test_text_cleaning/test_text_cleaning_base.py:44
```

### Pattern: `test_compute_framework_rule` (6+ occurrences)

```
tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/test_aggregated_feature_group.py:152
tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/test_polars_lazy_aggregated_feature_group.py:79
tests/test_plugins/feature_group/experimental/test_base_aggregated_feature_group/test_pyarrow_aggregated_feature_group.py:56
tests/test_plugins/feature_group/experimental/test_missing_value_feature_group/test_base_missing_value_feature_group.py
tests/test_plugins/feature_group/experimental/test_missing_value_feature_group/test_pandas_missing_value_feature_group.py
tests/test_plugins/feature_group/experimental/test_missing_value_feature_group/test_pyarrow_missing_value_feature_group.py
```

### Pattern: `test_match_feature_group_criteria` (9+ occurrences)

Tests feature name pattern matching logic with identical base logic.

### Consolidation Recommendation

Implement test inheritance hierarchy using pytest fixtures:
- Move common tests to a reusable base test mixin
- Use fixtures for different feature group instances
- Keep only framework-specific tests in implementation files

---

## 4. Common Method Tests (Low Priority)

### `test_methods_callable_on_class` (6 occurrences)

Tests that class methods are callable on the class itself.

### `test_integration_with_feature_parser` (6 occurrences)

Tests integration with feature configuration parser.

### `test_final_filters` (6 occurrences)

Tests final filter configuration across filter engine implementations.

---

## Consolidation Strategy

### Phase 1: Filter & Merge Engines
1. Create `tests/test_plugins/compute_framework/base_implementations/conftest.py` with shared fixtures
2. Create parameterized base test classes for filter and merge engines
3. Reduce 12 files to 2 parameterized test files + framework-specific fixtures

### Phase 2: Feature Group Tests
1. Create base test mixins for common functionality
2. Move `test_input_features`, `test_compute_framework_rule`, `test_match_feature_group_criteria` to shared base
3. Keep only implementation-specific tests in concrete test files

### Phase 3: Common Method Tests
1. Consolidate into single parameterized test files where appropriate

---

## Estimated Impact

- **Files reduced**: ~30 test files can be consolidated
- **Code reduction**: ~2000+ lines of duplicate test code
- **Maintenance benefit**: Single source of truth for test logic
- **Coverage preserved**: All test scenarios maintained through parameterization
