# Improvement 9: Eliminate type: ignore Suppressions

This document provides a detailed ticklist for eliminating `type: ignore` suppressions in the mloda codebase.

**Total Occurrences Found:** 207+ (across mloda_core, mloda_plugins, and tests)

---

## Summary by Category

| Category | Count | Difficulty | Priority |
|----------|-------|------------|----------|
| Optional Library Imports | ~65 | Medium | High |
| NumPy/Array Operations | ~45 | Hard | Medium |
| Multiprocessing Types | ~15 | Hard | High |
| Dynamic Method Creation | ~25 | Very Hard | Low |
| External API Returns | ~8 | Medium | Medium |
| Reflection/Inspection | ~6 | Medium | Medium |
| Union Type Narrowing | ~3 | Easy | High |
| Graph/Container Types | ~4 | Easy | High |
| Test/Mock Fixtures | ~90+ | Medium | Low |

---

## Phase 1: Quick Wins (Easy Fixes)

### 1.1 Union Type Narrowing
- [x] `mloda_core/api/prepare/setup_compute_framework.py:19` - Used `cast()` for proper type narrowing
- [x] `mloda_core/abstract_plugins/components/feature_chainer/feature_chain_parser.py:273` - Dead code removed (comparison was already eliminated)

### 1.2 Graph/Container Types
- [x] `mloda_core/prepare/graph/graph.py` - Changed Dict to DefaultDict for `nodes`, added specific `[arg-type]` error code
- [x] `mloda_core/prepare/graph/graph.py` - Changed Dict to DefaultDict for `edges`, added specific `[arg-type]` error code
  - Note: The `# type: ignore[arg-type]` remains because the defaultdict factory passes None to constructors that don't accept None. Making the constructor params Optional would cause cascading type errors throughout the codebase.

### 1.3 Optional Return Types
- [x] `mloda_core/abstract_plugins/components/options.py:191` - Used `cast("Feature", item)` for proper type annotation
- [x] `mloda_core/abstract_plugins/components/base_artifact.py:55` - Added proper None checks before accessing

---

## Phase 2: Optional Library Imports (~65 occurrences)

### Strategy: Use TYPE_CHECKING pattern with proper stubs

### 2.1 DuckDB Framework
- [ ] `mloda_plugins/compute_framework/base_implementations/duckdb/duckdb_framework.py:13`
- [ ] `mloda_plugins/compute_framework/base_implementations/duckdb/duckdb_merge_engine.py:10`
- [ ] `mloda_plugins/compute_framework/base_implementations/duckdb/duckdb_pyarrow_transformer.py:8`

### 2.2 Polars Framework
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/dataframe.py:12`
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/lazy_dataframe.py:10`
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/polars_filter_engine.py:8`
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/polars_merge_engine.py:10`
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/polars_lazy_merge_engine.py:8`
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/polars_pyarrow_transformer.py:8`
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/polars_lazy_pyarrow_transformer.py:8`

### 2.3 Iceberg Framework
- [ ] `mloda_plugins/compute_framework/base_implementations/iceberg/iceberg_framework.py:13-14` (Catalog, IcebergTable)
- [ ] `mloda_plugins/compute_framework/base_implementations/iceberg/iceberg_pyarrow_transformer.py:8`
- [ ] `mloda_plugins/compute_framework/base_implementations/iceberg/iceberg_filter_engine.py:9-15` (multiple filter types)

### 2.4 OpenTelemetry
- [ ] `mloda_plugins/function_extender/base_implementations/otel/otel_extender.py:12`

### 2.5 IPython/Jupyter
- [ ] `mloda_core/prepare/identify_feature_group.py` - IPython import and get_ipython() call

### 2.6 Aggregated Feature Groups
- [ ] `mloda_plugins/feature_group/experimental/aggregated_feature_group/polars_lazy.py`

---

## Phase 3: Multiprocessing Types (~15 occurrences)

### Strategy: Use `from __future__ import annotations` + `Queue[Any]` type parameters

### 3.1 Runner Class (run.py)
- [x] `mloda_core/runtime/run.py:62` - `process_register` Dict with `Queue[Any]` types
- [x] `mloda_core/runtime/run.py:65` - `result_queues_collection` with `Set[Queue[Any]]` type
- [x] `mloda_core/runtime/run.py` - `_get_execution_function` return type fixed to `Callable[[Any], None]`
- [x] `mloda_core/runtime/run.py` - Manager CfwManager call - kept `# type: ignore[attr-defined]` (dynamic registration)

### 3.2 Multiprocessing Worker
- [x] `mloda_core/runtime/worker/multiprocessing_worker.py` - All Queue parameters now use `Queue[Any]`
- [x] Added `from __future__ import annotations` to both files to enable runtime subscripting
- [x] Only 1 remaining ignore: `# type: ignore[assignment]` for reassigning `from_cfw` parameter

---

## Phase 4: NumPy/Array Operations (~45 occurrences)

### Strategy: Use `TYPE_CHECKING` with `NDArray` type hint

### 4.1 Dimensionality Reduction (Pandas)
- [x] `mloda_plugins/feature_group/experimental/dimensionality_reduction/pandas.py` - All NumPy type annotations fixed
  - Added `TYPE_CHECKING` import with `from numpy.typing import NDArray`
  - Changed all `np.ndarray` return types to `NDArray[Any]`
  - Changed all `np.ndarray` parameters to `NDArray[Any]`
  - Used `cast("NDArray[Any]", ...)` for sklearn return values
  - Note: 2 remaining `type: ignore` for optional import fallbacks (`pd = None`, `np = None`) - standard pattern

### 4.2 Clustering (Pandas)
- [x] `mloda_plugins/feature_group/experimental/clustering/pandas.py` - All NumPy array parameters and returns fixed
  - Same pattern as dimensionality reduction
  - Fixed ~15 method signatures with proper NDArray type hints
  - Note: 2 remaining `type: ignore` for optional import fallbacks - standard pattern

---

## Phase 5: External API Returns (~8 occurrences)

### Strategy: Use cast() or create typed wrapper functions

### 5.1 Spark Framework
- [ ] `mloda_plugins/compute_framework/base_implementations/spark/spark_framework.py:122`
- [ ] `mloda_plugins/compute_framework/base_implementations/spark/spark_framework.py:129`
- [ ] `mloda_plugins/compute_framework/base_implementations/spark/spark_framework.py:149`
- [ ] `mloda_plugins/compute_framework/base_implementations/spark/spark_framework.py:151`

### 5.2 Polars Lazy
- [ ] `mloda_plugins/compute_framework/base_implementations/polars/polars_lazy_pyarrow_transformer.py:69` - lazy() return

### 5.3 Python Dict Framework
- [ ] `mloda_plugins/compute_framework/base_implementations/python_dict/python_dict_framework.py:85`

---

## Phase 6: Reflection/Inspection (~6 occurrences)

### Strategy: Use proper typing for classmethod introspection

### 6.1 Feature Group Version
- [x] `mloda_core/abstract_plugins/components/feature_group_version.py` - `class_source_hash` method - Used `Type[Any]` instead of `Type`
- [x] `mloda_core/abstract_plugins/components/feature_group_version.py` - `module_name` method - Used `Type[Any]` instead of `Type`
- [x] `mloda_core/abstract_plugins/components/feature_group_version.py` - `version` method - Used `Type[Any]` instead of `Type`

### 6.2 Input Data Polymorphic Calls
- [x] `mloda_core/abstract_plugins/components/input_data/base_input_data.py` - `match_subclass_data_access` calls - Added `[attr-defined]` error code
- [x] `mloda_core/abstract_plugins/components/input_data/base_input_data.py` - `load_data` abstract call - Added `[arg-type]` error code

---

## Phase 7: Dynamic Method Creation (~25 occurrences)

### Strategy: Add specific error codes to all type: ignore comments

### 7.1 Dynamic Feature Group Factory
- [x] `mloda_plugins/feature_group/experimental/dynamic_feature_group_factory/dynamic_feature_group_factory.py`
  - [x] Added `[no-untyped-def]` to all 12 function definitions (dynamic methods with untyped `self`/`cls`)
  - [x] Added `[no-any-return]` to property dictionary calls returning `Any`
  - [x] Added `[misc, arg-type, no-any-return]` to `super()` calls referencing `new_class` before definition
  - Note: 34 type: ignore comments total, all now with specific error codes:
    - 12 `[no-untyped-def]` on function definitions
    - 11 `[no-any-return]` on property dictionary calls
    - 10 `[misc, arg-type, no-any-return]` on super() calls
    - 1 `[misc, arg-type]` on `calculate_feature` super() call (returns Any, no extra ignore needed)

---

## Phase 8: Apache Arrow / Flight Server

### Strategy: Add type stubs or use cast()

- [x] `mloda_core/runtime/flight/flight_server.py` - FlightServerBase inheritance - Added `[misc]` error code
  - Note: pyarrow doesn't have type stubs, so FlightServerBase has type `Any`. The `# type: ignore[misc]` suppresses "Class cannot subclass 'FlightServerBase' (has type 'Any')"

---

## Phase 9: Feature Group Step

- [x] `mloda_core/core/step/feature_group_step.py` - `set_artifact_to_save` - Added assertion for None check

---

## Phase 10: Test Files (~90+ occurrences)

### Strategy: Lower priority - fix as part of other changes

- [ ] Mock fixtures for optional dependencies (duckdb, polars, pyiceberg)
- [ ] Type assertions on test data
- [ ] Private attribute access in tests
- [ ] Dynamic test fixture creation

---

## Recommended Approach by Fix Type

### Pattern 1: TYPE_CHECKING Import Pattern
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb
else:
    try:
        import duckdb
    except ImportError:
        duckdb = None  # Now properly typed
```

### Pattern 2: TypeVar for Generic Containers
```python
from typing import TypeVar, Generic
from multiprocessing import Queue

T = TypeVar('T')
class TypedQueue(Queue, Generic[T]):
    pass
```

### Pattern 3: TypeGuard for Union Narrowing
```python
from typing import TypeGuard

def is_set_of_frameworks(obj: Union[Set, List]) -> TypeGuard[Set[Type[ComputeFrameWork]]]:
    return isinstance(obj, set)
```

### Pattern 4: cast() for External APIs
```python
from typing import cast
result = cast(DataFrame, spark.createDataFrame([], schema))
```

---

## Progress Tracking

| Phase | Status | Completed | Total |
|-------|--------|-----------|-------|
| Phase 1: Quick Wins | **DONE** | 6 | 6 |
| Phase 2: Optional Imports | In Progress | 1 | ~65 |
| Phase 3: Multiprocessing | **DONE** | 12 | ~15 |
| Phase 4: NumPy/Arrays | **DONE** | ~45 | ~45 |
| Phase 5: External APIs | Not Started | 0 | 8 |
| Phase 6: Reflection | **DONE** | 6 | 6 |
| Phase 7: Dynamic Methods | **DONE** | 34 | 34 |
| Phase 8: Arrow Flight | **DONE** | 1 | 1 |
| Phase 9: Feature Step | **DONE** | 1 | 1 |
| Phase 10: Test Files | Not Started | 0 | ~90 |
| **TOTAL** | | 60 | ~271 |

**Summary of changes in mloda_core:**
- Started with 28 `type: ignore` comments
- Now down to 10 `type: ignore` comments (18 eliminated/fixed)
- All remaining 10 have specific error codes:
  - 2 in `graph.py` - `[arg-type]` for defaultdict None values
  - 2 in `identify_feature_group.py` - `[attr-defined]`, `[no-untyped-call]` for IPython
  - 1 in `multiprocessing_worker.py` - `[assignment]` for parameter reassignment
  - 1 in `flight_server.py` - `[misc]` for Arrow Flight inheritance (pyarrow untyped)
  - 1 in `run.py` - `[attr-defined]` for dynamic manager registration
  - 3 in `base_input_data.py` - `[attr-defined]`, `[arg-type]` for polymorphic calls

---

## Notes

1. **Focus on mloda_core first** - These are the core framework files and have the highest impact
2. **Optional imports are systematic** - Once a pattern is established, apply it consistently
3. **Some type: ignore may remain** - Multiprocessing and metaclass patterns may be genuinely difficult to type
4. **Test files are lower priority** - Fix as encountered during other work
5. **Run mypy after each phase** - Validate fixes don't introduce new issues
