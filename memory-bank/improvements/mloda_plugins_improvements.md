# mloda_plugins Code Improvements

This document identifies 10 breaking-change improvements for the mloda_plugins codebase. Each improvement includes a rationale, pros/cons analysis, and testing checklist.

---

## 1. Unify Filter Parameter Interface

### Status: COMPLETED

- [x] Design FilterParameter Protocol/Interface (see [filter_parameter_design.md](filter_parameter_design.md))
- [x] Implement FilterParameter Protocol and FilterParameterImpl class
- [x] Integrate with SingleFilter (`handle_parameter()` returns `FilterParameterImpl`)
- [x] Refactor PandasFilterEngine
- [x] Refactor PolarsFilterEngine
- [x] Refactor SparkFilterEngine
- [x] Refactor DuckDBFilterEngine
- [x] Refactor PyArrowFilterEngine
- [x] Refactor PythonDictFilterEngine
- [x] Refactor IcebergFilterEngine
- [x] Update BaseFilterEngine (`get_min_max_operator()`)
- [x] Add 50 new tests (29 unit + 21 integration)
- [x] Update documentation (`docs/docs/in_depth/filter_data.md`)
- [x] Run tox validation - **1136 tests pass, mypy strict clean**

### Implementation Summary

**New file:** `mloda_core/filter/filter_parameter.py`
- `FilterParameter` Protocol with 5 properties: `value`, `values`, `min_value`, `max_value`, `max_exclusive`
- `FilterParameterImpl` frozen dataclass with `from_dict()` factory method

**Usage:**
```python
# Before (inconsistent)
for param in filter_feature.parameter:
    if param[0] == "value":
        value = param[1]

# After (unified)
value = filter_feature.parameter.value
```

### Original Rationale
Previously, filter engines had divergent parameter extraction logic. PandasFilterEngine directly accessed `filter_feature.parameter` as a simple value, while PolarsFilterEngine and SparkFilterEngine extracted parameters from tuple-based structures using nested loops. This inconsistency meant 7+ files contained duplicate extraction logic with subtle differences.

**Benefits Achieved:**
- Single source of truth for parameter extraction
- Type-safe parameter access with IDE autocompletion
- All 7 filter engines now use consistent property access
- ~50 lines of duplicate extraction code eliminated

---

## 2. Replace NotImplementedError with Abstract Base Classes

### Status
- [x] Convert AggregatedFeatureGroup to ABC
- [x] Convert ClusteringFeatureGroup base to ABC
- [x] Convert ForecastingFeatureGroup base to ABC
- [x] Convert TimeWindowFeatureGroup base to ABC
- [x] Convert DimensionalityReductionFeatureGroup to ABC
- [x] Convert DataQuality base to ABC
- [not done] Update ReadFile/ReadDB base classes

### Rationale
18 files use `raise NotImplementedError` without proper abstract contracts. Subclasses must implement 3-7 methods but there's no clear indication of which are required vs optional, no enforcement at class definition time, and no IDE support for implementation. By converting to proper `@abstractmethod` decorators from ABC, we get compile-time enforcement, clear contracts, and IDE autocompletion showing which methods must be implemented.

**Pros:**
- Fail-fast at class definition, not runtime
- Clear documentation of required methods
- IDE support for implementing abstract classes
- Prevents accidental partial implementations

**Cons:**
- All concrete implementations must be updated
- Requires careful distinction between required and optional hooks
- Breaking change for external plugins extending base classes

---

## 3. Eliminate Dual Interface Pattern (String-based vs Configuration-based)

### Status
- [ ] Design unified FeatureSpecification interface
- [ ] Create FeatureGroupTemplate base class
- [ ] Migrate AggregatedFeatureGroup
- [ ] Migrate ClusteringFeatureGroup
- [ ] Migrate ForecastingFeatureGroup
- [ ] Migrate TimeWindowFeatureGroup
- [ ] Migrate GeoDistanceFeatureGroup
- [ ] Migrate TextCleaningFeatureGroup
- [ ] Migrate DataQualityFeatureGroup
- [ ] Deprecate old interface methods
- [ ] Update documentation with migration guide
- [ ] Run tox validation

### Rationale
Every experimental feature group implements two parallel feature definition methods: string-based (e.g., `"customer_behavior__cluster_kmeans_5"`) and configuration-based (via Options dict). This results in 20+ lines of duplicate fallback logic in each feature group's `input_features` and `match_feature_group_criteria` methods. The dual interface causes inconsistent validation, different error messages between modes, and makes adding new validation rules require changes in two places per feature group.

**Pros:**
- Eliminates 200+ lines of duplicate code across feature groups
- Single validation path reduces bugs
- Consistent error messages
- Easier to add new feature groups

**Cons:**
- Major breaking change affecting all feature definitions
- Users must migrate existing feature definitions
- String-based API is convenient for quick prototyping

---

## 4. Standardize Class Naming Conventions

### Status: COMPLETED

- [x] Define naming convention standard (document)
- [x] Rename PandasPyarrowTransformer -> PandasPyArrowTransformer
- [x] Standardize all *PyarrowTransformer classes (7 classes renamed)
- [x] Standardize DuckDB vs Duckdb prefix usage (already consistent, no changes needed)
- [x] Standardize *Dataframe vs *DataFrame suffix (3 classes renamed)
- [x] Rename PyarrowTable -> PyArrowTable
- [x] Update all imports across codebase (~159 files)
- [x] Update documentation
- [x] Run tox validation - **1171 tests pass, mypy strict clean**

### Implementation Summary

**Classes Renamed:**

| Old Name | New Name |
|----------|----------|
| `PandasPyarrowTransformer` | `PandasPyArrowTransformer` |
| `PolarsPyarrowTransformer` | `PolarsPyArrowTransformer` |
| `PolarsLazyPyarrowTransformer` | `PolarsLazyPyArrowTransformer` |
| `SparkPyarrowTransformer` | `SparkPyArrowTransformer` |
| `DuckDBPyarrowTransformer` | `DuckDBPyArrowTransformer` |
| `PythonDictPyarrowTransformer` | `PythonDictPyArrowTransformer` |
| `IcebergPyarrowTransformer` | `IcebergPyArrowTransformer` |
| `PyarrowTable` | `PyArrowTable` |
| `PandasDataframe` | `PandasDataFrame` |
| `PolarsDataframe` | `PolarsDataFrame` |
| `PolarsLazyDataframe` | `PolarsLazyDataFrame` |

**Note:** DuckDB naming was already consistent throughout - no changes needed.

**Note:** Deprecation aliases were NOT created per user request (direct rename without backward compatibility).

### Rationale
The codebase had inconsistent naming patterns: `PandasPyarrowTransformer` vs `PolarsPyarrowTransformer` (case differences), and `Dataframe` vs `DataFrame` suffixes. Standardizing on consistent PascalCase with proper acronym handling (PyArrow, DataFrame) improves discoverability and reduces user friction.

**Benefits Achieved:**
- Consistent, predictable API
- Easier to discover and import classes
- Better IDE autocompletion
- Professional appearance

---

## 5. Replace Options Dict with Typed Configuration Classes

### Status
- [ ] Design FeatureOptions dataclass
- [ ] Design ValidationConfig dataclass
- [ ] Design SourceFeatureSpec dataclass
- [ ] Implement type converters for backward compatibility
- [ ] Refactor DefaultOptionKeys usage
- [ ] Update Options class to use typed configs
- [ ] Migrate all feature groups to typed configs
- [ ] Update all tests with new types
- [ ] Run tox validation

### Rationale
The current Options system uses `Dict[str, Any]` for everything, losing all type safety. DefaultOptionKeys has inconsistencies (e.g., `reference_time` enum maps to `"time_filter"` value). The PROPERTY_MAPPING pattern is repeated 20+ times across feature groups with identical boilerplate. Replacing with typed dataclasses provides compile-time validation, IDE autocompletion, and eliminates the repeated metadata structures.

**Pros:**
- Full type safety with IDE support
- Eliminates PROPERTY_MAPPING boilerplate
- Compile-time validation of option names
- Self-documenting configuration

**Cons:**
- Major refactor touching 30+ files
- External code using raw dicts will break
- Learning curve for new configuration pattern

---

## 6. Standardize Data Access Patterns

### Status
- [ ] Design DataAccessProtocol interface
- [ ] Create FileDataAccess implementation
- [ ] Create DatabaseDataAccess implementation
- [ ] Create CredentialDataAccess implementation
- [ ] Refactor ReadFile to use protocol
- [ ] Refactor ReadDB to use protocol
- [ ] Refactor SQLite to use protocol
- [ ] Standardize validation return types
- [ ] Update all data access tests
- [ ] Run tox validation

### Rationale
Three different patterns exist for data access: ReadFile checks `isinstance(data_access, (DataAccessCollection, str, Path))`, ReadDB checks `isinstance(data_access, (DataAccessCollection, HashableDict))`, and SQLite accesses `credentials.data.get()` directly. This inconsistency makes it hard to understand what data_access types are supported, leads to different validation approaches, and produces inconsistent error messages. A unified DataAccessProtocol would standardize behavior.

**Pros:**
- Consistent validation across all data sources
- Clear contract for data access objects
- Easier to add new data source types
- Predictable error handling

**Cons:**
- Breaking change to data access signatures
- Existing code using raw strings/paths needs updating
- More verbose for simple use cases

---

## 7. Fix Static vs Class Method Inconsistencies

### Status: COMPLETED

- [x] Audit all @staticmethod usages
- [x] Convert state-dependent statics to @classmethod
- [x] Remove global state from static methods
- [x] Standardize pd_dataframe/pd_series patterns
- [x] Update merge_engine/filter_engine patterns
- [x] Document when to use each decorator
- [x] Update tests for method signatures
- [x] Run tox validation - **1338 tests pass, mypy strict clean**

### Implementation Summary

Converted methods that access module-level globals or return class types from `@staticmethod` to `@classmethod`:

**Phase 1 - Framework Type Accessor Methods:**
- `PandasDataFrame.pd_dataframe()`, `pd_series()`, `pd_merge()`
- `PolarsDataFrame.pl_dataframe()`, `pl_series()`
- `PolarsLazyDataFrame.pl_lazy_frame()`, `pl_dataframe()`, `pl_series()`
- `DuckDBFramework.duckdb_relation()`
- `SparkFramework.spark_dataframe()`, `spark_session()`
- `PandasMergeEngine.pd_merge()`, `pd_concat()`

**Phase 2 - Engine Factory Methods:**
- `ComputeFrameWork.merge_engine()`, `filter_engine()`, `expected_data_framework()`
- All 8 subclass implementations updated

**New Tests:** `/workspace/tests/test_plugins/compute_framework/test_method_decorators.py`
- 35 tests verifying methods are bound classmethods using `inspect.ismethod()`
- Tests verify class-level and instance-level calling patterns

### Rationale
Methods like `PandasDataFrame.pd_dataframe()` are marked `@staticmethod` but access global `pd` variable, violating static method semantics. Instance methods like `merge_engine()` don't use `self` but aren't static. This inconsistency makes testing and mocking difficult, breaks IDE analysis, and confuses developers about method contracts. Proper classification enables better testing and clearer intent.

**Pros:**
- Proper semantic usage of decorators
- Easier testing and mocking
- Clear intent for each method
- Better IDE analysis and refactoring support

**Cons:**
- Changes method signatures (cls parameter added)
- May require dependency injection refactoring
- Affects all subclasses

---

## 8. Implement Proper Dependency Management

### Status
- [ ] Create extras in pyproject.toml for optional deps
- [ ] Create mloda[pandas] extra
- [ ] Create mloda[polars] extra
- [ ] Create mloda[duckdb] extra
- [ ] Create mloda[spark] extra
- [ ] Replace try/except ImportError with proper checks
- [ ] Add @requires_dependency decorator
- [ ] Update installation documentation
- [ ] Add CI tests for each extra
- [ ] Run tox validation

### Rationale
Files like `duckdb_framework.py` use `try: import duckdb except ImportError: duckdb = None` pattern, failing silently at import time and erroring only at runtime. There's no clear declaration of which features require which optional dependencies. Using setuptools extras (e.g., `pip install mloda[duckdb]`) with a `@requires_dependency` decorator provides clear dependency documentation, fail-fast behavior, and proper error messages.

**Pros:**
- Clear dependency documentation
- Fail-fast with helpful error messages
- Smaller base installation
- Proper optional dependency handling

**Cons:**
- Users must install correct extras
- More complex installation instructions
- CI needs to test each combination

---

## 9. Standardize Return Type Conventions

### Status: COMPLETED

- [x] Define return type standards document (see [return_type_standards.md](return_type_standards.md))
- [x] Replace `return False` with `return None` for "not found" scenarios
- [x] Standardize match_* method return types
- [x] Standardize validation method return types (documented Optional[bool] semantics)
- [x] Add type hints to all public methods
- [x] Update type checking in CI (mypy strict)
- [x] Update all affected tests
- [x] Run tox validation - **1303 tests pass, mypy strict clean**

### Implementation Summary

**Standards document:** `memory-bank/improvements/return_type_standards.md`

**Key changes:**
1. Match methods now return `None` instead of `False` for "not found"
2. Callers updated to check `is None` instead of `is False` or `== False`
3. `Optional[bool]` methods documented with clear semantics:
   - `None`: No validation needed/neutral
   - `True`: Validation passed
   - `False`: Validation failed

**Files modified:**
- `mloda_plugins/feature_group/input_data/read_file.py`
- `mloda_plugins/feature_group/input_data/read_db.py`
- `mloda_core/abstract_plugins/components/input_data/base_input_data.py`
- `mloda_core/abstract_plugins/components/match_data/match_data.py`
- `mloda_core/abstract_plugins/abstract_feature_group.py` (docstrings)
- `mloda_core/abstract_plugins/components/base_validator.py` (docstrings)
- 3 test files updated for consistency

### Original Rationale
Methods inconsistently return `False`, `None`, or raise exceptions for "not found" scenarios. `match_read_file_data_access` returns `False`, `input_features` raises `ValueError`, and `validate_input_features` returns `Optional[bool]`. This forces callers to check for boolean False AND None AND catch exceptions. Standardizing on `Optional[T]` for optional results and exceptions for errors provides consistent, type-safe behavior.

**Benefits Achieved:**
- Consistent, predictable API behavior following Python best practices (Django/SQLAlchemy patterns)
- Type-safe with Optional[T]
- Clear distinction between "not found" (return None) and "error" (raise exception)
- Easier to write correct calling code with `is None` checks

---

## 10. Extract Common Feature Group Utilities

### Status
- [ ] Create FeatureChainParserMixin
- [ ] Create SourceFeatureExtractorMixin
- [ ] Create ValidationMixin
- [ ] Create PropertyMappingMixin
- [ ] Refactor AggregatedFeatureGroup to use mixins
- [ ] Refactor ClusteringFeatureGroup to use mixins
- [ ] Refactor ForecastingFeatureGroup to use mixins
- [ ] Refactor remaining experimental feature groups
- [ ] Update tests for mixin behavior
- [ ] Run tox validation

### Rationale
Every experimental feature group reimplements identical logic: feature chain parsing (`FeatureChainParser.parse_feature_name`), source feature extraction from options, property mapping validation, and match criteria logic. This results in 50+ occurrences of near-identical code across the codebase. Extracting these into composable mixins eliminates duplication, centralizes bug fixes, and makes adding new feature groups straightforward.

**Pros:**
- Eliminates 500+ lines of duplicate code
- Single place to fix bugs
- Easier to add new feature groups
- Consistent behavior across all groups

**Cons:**
- Mixin inheritance can be confusing
- Breaking change to class hierarchy
- Requires careful design to avoid diamond inheritance issues

---

## 11. Clarify Time Concepts Design

### Status: COMPLETED

- [x] See [time_concepts_design.md](time_concepts_design.md) for full analysis
- [x] Implement `DefaultColumnNames` class
- [x] Update method/parameter naming
- [x] Standardize test patterns
- [x] Update documentation
- [x] Run tox validation - **1352 tests pass, mypy strict clean**

### Implementation Summary

**Updated file:** `mloda_plugins/feature_group/experimental/default_options_key.py`
- Added `time_travel = "time_travel_filter"` enum member
- Enum values serve as both option keys AND default column names
- Note: Initially created separate `DefaultColumnNames` class, then merged back into `DefaultOptionKeys` (see `default_column_names_merge.md`)

**Method renames:**
- `get_time_filter_feature()` → `get_reference_time_column()` (TimeWindowFeatureGroup, ForecastingFeatureGroup)
- `_check_time_filter_feature_exists` → `_check_reference_time_column_exists`
- `_check_time_filter_feature_is_datetime` → `_check_reference_time_column_is_datetime`

**Parameter renames (GlobalFilter.add_time_and_time_travel_filters):**
- `time_filter_feature` → `event_time_column`
- `time_travel_filter_feature` → `validity_time_column`

**Benefits:**
- Clear separation of concepts (column names vs option keys vs filters)
- Self-documenting API with accurate method/parameter names
- Consistent mental model for users

### Rationale
The codebase conflates three distinct time concepts, causing confusion:

1. **`reference_time`** - DATA COLUMN: The column containing event timestamps
2. **`time_filter`** - FILTER CONCEPT: Range filter on event time (event_from, event_to)
3. **`time_travel_filter`** - FILTER CONCEPT: Range filter on validity period (valid_from, valid_to)

The `DefaultOptionKeys.reference_time` enum serves triple duty (Options key, default column name, feature name), making the API confusing. The fix from `"time_filter"` to `"reference_time"` was semantically correct, but deeper refactoring would improve clarity.

**Proposed:** Separate `DefaultColumnNames` class for default column values, rename methods like `get_time_filter_feature()` to `get_reference_time_column()`.

**Pros:**
- Clearer API for users
- Self-documenting code
- Consistent mental model

**Cons:**
- Parameter/method rename is breaking change
- Requires updating documentation and examples

---

## Implementation Priority

| Priority | Improvement | Impact | Effort | Status |
|----------|-------------|--------|--------|--------|
| 1 | Unify Filter Parameter Interface | High | Medium | **DONE** |
| 2 | Replace NotImplementedError with ABC | Medium | Low | Pending |
| 3 | Eliminate Dual Interface Pattern | High | High | Pending |
| 4 | Standardize Class Naming | Medium | Low | **DONE** |
| 5 | Replace Options Dict with Typed Classes | High | High | Pending |
| 6 | Standardize Data Access Patterns | Medium | Medium | Pending |
| 7 | Fix Static vs Class Method | Low | Low | **DONE** |
| 8 | Proper Dependency Management | Medium | Medium | Pending |
| 9 | Standardize Return Types | Medium | Medium | **DONE** |
| 10 | Extract Common Utilities | High | Medium | Pending |
| 11 | Clarify Time Concepts Design | Medium | Medium | **DONE** |

---

## Testing Strategy

For each improvement:
1. Write failing tests for new interface/behavior
2. Implement changes
3. Run full test suite with `tox`
4. Verify no regressions
5. Update integration tests
6. Document migration path for breaking changes
