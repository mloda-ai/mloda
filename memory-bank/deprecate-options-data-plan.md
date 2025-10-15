# Plan: Deprecate options.data Property

## Overview
The `options.data` property is a legacy backward-compatibility layer that returns `options.group`. We need to replace all direct `.data` usage with appropriate alternatives (`.group`, `.context`, or `.get()`) across the codebase.

## Migration Strategy
For each file: Make change → Run `tox` → If pass, run `git add .`

## Progress Checklist

### Phase 1: Core Infrastructure Files (7 files)

- [x] `mloda_core/filter/global_filter.py` (Lines 104-112) ✅
  - Replaced with new helper methods: `.items()`, `.get()`, `.set()`, `__contains__`
  - Tests passed, changes staged

- [x] `mloda_core/abstract_plugins/components/feature_set.py` (Lines 29, 65) ✅
  - Line 29: Replaced `.data.keys()` with `.keys()`
  - Line 65: Replaced `.data.get()` with `.get()`
  - Tests passed, changes staged

- [x] `mloda_core/abstract_plugins/components/feature_collection.py` (Lines 36, 52-53) ✅
  - Line 36: Replaced `.data == {}` with check both `.group` and `.context`
  - Lines 52-53: Replaced `.data.items()` with `.items()`
  - Tests passed, changes staged

- [x] `mloda_core/abstract_plugins/components/feature.py` (Lines 59, 61) ✅
  - Lines 59, 61: Replaced `.data.get()` with `.get()`
  - Tests passed, changes staged

- [x] `mloda_core/abstract_plugins/components/feature_chainer/feature_chain_parser.py` (Lines 140-142) ✅
  - Removed deprecated `.data` validation check
  - Tests passed, changes staged

- [x] `mloda_core/prepare/execution_plan.py` (Line 778) ✅
  - Replaced `.data.items()` with `.items()`
  - Tests passed, changes staged

- [x] `mloda_core/abstract_plugins/components/input_data/base_input_data.py` (Line 52) ✅
  - Replaced `.data.items()` with `.items()`
  - Tests passed, changes staged

### Phase 2: Plugin Implementation Files (4 files)

- [x] `mloda_plugins/feature_group/experimental/sklearn/sklearn_artifact.py` (Lines 136, 149, 222, 257) ✅
  - All 4 occurrences: Replaced `.data.get()` and `.data.items()` with `.get()` and `.items()`
  - Tests passed, changes staged

- [x] `tests/test_plugins/feature_group/input_data/test_read_files/test_read_file.py` (Line 32) ✅
  - Replaced `.data.get()` with `.get()`
  - Tests passed, changes staged

### Phase 3: Test Files (4 files)

- [x] `tests/test_core/test_abstract_plugins/test_components/test_options.py` (Line 11) ✅
  - Removed `== options.data` comparison
  - Tests passed, changes staged

- [x] `tests/test_plugins/feature_group/experimental/test_forecasting/test_forecasting_artifact_integration.py` (Line 77) ✅
  - Replaced `.data[]=` with `.add_to_group()`
  - Tests passed, changes staged

- [x] `tests/test_plugins/feature_group/experimental/test_forecasting/test_forecasting_feature_group.py` (Line 159) ✅
  - Replaced `Options(data.copy())` with `Options(group=..., context=...)`
  - Tests passed, changes staged

- [x] `tests/test_plugins/feature_group/experimental/dynamic_feature_group_factory/test_dynamic_feature_group_factory.py` (Line 250) ✅
  - Replaced `.data.get()` with `.get()`
  - Tests passed, changes staged

### Phase 4: Remove Deprecated Property

- [x] `mloda_core/abstract_plugins/components/options.py` (Lines 57-65) ✅
  - Removed `@property data()` method
  - Tests passed, changes staged

## Verification Steps

After all changes:
- [x] Run full test suite: `tox` ✅ - All 975 tests passed
- [x] Check for any missed `.data` references ✅ - None found
- [ ] Update CHANGELOG or migration notes if applicable

## Summary

**Complete!** Successfully deprecated and removed `options.data` property.

- **14 files modified** across core, plugins, and tests
- **Added 5 new helper methods** to Options class: `.get()`, `.set()`, `.items()`, `.keys()`, `__contains__`
- **Improved documentation** for Options class
- **All tests passing** (975 passed, 112 skipped)

## Notes

### New Helper Methods Added to Options Class
- `.get(key)` - Read value from group or context (searches group first)
- `.set(key, value)` - Write value, auto-places in existing location or group by default
- `.items()` - Get all key-value pairs from both group and context
- `.keys()` - Get all keys from both group and context
- `key in options` - Check if key exists in either group or context (`__contains__`)

### Direct Access (when you need to distinguish group vs context)
- **Group access**: `.group` dict, `.add_to_group(key, value)`
- **Context access**: `.context` dict, `.add_to_context(key, value)`

## Statistics

- Total files to modify: 14 files
- Total occurrences: ~25 locations
- Test files: 4 files
- Core files: 7 files
- Plugin files: 3 files
