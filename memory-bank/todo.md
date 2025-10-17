# 🧩 TODO: Feature Configuration Plugin - Extended Support

## 🎯 Goal

Extend the **feature configuration system** under `mloda_plugins/config/feature/`
to support the core mloda patterns identified in `missing_feature_configs.md`.

* 🔹 Uses **Pydantic** for schema validation
* 🔹 Supports **chained features** with `__` syntax
* 🔹 Supports **group/context** options separation
* 🔹 Supports **multi-column access** with `~` syntax
* 🔹 Fully **TDD-driven**, with tests under the global `tests/` directory
* 🔹 **Integration JSON file** grows with each phase

---

## ✅ Phase 0-6: Basic Support (COMPLETED)

* [x] **Phase 0**: Initial Setup
* [x] **Phase 1**: End-to-End Test
* [x] **Phase 2**: Pydantic Schema
* [x] **Phase 3**: Parser Abstraction (JSON)
* [x] **Phase 4**: Loader Logic
* [x] **Phase 5**: Integration Test
* [x] **Phase 6**: Schema Export Utility

---

## ✅ Phase 7: Chained Feature Support (COMPLETED)

### Goal
Support mloda's core chaining pattern: `operation__source_feature`

### File Structure
```
tests/test_plugins/config/feature/
├── test_feature_config_chained.py          # New test file
├── test_config_features.json               # NEW: Growing integration test data
mloda_plugins/config/feature/
├── models.py                                # Update schema
├── parser.py                                # Update parser
├── loader.py                                # Update loader
```

### Tasks

* [x] **Create test file**: `tests/test_plugins/config/feature/test_feature_config_chained.py`
  * test_parse_simple_chained_feature
  * test_parse_multi_level_chained_feature
  * test_load_chained_feature_as_string
  * test_load_chained_feature_from_config

* [x] **Update `models.py`**: Add support for chained feature representation
  ```python
  class FeatureConfig(BaseModel):
      name: str
      options: Dict[str, Any] = {}
      mloda_source: Optional[str] = None        # NEW: "age"
  ```

* [x] **Update parser tests**: `tests/test_plugins/config/feature/test_feature_config_parser.py`
  * test_parse_json_with_mloda_source_field
  * test_parse_chained_feature_string

* [x] **Update `parser.py`**: Handle mloda_source field in JSON parsing

* [x] **Update loader tests**: `tests/test_plugins/config/feature/test_feature_config_loader.py`
  * test_load_features_with_mloda_source
  * test_load_features_mixed_chained_and_simple

* [x] **Update `loader.py`**: Handle mloda_source in feature creation
  ```python
  # Example: name="scale__impute__age", mloda_source="age"
  # Add mloda_source to context options
  ```

* [x] **Create/Update integration JSON**: `tests/test_plugins/config/feature/test_config_features.json`
  ```json
  [
    "age",
    {"name": "weight", "options": {"imputation_method": "mean"}},
    {
      "name": "standard_scaled__mean_imputed__age",
      "mloda_source": "age"
    },
    {
      "name": "max_aggr__onehot_encoded__state",
      "mloda_source": "state"
    }
  ]
  ```

* [x] **Update end-to-end test**: `tests/test_plugins/config/feature/test_feature_config_end2end.py`
  * test_end2end_chained_features
  * test_integration_json_file (loads and validates test_config_features.json)
  * Verify chained features work with mlodaAPI.run_all

* [x] **End of Phase**: Run `tox` → All tests pass ✓ → Run `git add .`

---

## ✅ Phase 8: Group/Context Options Separation (COMPLETED)

### Goal
Support the new Options architecture with group/context separation for performance optimization.

### File Structure
```
tests/test_plugins/config/feature/
├── test_feature_config_options.py           # New test file
├── test_config_features.json                # UPDATE: Add group/context examples
mloda_plugins/config/feature/
├── models.py                                # Update schema
├── loader.py                                # Update loader
```

### Tasks

* [x] **Create test file**: `tests/test_plugins/config/feature/test_feature_config_options.py`
  * test_parse_group_options
  * test_parse_context_options
  * test_parse_group_and_context_together
  * test_load_creates_options_with_group_context

* [x] **Update `models.py`**: Add group/context fields
  ```python
  class FeatureConfig(BaseModel):
      name: str
      options: Dict[str, Any] = {}              # Convenient flat options if you don't care about group/context
      group_options: Optional[Dict[str, Any]] = None    # NEW
      context_options: Optional[Dict[str, Any]] = None  # NEW
      mloda_source: Optional[str] = None
  ```

* [x] **Add validation**: Ensure options + group_options/context_options are mutually exclusive
  ```python
  @model_validator(mode='after')
  def validate_options_structure(self) -> 'FeatureConfig':
      if self.options and (self.group_options or self.context_options):
          raise ValueError("Cannot use both 'options' and 'group_options'/'context_options'")
      return self
  ```

* [x] **Update parser tests**: `tests/test_plugins/config/feature/test_feature_config_parser.py`
  * test_parse_json_with_group_context_options
  * test_parse_json_rejects_mixed_options_formats

* [x] **Update `parser.py`**: Handle group_options/context_options fields

* [x] **Update loader tests**: `tests/test_plugins/config/feature/test_feature_config_loader.py`
  * test_load_features_with_legacy_options
  * test_load_features_with_group_context_options
  * test_load_creates_proper_options_object

* [x] **Update `loader.py`**: Create Options with group/context separation
  ```python
  from mloda_core.abstract_plugins.components.options import Options

  if item.group_options or item.context_options:
      options_obj = Options(
          group=item.group_options or {},
          context=item.context_options or {}
      )
  else:
      options_obj = Options(data=item.options)  # Legacy
  ```

* [x] **Update integration JSON**: `tests/test_plugins/config/feature/test_config_features.json`
  ```json
  [
    "age",
    {"name": "weight", "options": {"imputation_method": "mean"}},
    {
      "name": "standard_scaled__mean_imputed__age",
      "mloda_source": "age"
    },
    {
      "name": "production_feature",
      "group_options": {
        "data_source": "production"
      },
      "context_options": {
        "aggregation_type": "sum",
        "window_size": 7
      }
    }
  ]
  ```

* [x] **Update end-to-end test**: `tests/test_plugins/config/feature/test_feature_config_end2end.py`
  * test_end2end_group_context_options
  * test_integration_json_file (verify new entries work)
  * Verify group/context separation works in full pipeline

* [x] **End of Phase**: Run `tox` → All tests pass ✓ → Run `git add .`

---

## ✅ Phase 9: Multi-Column Access Support (COMPLETED)

### Goal
Support the `~` syntax for accessing specific columns from multi-output transformations.

### File Structure
```
tests/test_plugins/config/feature/
├── test_feature_config_column_selector.py   # New test file
├── test_config_features.json                # UPDATE: Add column selector examples
mloda_plugins/config/feature/
├── models.py                                # Update schema
├── parser.py                                # Update parser
├── loader.py                                # Update loader
```

### Tasks

* [x] **Create test file**: `tests/test_plugins/config/feature/test_feature_config_column_selector.py`
  * test_parse_feature_with_column_index
  * test_parse_feature_with_tilde_syntax
  * test_load_column_selector_feature

* [x] **Update `models.py`**: Add column_index field
  ```python
  class FeatureConfig(BaseModel):
      name: str
      options: Dict[str, Any] = {}
      group_options: Optional[Dict[str, Any]] = None
      context_options: Optional[Dict[str, Any]] = None
      mloda_source: Optional[str] = None
      column_index: Optional[int] = None        # NEW: Access specific column
  ```

* [x] **Update parser tests**: `tests/test_plugins/config/feature/test_feature_config_parser.py`
  * test_parse_json_with_column_index
  * test_parse_json_handles_tilde_in_name

* [x] **Update `parser.py`**: Handle column_index field and `~` in names

* [x] **Update loader tests**: `tests/test_plugins/config/feature/test_feature_config_loader.py`
  * test_load_features_with_column_index
  * test_load_appends_tilde_syntax_to_name

* [x] **Update `loader.py`**: Append `~{index}` to feature name when column_index is set
  ```python
  feature_name = item.name
  if item.column_index is not None:
      feature_name = f"{feature_name}~{item.column_index}"
  ```

* [x] **Update integration JSON**: `tests/test_plugins/config/feature/test_config_features.json`
  ```json
  [
    "age",
    {"name": "weight", "options": {"imputation_method": "mean"}},
    {
      "name": "standard_scaled__mean_imputed__age",
      "mloda_source": "age"
    },
    {
      "name": "production_feature",
      "group_options": {"data_source": "production"},
      "context_options": {"aggregation_type": "sum"}
    },
    {
      "name": "onehot_encoded__state",
      "column_index": 0
    },
    {
      "name": "onehot_encoded__state",
      "column_index": 1
    }
  ]
  ```

* [x] **Update end-to-end test**: `tests/test_plugins/config/feature/test_feature_config_end2end.py`
  * test_end2end_multi_column_access
  * test_integration_json_file (verify column selectors work)
  * Verify `~` syntax works with one-hot encoding

* [x] **End of Phase**: Run `tox` → All tests pass ✓ → Run `git add .`

---

## ✅ Phase 11: Nested Feature References (COMPLETED)

### Goal
Support referencing other features as sources using `@feature_name` syntax.

### File Structure
```
tests/test_plugins/config/feature/
├── test_feature_config_nested_refs.py       # New test file
├── test_config_features.json                # UPDATE: Add reference examples
mloda_plugins/config/feature/
├── models.py                                # Update schema
├── parser.py                                # Update parser
├── loader.py                                # Update loader
```

### Tasks

* [x] **Create test file**: `tests/test_plugins/config/feature/test_feature_config_nested_refs.py`
  * test_parse_feature_with_reference
  * test_load_resolves_feature_references
  * test_load_handles_nested_references

* [x] **Update `models.py`**: Support `@feature_name` in mloda_source field
  * No schema change needed, handled in loader

* [x] **Update parser tests**: `tests/test_plugins/config/feature/test_feature_config_parser.py`
  * test_parse_json_with_at_reference_in_mloda_source
  * test_parse_json_preserves_reference_syntax

* [x] **Update `parser.py`**: Preserve `@` prefix in mloda_source strings

* [x] **Update loader tests**: `tests/test_plugins/config/feature/test_feature_config_loader.py`
  * test_load_detects_feature_references
  * test_load_resolves_references_to_feature_objects
  * test_load_handles_forward_references

* [x] **Update `loader.py`**: Resolve `@feature_name` references to Feature objects
  ```python
  def resolve_references(config_items: List[FeatureConfig]) -> List[Feature]:
      """Two-pass loader: create all features, then resolve references."""
      # Pass 1: Create all features
      features_by_name = {}
      for item in config_items:
          feature = create_feature_from_config(item)
          features_by_name[item.name] = feature

      # Pass 2: Resolve @references in mloda_source/mloda_source_feature
      for item in config_items:
          if item.mloda_source and item.mloda_source.startswith("@"):
              ref_name = item.mloda_source[1:]  # Remove @
              if ref_name in features_by_name:
                  # Update feature's options to reference the Feature object
                  pass

      return list(features_by_name.values())
  ```

* [x] **Update integration JSON**: `tests/test_plugins/config/feature/test_config_features.json`
  ```json
  [
    "age",
    {"name": "weight", "options": {"imputation_method": "mean"}},
    {
      "name": "standard_scaled__mean_imputed__age",
      "mloda_source": "age"
    },
    {
      "name": "production_feature",
      "group_options": {"data_source": "production"},
      "context_options": {"aggregation_type": "sum"}
    },
    {
      "name": "onehot_encoded__state",
      "column_index": 0
    },
    {
      "name": "pandas_feature",
      "compute_framework": "PandasDataframe"
    },
    {
      "name": "derived_from_scaled",
      "mloda_source": "@scaled_age",
      "options": {"transformation": "log"}
    },
    {
      "name": "nested_reference",
      "mloda_source": "@derived_from_scaled",
      "options": {"normalization": "minmax"}
    }
  ]
  ```

* [x] **Update end-to-end test**: `tests/test_plugins/config/feature/test_feature_config_end2end.py`
  * test_end2end_nested_feature_references
  * test_integration_json_file (verify references resolve correctly)
  * Verify feature dependencies work correctly

* [x] **End of Phase**: Run `tox` → All tests pass ✓ → Run `git add .`

---

## ✅ Phase 12: Multiple Source Features (COMPLETED)

### Goal
Support features that require multiple source features (e.g., distance calculations).

### File Structure
```
tests/test_plugins/config/feature/
├── test_feature_config_multi_source.py      # New test file
├── test_config_features.json                # UPDATE: Add multi-source examples
mloda_plugins/config/feature/
├── models.py                                # Update schema
├── parser.py                                # Update parser
├── loader.py                                # Update loader
```

### Tasks

* [x] **Create test file**: `tests/test_plugins/config/feature/test_feature_config_multi_source.py`
  * test_parse_feature_with_multiple_sources
  * test_load_multiple_sources_as_frozenset
  * test_validate_source_vs_sources_mutual_exclusion

* [x] **Update `models.py`**: Add sources field (plural)
  ```python
  class FeatureConfig(BaseModel):
      name: str
      options: Dict[str, Any] = {}
      group_options: Optional[Dict[str, Any]] = None
      context_options: Optional[Dict[str, Any]] = None
      mloda_source: Optional[str] = None        # Single source
      mloda_sources: Optional[List[str]] = None # NEW: Multiple sources
      column_index: Optional[int] = None
      compute_framework: Optional[str] = None

      @model_validator(mode='after')
      def validate_source_fields(self) -> 'FeatureConfig':
          if self.mloda_source and self.mloda_sources:
              raise ValueError("Cannot specify both 'mloda_source' and 'mloda_sources'")
          return self
  ```

* [x] **Update parser tests**: `tests/test_plugins/config/feature/test_feature_config_parser.py`
  * test_parse_json_with_mloda_sources_array
  * test_parse_json_rejects_mloda_source_and_mloda_sources_together

* [x] **Update `parser.py`**: Handle mloda_sources field

* [x] **Update loader tests**: `tests/test_plugins/config/feature/test_feature_config_loader.py`
  * test_load_features_with_multiple_mloda_sources
  * test_load_creates_frozenset_for_mloda_sources
  * test_load_adds_mloda_sources_to_mloda_source_feature_option

* [x] **Update `loader.py`**: Convert mloda_sources to frozenset in options
  ```python
  if item.mloda_sources:
      options_obj.add_to_context(
          DefaultOptionKeys.mloda_source_features,
          frozenset(item.mloda_sources)
      )
  ```

* [x] **Update integration JSON**: `tests/test_plugins/config/feature/test_config_features.json`
  ```json
  [
    "age",
    {"name": "weight", "options": {"imputation_method": "mean"}},
    {
      "name": "standard_scaled__mean_imputed__age",
      "mloda_source": "age"
    },
    {
      "name": "production_feature",
      "group_options": {"data_source": "production"},
      "context_options": {"aggregation_type": "sum"}
    },
    {
      "name": "onehot_encoded__state",
      "column_index": 0
    },
    {
      "name": "pandas_feature",
      "compute_framework": "PandasDataframe"
    },
    {
      "name": "derived_from_scaled",
      "mloda_source": "@scaled_age",
      "options": {"transformation": "log"}
    },
    {
      "name": "distance_feature",
      "mloda_sources": ["latitude", "longitude"],
      "options": {"distance_type": "euclidean"}
    },
    {
      "name": "multi_source_aggregation",
      "mloda_sources": ["sales", "revenue", "profit"],
      "options": {"aggregation": "sum"}
    }
  ]
  ```

* [x] **Update end-to-end test**: `tests/test_plugins/config/feature/test_feature_config_end2end.py`
  * test_end2end_multiple_source_features
  * test_integration_json_file (verify multi-source features work)
  * Verify features with multiple sources work (e.g., distance calculations)

* [x] **End of Phase**: Run `tox` → All tests pass ✓ → Run `git add .`

---

## ✅ Phase 13: Final Integration Test and Documentation (COMPLETED)

### Goal
Comprehensive integration test with full JSON file and complete documentation.

### File Structure
```
tests/test_plugins/config/feature/
├── test_config_features.json                # FINAL: Complete example
├── test_feature_config_end2end.py           # FINAL: Comprehensive test
docs/plugins/
├── feature_config.md                        # Complete documentation
```

### Tasks

* [x] **Finalize integration JSON**: `tests/test_plugins/config/feature/test_config_features.json`
  * Verify all patterns are represented
  * Add comments (if JSON5 or external doc)
  * Ensure valid and comprehensive

* [x] **Create comprehensive integration test**: `tests/test_plugins/config/feature/test_feature_config_end2end.py`
  * test_complete_integration_json validates ALL supported patterns

* [x] **Update `mloda_plugins/config/feature/__init__.py`**: Add comprehensive docstring
  * Added full documentation with supported patterns
  * Basic usage examples
  * Schema export instructions

* [x] **Create doc page**: `docs/docs/plugins/feature_config.md`
  * Complete overview of configuration system
  * JSON schema documentation
  * Examples for all 7 supported patterns
  * Integration with mlodaAPI
  * Migration guide from code to config
  * Best practices and performance tips

* [x] **Create example file**: `docs/docs/examples/feature_config_examples.json`
  * Comprehensive example with all patterns
  * Copied from test_config_features.json

* [x] **Update README.md**: Add section on configuration-based feature definition
  * Quick start example with JSON configuration
  * Supported patterns overview
  * Benefits and use cases
  * Link to full documentation

* [x] **Create tutorial**: `docs/docs/tutorials/feature_configuration.md`
  * Step-by-step guide (7 steps)
  * Common patterns and recipes
  * Complete real-world example
  * Troubleshooting guide
  * Performance considerations

* [x] **End of Phase**: Run `tox` → All tests pass ✓ → Run `git add .`

---

## 📋 Summary

### Phase Priority
1. ✅ **Phases 0-6**: Basic support (COMPLETED)
2. ✅ **Phase 7**: Chained features (COMPLETED)
3. ✅ **Phase 8**: Group/context options (COMPLETED)
4. ✅ **Phase 9**: Multi-column access (COMPLETED)
5. ✅ **Phase 11**: Nested references (COMPLETED)
6. ✅ **Phase 12**: Multiple sources (COMPLETED)
7. ✅ **Phase 13**: Final integration & docs (COMPLETED)

### Key Integration Test File
```
tests/test_plugins/config/feature/test_config_features.json
```
This file grows with each phase, accumulating examples of all supported patterns.

### Key Files Modified
```
mloda_plugins/config/feature/
├── __init__.py          # Docstrings
├── models.py            # Extended schema (Phases 7-12)
├── parser.py            # Enhanced parsing (Phases 7-12)
├── loader.py            # Reference resolution (Phases 7-12)

tests/test_plugins/config/feature/
├── test_config_features.json                # GROWS EACH PHASE
├── test_feature_config_end2end.py           # Integration tests
├── test_feature_config_model.py             # Schema tests
├── test_feature_config_parser.py            # Parser tests
├── test_feature_config_loader.py            # Loader tests
├── test_feature_config_schema.py            # Schema export tests
├── test_feature_config_chained.py           # Phase 7
├── test_feature_config_options.py           # Phase 8
├── test_feature_config_column_selector.py   # Phase 9
├── test_feature_config_nested_refs.py       # Phase 11
├── test_feature_config_multi_source.py      # Phase 12
```

### Testing Protocol
After EVERY phase:
1. **Update** `test_config_features.json` with new pattern examples
2. Run `tox` to execute all tests (including integration test with full JSON)
3. Verify all tests pass ✓
4. If tests pass: Run `git add .` to stage changes (including JSON file)
5. If tests fail: Fix issues before proceeding to next phase

### Final Deliverable
A comprehensive `test_config_features.json` that demonstrates ALL supported patterns:
- Simple strings
- Objects with options
- Chained features (with mloda_source)
- Group/context options
- Multi-column access (column_index)
- Feature references (@feature_name)
- Multiple source features (mloda_sources array)

This JSON file serves as both:
- Integration test data
- Living documentation
- Reference implementation for users
