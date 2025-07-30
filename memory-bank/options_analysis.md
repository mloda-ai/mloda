# Options Object Analysis and Redesign

## Executive Summary

The Options object in mloda is a fundamental component used to configure Feature Groups, but its current design has significant flaws that impact the Feature Group resolution system. This document provides a comprehensive analysis of the problems and proposes a complete redesign with `group`/`context` separation.

**Key Finding**: The current Options object treats all parameters equally, causing incorrect Feature Group splitting during resolution. Some parameters should cause Feature Groups to split into independent resolved feature objects (isolation-requiring), while others should not (metadata/context).

## Current Design Problems

### 1. Feature Group Resolution Issue (Critical)

**Problem**: During Feature Group resolution, the system currently splits Feature Groups based on ANY difference in their Options objects. This is fundamentally incorrect because:

- **Over-splitting**: Feature Groups that should resolve together are split unnecessarily when they have different metadata/context options
- **Under-splitting**: No mechanism to ensure Feature Groups split when they need independent resolved feature objects

**Impact**: Inefficient execution, incorrect caching behavior, and poor performance due to unnecessary Feature Group proliferation.

### 2. Type Safety Issues

**Current Implementation**:
```python
class Options:
    def __init__(self, data: Optional[dict[str, Any]] = None) -> None:
        self.data = data or {}
```

**Problems**:
- No compile-time type checking
- Runtime errors when accessing non-existent keys
- `Any` type provides no safety guarantees
- No validation of parameter values

### 3. Inconsistent Usage Patterns

**Found across 135+ locations**:
- Mixed initialization: `Options(dict)` vs `Options()` vs direct dict usage
- Unsafe key access: `.get()` without defaults vs direct access
- Magic string dependencies via `DefaultOptionKeys`
- Complex merging logic with special cases

### 4. Validation and Schema Issues

**Current State**:
- No schema validation
- No required vs optional parameter checking
- No type constraints on values
- Silent failures when wrong types are provided

## Usage Analysis

### Distribution Across Codebase
- **mloda_core**: 58+ usages
- **mloda_plugins**: 77+ usages  
- **tests**: 300+ usages
- **Total**: 435+ locations requiring migration

### Common Patterns Found
1. **Feature Group Configuration**: Algorithm parameters, data source configs
2. **Runtime Metadata**: Source feature tracking, debugging info
3. **Execution Hints**: Performance settings, parallelization modes
4. **Validation Parameters**: Required fields, constraints

## Proposed Solution: Group/Context Separation

### New Architecture

```python
class Options:
    def __init__(self, group: Optional[dict[str, Any]] = None, context: Optional[dict[str, Any]] = None):
        self.group = group or {}      # Parameters that require independent resolved feature objects
        self.context = context or {}  # Contextual parameters that don't affect grouping
```

### Design Principles

1. **`group` parameters**: Require Feature Groups to have independent resolved feature objects
   - **Data source isolation**: Different data sources (A vs B) for testing migrations
   - **Environment separation**: Production vs staging data sources
   - **Temporal isolation**: Different time periods that must be kept separate
   - **Security boundaries**: Different access levels or data domains
   - **Testing scenarios**: Comparing different data versions or sources

2. **`context` parameters**: Provide metadata without requiring isolation
   - Algorithm parameters (same AggregationFeatureGroup can handle different aggregation types)
   - Source feature tracking (`mloda_source_feature`)
   - Debugging information
   - UI display preferences
   - Performance hints that don't change output
   - Transformation settings within the same logical group

### Feature Group Resolution Logic

```python
def should_split_feature_groups(options1: Options, options2: Options) -> bool:
    # Only compare group parameters for splitting decisions
    return options1.group != options2.group
    # context parameters are ignored for splitting
```

## Key Insight: Flexibility in Grouping

The `group` parameters are **not about algorithm types or transformation parameters**. A single Feature Group (e.g., AggregationFeatureGroup) can handle multiple different configurations and still resolve into one resolved features object.

**Example**: 
- `AggregationFeatureGroup` with `aggregation_type: "sum"` 
- `AggregationFeatureGroup` with `aggregation_type: "avg"`
- These can resolve together because they don't require isolation

**Counter-example**:
- `AggregationFeatureGroup` with `data_source: "production_db"`
- `AggregationFeatureGroup` with `data_source: "staging_db"`  
- These MUST resolve separately because we want to test data migration between sources

## Migration Strategy

### Phase 1: Core Options Class Redesign
1. Implement new Options class with `group`/`context` separation
2. Add validation and type safety mechanisms
3. Create migration utilities for existing code

### Phase 2: Systematic Codebase Migration
1. **Priority 1**: Core engine and resolution logic
2. **Priority 2**: Abstract feature groups and base classes
3. **Priority 3**: Plugin implementations
4. **Priority 4**: Test suite updates

### Phase 3: Validation and Cleanup
1. Remove old Options implementation
2. Add comprehensive validation
3. Update documentation and examples

## Parameter Categorization Guidelines

### Group Parameters (Require Independent Resolution)
- **Data source isolation**: 
  - `"data_source": "production_db"` vs `"data_source": "staging_db"`
  - `"table_name": "sales_2023"` vs `"table_name": "sales_2024"`
- **Environment separation**:
  - `"environment": "prod"` vs `"environment": "test"`
- **Temporal boundaries**:
  - `"data_version": "v1"` vs `"data_version": "v2"`
- **Security/access domains**:
  - `"access_level": "public"` vs `"access_level": "restricted"`
- **Testing scenarios**:
  - `"test_scenario": "baseline"` vs `"test_scenario": "experimental"`

### Context Parameters (Metadata Only)
- **Algorithm configurations**: `"aggregation_type": "sum"`, `"scaler_type": "standard"`
- **Transformation settings**: `"cleaning_operations": ["normalize"]`, `"n_clusters": 5`
- **Source tracking**: `"mloda_source_feature": "sales_data"`
- **Feature group tracking**: `"mloda_source_feature_group": "AggregatedFeatures"`
- **Time filters**: `"reference_time": "2023-01-01"` (when used as metadata)
- **Debugging info**: Logging levels, trace information
- **UI hints**: Display names, descriptions

## Implementation Examples

### Before (Current)
```python
# All parameters mixed together - causes incorrect splitting
options = Options({
    "data_source": "production_db",     # Should be group (requires isolation)
    "aggregation_type": "sum",          # Should be context (doesn't require isolation)
    "mloda_source_feature": "sales",   # Should be context
    "debug_mode": True                  # Should be context
})
```

### After (Proposed)
```python
# Clear separation - correct splitting behavior
options = Options(
    group={
        "data_source": "production_db"  # Only this causes splitting
    },
    context={
        "aggregation_type": "sum",      # These don't cause splitting
        "mloda_source_feature": "sales",
        "debug_mode": True
    }
)
```

### Real-world Scenario: Data Migration Testing
```python
# Production data source
prod_options = Options(
    group={"data_source": "production_db"},
    context={"aggregation_type": "sum", "test_label": "baseline"}
)

# Staging data source (must be resolved independently)
staging_options = Options(
    group={"data_source": "staging_db"},  # Different group = separate resolution
    context={"aggregation_type": "sum", "test_label": "migration_test"}
)

# These will create separate resolved feature objects for comparison
```

## Critical Design Decision: Feature Group Definition Authority

**The individual Feature Group implementation is the ultimate authority for determining which parameters belong in `group` vs `context`.**

### Feature Group Responsibility
Each Feature Group class must define:
1. **Which parameters require isolation** (belong in `group`)
2. **Which parameters are metadata only** (belong in `context`)
3. **Validation rules** for both categories
4. **Default categorization** for new parameters

### Implementation Pattern
```python
class AggregationFeatureGroup(AbstractFeatureGroup):
    # Feature Group defines what requires isolation
    GROUP_PARAMETERS = {"data_source", "environment", "data_version"}
    CONTEXT_PARAMETERS = {"aggregation_type", "mloda_source_feature", "debug_mode"}
    
    def validate_options(self, options: Options) -> None:
        # Validate that parameters are in correct categories
        for key in options.group.keys():
            if key not in self.GROUP_PARAMETERS:
                raise ValueError(f"Parameter '{key}' should be in context, not group")
        
        for key in options.context.keys():
            if key not in self.CONTEXT_PARAMETERS:
                raise ValueError(f"Parameter '{key}' should be in group, not context")
```

### Why Feature Group Authority Matters
1. **Domain Expertise**: Feature Group implementers understand their isolation requirements
2. **Flexibility**: Different Feature Groups may categorize the same parameter differently
3. **Evolution**: Feature Groups can evolve their categorization as requirements change
4. **Validation**: Feature Groups can enforce correct usage patterns
5. **Documentation**: Clear contracts about what affects resolution behavior

## Benefits of New Design

1. **Correct Feature Group Resolution**: Only split when isolation is actually required
2. **Flexible Grouping**: Same Feature Group type can handle multiple configurations
3. **Data Source Testing**: Easy to compare different data sources independently
4. **Feature Group Authority**: Each Feature Group controls its own categorization logic
5. **Type Safety**: Can add proper typing and validation to each category
6. **Clear Contracts**: Explicit about what requires isolation vs what's metadata
7. **Performance**: Reduced unnecessary Feature Group splitting

## Migration Considerations

### No Backwards Compatibility Required
- Complete redesign allowed
- Breaking changes acceptable
- Can optimize for best architecture
- Direct replacement of all existing usage

### Risk Mitigation
- Comprehensive test coverage during migration
- Module-by-module migration approach
- Validation tools to ensure correct categorization
- Clear documentation for developers
- Feature Group-specific validation during migration

## Phase 2: Feature Groups Migration Checklist

### Migration Priorities

#### Priority 1: Core Infrastructure (Foundation)
- [ x ] **AbstractFeatureGroup** - Base class parameter handling and validation framework
- [ ] **ComputeFrameWork** - Framework-specific options categorization
- [ ] **Core Engine Components** - Resolution logic and parameter processing

#### Priority 2: Experimental Feature Groups (High Usage)
- [ x ] **AggregatedFeatureGroup** 
  - **Context**: `aggregation_type` (sum, avg, min, max), `mloda_source_feature`
  - **Group**: Data source isolation parameters (if any)
  - **Authority**: Define aggregation algorithm parameters as context-only

- [ ] **MissingValueFeatureGroup**
  - **Context**: `imputation_method` (mean, median, mode, constant, ffill, bfill), `mloda_source_feature`
  - **Group**: Data source/version parameters for A/B testing different datasets
  - **Authority**: Imputation methods don't require isolation

- [ ] **TimeWindowFeatureGroup**
  - **Context**: `window_function`, `window_size`, `time_unit`, `mloda_source_feature`
  - **Group**: `data_source`, `environment` for temporal data isolation
  - **Authority**: Window calculations are metadata, data sources require isolation

- [ ] **TextCleaningFeatureGroup**
  - **Context**: `cleaning_operations` (normalize, remove_stopwords, remove_punctuation), `mloda_source_feature`
  - **Group**: `data_source`, `language_model_version` for different text processing environments
  - **Authority**: Text processing parameters are algorithmic, data sources require isolation

- [ ] **ClusteringFeatureGroup**
  - **Context**: `clustering_algorithm`, `n_clusters`, `auto_k`, `mloda_source_feature`
  - **Group**: `data_source`, `feature_set_version` for comparing clustering on different datasets
  - **Authority**: Clustering algorithms are computational, data versions require isolation

#### Priority 3: Advanced Feature Groups
- [ ] **DimensionalityReductionFeatureGroup**
  - **Context**: `algorithm` (pca, tsne, ica, lda, isomap), `n_components`, `mloda_source_feature`
  - **Group**: `data_source`, `preprocessing_version` for different data preparation pipelines
  - **Authority**: Dimensionality algorithms are computational parameters

- [ ] **ForecastingFeatureGroup**
  - **Context**: `algorithm`, `horizon`, `time_unit`, `mloda_source_feature`, artifact parameters
  - **Group**: `data_source`, `training_period`, `model_version` for comparing forecasts on different data
  - **Authority**: Forecasting algorithms are computational, training data requires isolation

- [ ] **GeoDistanceFeatureGroup**
  - **Context**: `distance_type` (haversine, euclidean, manhattan), `mloda_source_feature`
  - **Group**: `coordinate_system`, `data_source` for different geographic reference systems
  - **Authority**: Distance calculations are computational, coordinate systems require isolation

- [ ] **NodeCentralityFeatureGroup**
  - **Context**: `centrality_type`, `graph_type` (directed/undirected), `mloda_source_feature`
  - **Group**: `graph_data_source`, `network_version` for comparing centrality on different networks
  - **Authority**: Centrality algorithms are computational, network data requires isolation

#### Priority 4: Sklearn Feature Groups
- [ ] **SklearnPipelineFeatureGroup**
  - **Context**: `pipeline_name`, `pipeline_config`, `mloda_source_feature`, artifact storage parameters
  - **Group**: `data_source`, `sklearn_version`, `model_environment` for ML pipeline isolation
  - **Authority**: Pipeline configurations are computational, model environments require isolation

- [ ] **SklearnScalingFeatureGroup**
  - **Context**: `scaler_type` (standard, minmax, robust), scaler parameters, `mloda_source_feature`
  - **Group**: `data_source`, `training_set_version` for scaling based on different training data
  - **Authority**: Scaling algorithms are computational, training data requires isolation

- [ ] **SklearnEncodingFeatureGroup**
  - **Context**: `encoder_type` (onehot, label, ordinal), encoder parameters, `mloda_source_feature`
  - **Group**: `data_source`, `category_mapping_version` for different encoding schemes
  - **Authority**: Encoding algorithms are computational, category mappings require isolation

#### Priority 5: Input Data Feature Groups
- [ ] **ReadFileFeatureGroup** (CSV, Parquet, JSON, Feather, ORC, Text)
  - **Context**: `file_format_options`, `parsing_parameters`, `column_selection`
  - **Group**: `file_path`, `data_source`, `file_version` for reading different data sources
  - **Authority**: File paths and versions require isolation, parsing options are metadata

- [ ] **ReadDBFeatureGroup** (SQLite, etc.)
  - **Context**: `query_parameters`, `result_formatting`, `column_selection`
  - **Group**: `database_connection`, `schema_version`, `environment` for different databases
  - **Authority**: Database connections require isolation, query parameters are metadata

- [ ] **APIDataFeatureGroup**
  - **Context**: `request_parameters`, `response_formatting`, `retry_settings`
  - **Group**: `api_endpoint`, `api_version`, `environment`, `authentication` for different API sources
  - **Authority**: API endpoints and auth require isolation, request parameters are metadata

### Feature Group Authority Implementation Pattern

Each Feature Group must implement:

```python
class ExampleFeatureGroup(AbstractFeatureGroup):
    # Define which parameters require isolation (group)
    GROUP_PARAMETERS = {"data_source", "environment", "data_version"}
    
    # Define which parameters are metadata only (context)
    CONTEXT_PARAMETERS = {"algorithm_type", "mloda_source_feature", "debug_mode"}
    
    def validate_options(self, options: Options) -> None:
        """Validate that parameters are in correct categories."""
        # Validate group parameters
        for key in options.group.keys():
            if key not in self.GROUP_PARAMETERS:
                if key in self.CONTEXT_PARAMETERS:
                    raise ValueError(f"Parameter '{key}' should be in context, not group")
                # Allow unknown parameters in group during migration
        
        # Validate context parameters
        for key in options.context.keys():
            if key not in self.CONTEXT_PARAMETERS:
                if key in self.GROUP_PARAMETERS:
                    raise ValueError(f"Parameter '{key}' should be in group, not context")
                # Allow unknown parameters in context during migration
    
    @classmethod
    def migrate_legacy_options(cls, legacy_options: dict[str, Any]) -> Options:
        """Helper method to migrate legacy options to group/context separation."""
        group = {}
        context = {}
        
        for key, value in legacy_options.items():
            if key in cls.GROUP_PARAMETERS:
                group[key] = value
            elif key in cls.CONTEXT_PARAMETERS:
                context[key] = value
            else:
                # During migration, unknown parameters go to group for safety
                group[key] = value
        
        return Options(group=group, context=context)
```

### Migration Testing Strategy

For each Feature Group:

1. **Parameter Analysis**: Review current usage patterns to identify group vs context parameters
2. **Authority Definition**: Implement GROUP_PARAMETERS and CONTEXT_PARAMETERS sets
3. **Validation Testing**: Ensure parameters are correctly categorized
4. **Behavior Testing**: Verify Feature Group resolution works correctly after migration
5. **Performance Testing**: Confirm reduced unnecessary splitting improves performance

### Migration Success Criteria

- [ ] **Functional**: Feature Group resolution correctly isolates only when needed
- [ ] **Authority**: Each Feature Group properly validates its parameter categorization
- [ ] **Performance**: Reduced unnecessary Feature Group splitting
- [ ] **Compatibility**: All existing functionality continues to work
- [ ] **Documentation**: Clear examples of group vs context usage for each Feature Group

## Next Steps

1. **✅ COMPLETED: Implement new Options class** with group/context separation
2. **Define Feature Group categorization interfaces** for parameter validation
3. **Create migration utilities** to help categorize existing parameters
4. **Update core resolution logic** to use only group parameters
5. **Begin systematic migration** using the checklist above
6. **Add Feature Group-specific validation** for parameter categorization
7. **Update all documentation** and examples

## Success Metrics

- **Functional**: Feature Group resolution works correctly with proper isolation
- **Flexibility**: Same Feature Group types can handle multiple configurations
- **Authority**: Feature Groups properly control their own parameter categorization
- **Performance**: Reduced unnecessary Feature Group splitting
- **Code Quality**: Type safety and validation in place
- **Developer Experience**: Clear, intuitive API for configuration

This redesign addresses the fundamental architectural issues with the Options object and provides a clear path forward for the entire mloda codebase, with proper understanding of when Feature Groups should be isolated vs when they can be grouped together, and with each Feature Group maintaining authority over its own parameter categorization.
