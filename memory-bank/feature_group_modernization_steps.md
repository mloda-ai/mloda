# Feature Group Modernization Steps

This document outlines the comprehensive steps needed to modernize feature groups to use the new configuration-based approach with proper group/context parameter separation.

## Overview

The modernization process transforms feature groups from old string-based parsing to the new unified `FeatureChainParser.match_configuration_feature_chain_parser` approach with proper `PROPERTY_MAPPING` configuration.

Updated example is the aggregation feature group.

## todo

[x] aggregated_feature_group
[x] clustering
[x] data quality (missing_value)
[x] dimensionality_reduction
[x] forecasting
[x] geo_distance
[x] node_centrality
[x] sklearn
[x] text_cleaning
[x] time_window

## Step-by-Step Modernization Process

### Phase 1: Add PROPERTY_MAPPING Configuration

1. **Define PROPERTY_MAPPING** in the base feature group class:
   ```python
   PROPERTY_MAPPING = {
       # Feature-specific parameters (e.g., AGGREGATION_TYPE)
       FEATURE_PARAMETER: {
           **VALID_VALUES_DICT,  # All supported values as valid options
           DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
           DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
       },
       # Source feature parameter
       DefaultOptionKeys.mloda_source_feature: {
           "explanation": "Source feature description",
           DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
           DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
       },
   }
   ```

2. **Parameter Classification Rules**:
   - **Context Parameters**: Don't affect Feature Group resolution/splitting
     - Feature-specific parameters (aggregation_type, algorithm_type, etc.)
     - mloda_source_feature
   - **Group Parameters**: Affect Feature Group resolution/splitting
     - Data source isolation parameters
     - Environment-specific parameters

### Phase 2: Update match_feature_group_criteria

**Replace current implementation:**
```python
# OLD: Pattern-only or hybrid approach
if not FeatureChainParser.match_configuration_feature_chain_parser(
    feature_name, options, pattern=cls.PATTERN, prefix_patterns=[cls.PREFIX_PATTERN]
):
    return False
# ... additional validation logic

# NEW: Unified approach with property mapping
return FeatureChainParser.match_configuration_feature_chain_parser(
    feature_name, options, 
    property_mapping=cls.PROPERTY_MAPPING,
    pattern=cls.PATTERN, 
    prefix_patterns=[cls.PREFIX_PATTERN]
)
```

### Phase 3: Refactor calculate_feature Method

**Current Approach** (string-based parsing):
```python
for feature_name in features.get_all_names():
    param, source_feature = FeatureChainParser.parse_feature_name(
        feature_name, cls.PATTERN, [cls.PREFIX_PATTERN]
    )
    # Process using parsed values
```

**New Approach** (configuration-based):
```python
for feature in features.features:
    # Try configuration-based approach first
    try:
        source_features = feature.options.get_source_features()
        source_feature = next(iter(source_features))
        source_feature_name = source_feature.get_name()
        
        # Extract parameters from options
        param_value = feature.options.get("parameter_name")
        
    except (ValueError, StopIteration):
        # Fall back to string-based approach for legacy features
        param_value, source_feature_name = FeatureChainParser.parse_feature_name(
            feature.name, cls.PATTERN, [cls.PREFIX_PATTERN]
        )
    
    # Process using extracted values
```

### Phase 4: Update input_features Method

**Current**: String parsing only
```python
_, source_feature = FeatureChainParser.parse_feature_name(feature_name, self.PATTERN, [self.PREFIX_PATTERN])
if source_feature is not None:
    return {Feature(source_feature)}
```

**New**: Handle both approaches - string-based parsing first, then configuration-based fallback

```python
def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
    """Extract source feature from either configuration-based options or string parsing."""

    source_feature: str | None = None

    # Try string-based parsing first
    _, source_feature = FeatureChainParser.parse_feature_name(feature_name, self.PATTERN, [self.PREFIX_PATTERN])
    if source_feature is not None:
        return {Feature(source_feature)}

    # Fall back to configuration-based approach
    source_features = options.get_source_features()
    if len(source_features) != 1:
        raise ValueError(
            f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
        )
    return set(source_features)
```

### Phase 5: Update All Implementations

For each compute framework implementation (Pandas, PyArrow, Polars, etc.):

1. **Update calculate_feature method** to use the new approach
2. **Test both string-based and configuration-based features**
3. **Ensure proper error handling** for both approaches

### Phase 6: Update Tests

**Breaking Changes in Tests**:
1. **Parameter placement**: Move parameters to correct group/context categories
2. **Feature creation**: Update to use new Options structure
3. **Validation**: Update expected validation behavior
4. **Integration tests**: Ensure both approaches work

**Test Update Pattern**:
```python
# OLD
feature = Feature("sum_aggr__sales", Options({"some_param": "value"}))

# NEW - Configuration-based
feature = Feature(
    "placeholder",
    Options(
        group={
            # Group parameters (affect resolution)
        },
        context={
            "aggregation_type": "sum",
            DefaultOptionKeys.mloda_source_feature: "sales",
            # Other context parameters
        }
    )
)
```

### Phase 7: Update Documentation

1. **Update docstrings** to reflect new parameter classification
2. **Update examples** to use new Options structure
3. **Document breaking changes** and migration path
4. **Update feature naming conventions** if needed

## Implementation Order

1. **Base class** - Core PROPERTY_MAPPING and method updates
2. **One implementation** - Validate approach (e.g., Pandas)
3. **Run targeted tests** - Identify what breaks
4. **Fix broken tests** - Update to new approach
5. **Remaining implementations** - Apply same pattern
6. **Integration tests** - End-to-end validation
7. **Documentation** - Update examples and guides

## Key Considerations

### Breaking Changes
- **Non-backwards compatible**: Old string-based approach may not work
- **Parameter placement**: Context vs group classification affects behavior
- **Validation**: Stricter validation may reject previously valid inputs

### Benefits
- **Modern architecture**: Aligned with new Options group/context separation
- **Proper parameter classification**: Context parameters don't affect Feature Group splitting
- **Consistent validation**: Built-in validation rules
- **Flexibility**: Supports both creation approaches during transition

### Testing Strategy
- **Dual approach testing**: Validate both string-based and configuration-based work
- **Parameter classification**: Ensure context parameters don't affect resolution
- **Validation rules**: Test strict validation behavior
- **Integration**: End-to-end testing with mlodaAPI

## Template for Other Feature Groups

This process can be applied to any feature group that needs modernization:

1. **Identify parameters** and classify as group vs context
2. **Create PROPERTY_MAPPING** with proper validation rules
3. **Update match_feature_group_criteria** to use unified parser
4. **Refactor calculate_feature** to handle both approaches
5. **Update input_features** for dual support
6. **Update all implementations** consistently
7. **Try to get rid of configurable_feature_chain_parser in the FeatureGroup** as it has no task anymore.

8. **Fix tests** and update documentation

## Success Criteria

- ✅ Both string-based and configuration-based features work
- ✅ Context parameters don't affect Feature Group resolution
- ✅ Strict validation works as expected
- ✅ All tests pass with new approach
- ✅ Integration tests validate end-to-end functionality
- ✅ Documentation reflects new approach

## Completed Implementations

### ✅ Data Quality (Missing Value) Feature Group - COMPLETED

**Key Learnings for Future Modernizations:**

1. **Dual Approach Success**: Both string-based and configuration-based features working together seamlessly
2. **Parameter Parsing Logic**: Get rid of `get_prefix_part()` by using `parse_feature_name()` + string manipulation. Fix: apply the fix depending on the feature group.
3. **Default Values**: Explicit defaults needed in PROPERTY_MAPPING for all optional parameters (constant_value=None, group_by_features=None)
4. **Unified Parser Integration**: Replace old single-approach parsers with `match_configuration_feature_chain_parser`

### ✅ Dimensionality Reduction Feature Group - COMPLETED

**Key Learnings and Unexpected Challenges:**

1. **Complex Prefix Patterns Need Hybrid Validation**: For feature groups with complex prefix patterns containing multiple parameters (like `{algorithm}_{dimension}d__`), the unified parser alone is insufficient. Need hybrid approach like clustering feature group:
   - PROPERTY_MAPPING with validation functions for configuration-based features
   - Additional validation in `match_feature_group_criteria` for string-based features

2. **Validation Function Requirements**: PROPERTY_MAPPING needs `mloda_validation_function` for complex parameter validation (e.g., dimension must be positive integer)

### ✅ Forecasting Feature Group - COMPLETED

**Key Learnings and Implementation Notes:**

1. **Dual Approach Success**: Both string-based and configuration-based features working correctly
2. **Parameter Classification**: All forecasting parameters (algorithm, horizon, time_unit, source_feature) classified as context parameters
3. **Unified Parser Integration**: Successfully replaced old validation with `match_configuration_feature_chain_parser`

### ✅ Geo Distance Feature Group - COMPLETED

**Key Learnings and Implementation Notes:**

1. **Validation Function Flexibility**: PROPERTY_MAPPING validation functions need to handle both individual elements and collections:
   ```python
   DefaultOptionKeys.mloda_validation_function: lambda x: (
       isinstance(x, str) or  # Individual strings when parser iterates
       (isinstance(x, (list, tuple, frozenset, set)) and len(x) == 2)  # Collections
   )
   ```

3. **Integration Test Modernization**: Tests using old `configurable_feature_chain_parser()` method need to be updated to use configuration-based feature creation with proper Options structure.

4. **Dual Approach Success**: Both string-based (`"haversine_distance__sf__nyc"`) and configuration-based features working seamlessly with proper parameter extraction logic.

5. **Multiple Source Features**: Successfully handled feature groups requiring exactly 2 source features with proper validation in both string parsing and configuration approaches.

### ✅ Sklearn Pipeline Feature Group - COMPLETED

**Key Learnings and Implementation Notes:**

1. **Tuple Support in Unified Parser**: The unified parser initially rejected tuples, requiring modification to convert tuples to string representations for hashability:
   ```python
   if isinstance(found_property_val, tuple):
       # Convert tuple to string representation for hashability
       found_property_val = str(found_property_val)
   ```

2. **Mutual Exclusivity Validation**: Successfully implemented custom validation for mutually exclusive parameters (PIPELINE_NAME vs PIPELINE_STEPS) in `match_feature_group_criteria`:
   ```python
   # For configuration-based features, must have exactly one of PIPELINE_NAME or PIPELINE_STEPS
   if has_pipeline_name and has_pipeline_steps:
       return False
   ```

3. **Complex Parameter Handling**: Handled complex parameter structures including:
   - `PIPELINE_STEPS`: Frozenset of (name, transformer_class_name) tuples
   - `PIPELINE_PARAMS`: Dictionary/frozenset of pipeline parameters
   - Mutual exclusivity between predefined and custom pipelines

4. **Frozenset Reconstruction**: Implemented methods to convert frozensets back to usable sklearn objects:
   - `_reconstruct_pipeline_steps_from_frozenset()`: Convert frozenset back to list of (name, transformer) tuples
   - `_reconstruct_pipeline_params_from_frozenset()`: Convert frozenset back to parameter dictionary

5. **Dual Approach Success**: Both string-based (`"sklearn_pipeline_scaling__income"`) and configuration-based features working with proper parameter extraction and mutual exclusivity validation.

6. **Configuration-Based Complexity**: Successfully handled the most complex configuration-based feature group to date, with multiple parameter types, mutual exclusivity, and complex data structure conversions.

### ✅ TimeWindow Feature Group - COMPLETED

**Key Learnings and Implementation Notes:**

1. **Dual Approach Success**: Both string-based (`"avg_3_day_window__temperature"`) and configuration-based features working seamlessly
2. **Parameter Classification**: All TimeWindow parameters classified as context parameters:
   - `WINDOW_FUNCTION`: Window operation type (sum, avg, max, etc.)
   - `WINDOW_SIZE`: Size of time window (positive integer)
   - `TIME_UNIT`: Time unit (day, hour, minute, etc.)
   - `mloda_source_feature`: Source feature for window operation

3. **Unified Parser Integration**: Successfully replaced old validation with `match_configuration_feature_chain_parser`

4. **PyArrow Duplicate Column Issue**: Discovered and fixed PyArrow-specific issue where duplicate column creation failed:
   - **Problem**: PyArrow's `append_column()` fails if column already exists, while Pandas overwrites gracefully
   - **Solution**: Added duplicate column detection in `_add_result_to_data()` method:
     ```python
     if feature_name in data.schema.names:
         # Remove existing column and add new one
         column_index = data.schema.names.index(feature_name)
         data = data.remove_column(column_index)
         return data.append_column(feature_name, result)
     ```
   - **Impact**: Fixed feature chaining scenarios where same feature is both explicit and dependency

5. **Cross-Framework Compatibility**: Ensured consistent behavior between Pandas and PyArrow implementations

6. **Complex Feature Chaining**: Successfully handles multi-step feature chains like:
   ```
   price → mean_imputed__price → sum_7_day_window__mean_imputed__price → max_aggr__sum_7_day_window__mean_imputed__price
   ```

7. **Modernization Pattern Established**: TimeWindow modernization follows the established pattern used by all other modernized feature groups, completing the full modernization suite.
