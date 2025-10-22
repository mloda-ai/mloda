# Multiple Result Columns

## Overview

Feature groups in mloda can return multiple related columns using a naming convention pattern. This allows for more flexible and powerful feature engineering, especially when a single feature computation produces multiple related outputs.

## Naming Convention

The naming convention for multiple result columns follows this pattern:

```
feature_name~column_suffix
```

Where:
- `feature_name` is the base name of the feature
- `~` is the separator character
- `column_suffix` is a unique identifier for each related column

## Example

A feature group that computes statistical properties might return multiple columns:

```
temperature~mean
temperature~min
temperature~max
temperature~std
```

## Implementation

### Returning Multiple Columns

When implementing a feature group that returns multiple columns:

``` python
class MultiColumnFeatureGroup(AbstractFeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = features.get_name_of_one_feature().name
        
        # Return multiple columns with the naming convention
        return {
            f"{feature_name}~mean": [1, 2, 3],
            f"{feature_name}~max": [4, 5, 6],
            f"{feature_name}~min": [0, 1, 2]
        }
```

### Consuming Multiple Columns

#### Automatic Column Discovery (Recommended)

Use the `resolve_multi_column_feature()` utility to automatically discover all columns matching the pattern:

``` python
class MultiColumnConsumer(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.not_typed("MultiColumnFeature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Automatically discover all columns matching the pattern
        columns = cls.resolve_multi_column_feature(
            "MultiColumnFeature",
            set(data.columns)
        )
        # Returns: ["MultiColumnFeature~mean", "MultiColumnFeature~max", "MultiColumnFeature~min"]

        # Process all discovered columns
        result = sum(data[col] for col in columns)

        feature_name = features.get_name_of_one_feature().name
        return {feature_name: result}
```

**Benefits**:
- No need to manually enumerate column names
- Automatically adapts if number of columns changes
- Cleaner, more maintainable code

#### Manual Column Access (Legacy)

For backwards compatibility, you can still access columns manually:

``` python
class MultiColumnConsumer(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature.not_typed("MultiColumnFeature")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Manual access to specific columns
        mean_values = data["MultiColumnFeature~mean"]
        max_values = data["MultiColumnFeature~max"]

        # Perform calculations using these columns
        result = mean_values + max_values

        feature_name = features.get_name_of_one_feature().name
        return {feature_name: result}
```

## How It Works

The mloda framework automatically handles the selection of columns that follow this naming convention:

1. When a feature group requests a feature by name, the framework identifies all columns that either:
   - Match the feature name exactly, or
   - Follow the pattern `feature_name~suffix`

2. This is implemented in the `identify_naming_convention` method in the `ComputeFrameWork` class:

``` python
def identify_naming_convention(self, selected_feature_names: Set[FeatureName], column_names: Set[str]) -> Set[str]:
    feature_name_strings = {f.name for f in selected_feature_names}
    _selected_feature_names: Set[str] = set()

    for col in column_names:
        for feature_name in feature_name_strings:
            if col == feature_name:
                _selected_feature_names.add(col)
                continue

            if col.startswith(f"{feature_name}~"):
                _selected_feature_names.add(col)

    if not _selected_feature_names:
        raise ValueError(
            f"No columns found that match feature names {feature_name_strings} or follow the naming convention 'feature_name~column_name'"
        )

    return _selected_feature_names
```

## Best Practices

1. **Use Automatic Discovery**: Prefer `resolve_multi_column_feature()` over manual column enumeration
2. **Consistent Naming**: Use consistent suffixes across related feature groups
3. **Use `apply_naming_convention()`**: When producing multi-column outputs, use the utility for consistency
4. **Documentation**: Document the meaning of each suffix in your feature group's docstring
5. **Validation**: Consider validating that all expected columns are present when consuming multiple columns
6. **Error Handling**: Handle cases where expected columns might be missing

## Available Utilities

mloda provides several utilities for working with multi-column features:

| Method | Purpose | Use Case |
|--------|---------|----------|
| `apply_naming_convention(result, feature_name)` | Create multi-column outputs | Producer: Generate `~N` suffixed columns from arrays |
| `resolve_multi_column_feature(feature_name, columns)` | Discover multi-column inputs | Consumer: Auto-find all `~N` columns |
| `expand_feature_columns(feature_name, num_columns)` | Generate column name list | Producer: Pre-generate expected column names |
| `get_column_base_feature(column_name)` | Strip suffix from column | Both: Extract base feature from `feature~N` |
