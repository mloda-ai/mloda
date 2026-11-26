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

When implementing a feature group that returns multiple columns, build the names by hand or use `apply_naming_convention()`:

```py
class MultiColumnFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = str(features.get_name_of_one_feature())
        
        # Return multiple columns with the naming convention
        return {
            f"{feature_name}~mean": [1, 2, 3],
            f"{feature_name}~max": [4, 5, 6],
            f"{feature_name}~min": [0, 1, 2]
        }
```

With `apply_naming_convention()`, a 2D array becomes `~N` suffixed columns:

```py
class MultiColumnProducer(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result = encoder.transform(data)  # 2D numpy array (n_samples, n_features)

        feature_name = str(features.get_name_of_one_feature())
        return cls.apply_naming_convention(result, feature_name)
        # {"category__onehot_encoded~0": ..., "category__onehot_encoded~1": ..., ...}
```

### Consuming Multiple Columns

#### Automatic Column Discovery (Recommended)

Use the `resolve_multi_column_feature()` utility to automatically discover all columns matching the pattern:

```py
class MultiColumnConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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

        feature_name = str(features.get_name_of_one_feature())
        return {feature_name: result}
```

#### Specific Sub-Column Dependency

You can declare a dependency on a specific sub-column directly, without needing all columns:

```py
class SpecificSubColumnConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        # Depend on ONLY base_feature~1, not all columns
        return {Feature("base_feature~1")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Only base_feature~1 is available
        values = data["base_feature~1"]

        feature_name = str(features.get_name_of_one_feature())
        return {feature_name: values * 10}
```

**Behavior Summary**:

- `Feature("base_feature")` → Returns ALL columns: `base_feature~0`, `base_feature~1`, etc.
- `Feature("base_feature~1")` → Returns ONLY `base_feature~1` column
- `Feature("measure~value")` → Returns ONLY `measure~value`, if the group declares `value` as an output part (see below)

### Declaring Output Parts

A group that returns named parts declares them, so `Feature("measure~value")` resolves to the group that produces `measure`, at the top level and as a dependency:

```python
from typing import Any

from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import PluginCollector, mloda


class DocMeasureFeatureGroup(DataCreatorFeatureGroup):
    OUTPUT_PARTS = ("value", "unit")  # or (ANY_OUTPUT_PART,) if parts are only named at run time

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"measure"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"measure~value": [1, 2], "measure~unit": ["m", "m"]}


result = mloda.run_all(
    ["measure~value"],
    compute_frameworks=["PandasDataFrame"],
    plugin_collector=PluginCollector.enabled_feature_groups({DocMeasureFeatureGroup}),
)
assert list(result[0].columns) == ["measure~value"]
```

If the parts depend on the feature name or options, override the classmethod `output_parts(feature_name, options)` instead.

- `measure` is computed once, even if several parts are requested with the same options and data type; only the requested part columns are returned.
- An undeclared part (`measure~bogus`) fails at resolution and lists the declared parts. With `ANY_OUTPUT_PART`, a part that was not produced fails after the computation.
- A source group whose declared name prefix claims `measure~value` reads it as `measure` plus that part when it declares the part (or `ANY_OUTPUT_PART`); literal columns, fixed declared names and undeclared parts stay whole.
- A name any group matches as a whole (a literal `a~b` column or a config-based feature) is kept whole and wins over another group's part reading; a base name that itself contains `~` cannot have parts.
- An artifact saved by a part request covers only the requested parts; a later whole-feature request cannot reuse it.
- A declared part matches only in full: declare `0` and `0~lower` to resolve both `f~0` and `f~0~lower`, or override `output_parts` for variable-width nested parts. A request for `f~0` still returns its nested columns.
- Groups that declare nothing still resolve `vec~1`, which now returns only `vec~1`, and a global filter on `vec~1` filters by that column; a missing index fails after the computation.
- `supports_compute_framework` and `Implementation.accepts` take a keyword-only `part`, the requested declared output part, so a part can run on a framework that refuses the whole.
- A whole and a part request fold into one step when one framework allows both. Otherwise (pins, or a consumer whose framework rule excludes the whole's framework) each runs as its own step computing the base, so the producer should honor `parts_only`.
- A trailing `~N` is also read against the declared parts: `measure~0` fails at resolution unless `0` is declared.
- A part's declared data type is checked on its own column (`measure~value`).

## How It Works

The mloda framework automatically handles the selection of columns that follow this naming convention:

1. When a feature group requests a feature by name, the framework identifies all columns that either:
   - Match the feature name exactly, or
   - Follow the pattern `feature_name~suffix`

2. `ComputeFramework.identify_naming_convention` selects them.

It collects every column that equals a requested feature name or starts with `feature_name~`, and raises a `ValueError` if no column matches. By default it returns a list sorted by name; with `ordering="alphabetical"` it returns a sorted list, and with `ordering="request_order"` it returns a list ordered by `request_feature_order` (falling back to `selected_feature_names`). Any other `ordering` value raises a `ValueError`.

## Available Utilities

mloda provides several utilities for working with multi-column features:

| Method | Purpose | Use Case |
|--------|---------|----------|
| `apply_naming_convention(result, feature_name)` | Create multi-column outputs | Producer: Generate `~N` suffixed columns from arrays |
| `resolve_multi_column_feature(feature_name, columns)` | Discover multi-column inputs | Consumer: Auto-find all `~N` columns |
| `expand_feature_columns(feature_name, num_columns)` | Generate column name list | Producer: Pre-generate expected column names |
| `get_column_base_feature(column_name)` | Strip trailing `~N` suffix from column | Both: Extract base feature from `feature~N` (last `~` wins) |

**Note**: `Feature("base_feature~N")` and `Feature("base_feature~part")` resolve to the group that produces `base_feature` and extract only that sub-column. A numeric suffix works for any group that supports the base feature; a non-digit suffix such as `~mean` needs a group that declares it as an output part. `get_column_base_feature` strips only a trailing numeric suffix.
