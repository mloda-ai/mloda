## Mask Engine

The mask engine provides boolean masking primitives for use inside
`calculate_feature()`. It lets a FeatureGroup selectively include or exclude
values without removing rows from the output.

For user-provided filter propagation (GlobalFilter, SingleFilter, row elimination),
see [Filter Data](filter_data.md).

### How the mask engine is wired

Each `ComputeFramework` provides a `BaseMaskEngine` subclass via its
`mask_engine()` classmethod. The framework sets `features.mask_engine` on the
`FeatureSet` before `calculate_feature()` is called. This wiring is independent
of `features.filters` and `final_filters()`.

```
@classmethod
def calculate_feature(cls, data, features: FeatureSet):
    engine = features.mask_engine   # always available when the framework provides one
    mask = engine.equal(data, "status", "active")
    # ... apply mask
```

### Primitives

Every `BaseMaskEngine` subclass implements these abstract classmethods:

| Method | Returns mask where... |
|--------|----------------------|
| `equal(data, column, value)` | `data[column] == value` |
| `greater_equal(data, column, value)` | `data[column] >= value` |
| `greater_than(data, column, value)` | `data[column] > value` |
| `less_equal(data, column, value)` | `data[column] <= value` |
| `less_than(data, column, value)` | `data[column] < value` |
| `is_in(data, column, values)` | `data[column]` is in `values` |
| `combine(mask1, mask2)` | logical AND of two masks |
| `all_true(data)` | all `True` (no filtering) |

### Convenience methods

Built from the primitives above. No per-engine implementation needed.

#### `between(data, column, min_value, max_value, *, min_exclusive=False, max_exclusive=False)`

Range check combining a lower and upper bound:

```
# Inclusive: value in [10, 100]
mask = engine.between(data, "value", 10, 100)

# Exclusive bounds: value in (10, 100)
mask = engine.between(data, "value", 10, 100, min_exclusive=True, max_exclusive=True)

# Half-open: value in [10, 100)
mask = engine.between(data, "value", 10, 100, max_exclusive=True)
```

#### `all_of(data, masks)`

AND-combine a list of masks. Returns an all-True mask if the list is empty.

```
mask = engine.all_of(data, [
    engine.equal(data, "status", "active"),
    engine.between(data, "value", 10, 100),
])
```

### Masking patterns

#### Inline masking, skip row elimination

Set `final_filters() = False` to tell the framework not to remove rows.
Use the mask engine to selectively null out values before aggregation.

```
class MaskedRegionSum(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool:
        return False  # skip row elimination

    @classmethod
    def calculate_feature(cls, data, features: FeatureSet):
        engine = features.mask_engine
        mask = engine.equal(data, "status", "active")
        masked = pc.if_else(mask, data["sales"], None)
        # aggregate masked values, broadcast back to all rows
        return pa.table({cls.get_class_name(): broadcast_sum(masked, data["region"])})
```

Result: all rows preserved, but only matching values contributed to the sum.

#### Inline masking + row elimination

Use the mask engine for conditional logic **and** return `final_filters() = True`
so the framework also eliminates non-matching rows afterward. You must preserve
the filter column in your output (see the
[overlap contract](filter_data.md#the-overlap-contract) in the filter docs).

```
class MaskedRegionSumActiveOnly(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool:
        return True  # also eliminate non-matching rows

    @classmethod
    def calculate_feature(cls, data, features: FeatureSet):
        engine = features.mask_engine
        mask = engine.equal(data, "status", "active")
        masked = pc.if_else(mask, data["sales"], None)
        sums = broadcast_sum(masked, data["region"])
        # Return all rows; the framework will remove non-matching ones
        return pa.table({
            cls.get_class_name(): sums,
            "status": data["status"],  # preserve the filter column
        })
```

Result: only matching rows remain, with aggregated values computed from masked data.

### Pipeline data flow (inline masking)

```
Input:
    region | status   | value
    A      | active   |  10
    A      | inactive |  20
    B      | active   |  30
    B      | inactive |  40

Mask predicate: status == "active"
```

| Stage | Data |
|-------|------|
| Use `features.mask_engine` | `engine.equal(data, "status", "active")` |
| Build mask | `[True, False, True, False]` |
| Apply mask to value column | `masked_value = [10, NULL, 30, NULL]` |
| Aggregate (sum by region, broadcast) | Region A: 10, Region B: 30 |
| **Final result** | `[10, 10, 30, 30]` (4 rows, all preserved) |

The framework skips `run_final_filter()` because `final_filters() = False`.
