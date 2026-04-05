## Filter

In the data and machine learning world, a filter is a technique used to narrow down or preprocess data by removing irrelevant or unwanted information based on specific conditions, improving the quality and relevance of the dataset for analysis or modeling.

That however means that a data project typically contains multiple filters. For this, this project uses the **GlobalFilter** as a container for multiple **SingleFilters**. 

#### SingleFilter

-    filter_feature: It can be a Feature or a feature name as string.
-    parameter: A dictionary of parameters to filter. Example: {"min": 2, "max": 3}
-    filter_type: It can be a str or a FilterType.

#### FilterType

This class is supposed to created similarity in the framework and by framework users.

``` python
class FilterType(Enum):
    MIN = "min"
    MAX = "max"
    EQUAL = "equal"
    RANGE = "range"
    REGEX = "regex"
    CATEGORICAL_INCLUSION = "categorical_inclusion"
```

#### GlobalFilter

The GlobalFilter provides methods to add filters to the collection. The prefered way to use the GlobalFilter is by using the following functions.

**add_filter**: Adds a single filter to the GlobalFilter object.

-   filter_feature
-   filter_type
-   parameter

**add_time_and_time_travel_filters**: Adds time and time travel filtering to the GlobalFiltering. This is a convenience method. Due to the complexity of time in data/ml/ai projects, this function should be used.

This method is useful for **filtering data based on time ranges** (event) and **validity periods** (valid).

**Event Time Filter**: For historical data (e.g., checking if a customer had a valid contract at the event time), only the event time filter is needed.
    
**Time Travel Filter**: If prior actions (e.g., payments made before the event) are relevant, the time travel filter is required.

Typically, **valid_to matches the event timestamp**, but in cases like payment plans, where payments occur after creation, some payments may be excluded based on the valid_to data.

Parameters:

-   event_from (datetime): Start of the time range (with timezone).
-   event_to (datetime): End of the time range (with timezone).
-   valid_from (Optional[datetime]): Start of the validity period (optional, with timezone).
-   valid_to (Optional[datetime]): End of the validity period (optional, with timezone).
-   max_exclusive (bool): If True, the upper bounds (event_to, valid_to) are treated as exclusive.
-   event_time_column: The column name containing event timestamps. Default is "reference_time".
-   validity_time_column: The column name containing validity timestamps. Default is "time_travel".

The **single_filters** created will be converted to UTC as ISO 8601 formatted strings to ensure consistency
    across time zones and avoid ambiguity when comparing or processing time-based data.

#### How to create a collection of single filters (GlobalFilter)

In this example, we simply instantiate a GlobalFilter and add a SingleFilter.

```python
from mloda.user import GlobalFilter

global_filter = GlobalFilter()
global_filter.add_filter("example_order_id", "equal", {"value": 1})

global_filter.filters
```

Result

``` python
{<SingleFilter(feature_name=example_order_id, type=equal, parameters=(('value', 1),))>}
```

#### How to deal with time filters

In this example, we show how one can manage datetime relations.

```python
from datetime import datetime, timezone

event_from = datetime(2023, 1, 1, tzinfo=timezone.utc)
event_to = datetime(2023, 12, 31, tzinfo=timezone.utc)
valid_from = datetime(2022, 1, 1, tzinfo=timezone.utc)
valid_to = datetime(2022, 12, 31, tzinfo=timezone.utc)

global_filter.add_time_and_time_travel_filters(event_from, event_to, valid_from, valid_to)

global_filter.filters
```

Result

``` python
{<SingleFilter(feature_name=time_travel, type=range, parameters=(('max', '2022-12-31T00:00:00+00:00'), ('max_exclusive', True), ('min', '2022-01-01T00:00:00+00:00')))>,
 <SingleFilter(feature_name=reference_time, type=range, parameters=(('max', '2023-12-31T00:00:00+00:00'), ('max_exclusive', True), ('min', '2023-01-01T00:00:00+00:00')))>}
```


#### Example access to Filters in the Feature Group

Now, we need to also use a FeatureGroup which supports it. In this example, we show where we can access the SingleFilters.
The implementation of the concrete filters is dependent on the feature group. This example is rather complex as we filter a python dictionary. 

Further, the feature is a data creator, so we create the data here itself. 

``` python
from mloda.user import mloda
from mloda.provider import FeatureGroup, FeatureSet, ComputeFramework, BaseInputData, DataCreator
from typing import Any, Union, Set, Type, Optional
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

class ExampleOrderFilter(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name(), "example_order_id"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        _data_creator = {cls.get_class_name(): [1, 2, 3],
                         "example_order_id": [2, 1, 1]
                         }
        # The following algorithm is naive and rather should show an example than a normal use case.
        # The filter implementation highly depends on the feature group!
        # Extract the filter value and filter_name information from the filters.
        for filter in features.filters:
            filter_value = filter.parameter.value
            filter_name = filter.filter_feature.name
            break
        # Create the order_id filter
        order_id_filter = [i for i, order_id in enumerate(_data_creator[filter_name]) if order_id == filter_value]
        # Apply the filter
        filtered_data = {
            key: [value[i] for i in order_id_filter]
            for key, value in _data_creator.items()
            if key != filter_name
        }
        return filtered_data
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

result = mloda.run_all(
    ["ExampleOrderFilter"],
    global_filter=global_filter
)
result[0]
```
Expected Output

```python
ExampleOrderFilter: [[2,3]]
```

Although this example is complex, it is noteworthy, that the framework considers filters as features and setup as that in the framework. 

### Summary

This filtering system improves data preprocessing through GlobalFilter and SingleFilters, allowing flexible, condition-based refinement, including time-based filtering. It maintains consistency using FilterType and supports complex machine learning use cases. If you encounter a commonly used filter not yet included, feel free to open an issue or submit a pull request.

---

## Handling Filters in a FeatureGroup

The previous sections show how **users** create filters. This section explains how
**FeatureGroup authors** work with those filters during calculation.

### How filters reach your FeatureGroup

When a user passes a `GlobalFilter`, the framework matches each `SingleFilter` to the
FeatureGroups that declare the filter's column as an input. Matched filters are attached
to the `FeatureSet` before `calculate_feature()` is called. Inside your calculation you
can access them via `features.filters`:

```
@classmethod
def calculate_feature(cls, data, features: FeatureSet):
    if features.filters is not None:
        for single_filter in features.filters:
            column = single_filter.name             # e.g. "status"
            value  = single_filter.parameter.value  # e.g. "active"
            # ... use however you need
```

`features.filters` is **always available**, regardless of any other setting.

### Two independent concerns

Filters involve two decisions that are **independent** of each other:

| Concern | Who decides | When it happens |
|---------|------------|-----------------|
| **Inline reading** -- should the FeatureGroup read `features.filters` during calculation? | The FeatureGroup author (you) | During `calculate_feature()` |
| **Row elimination** -- should the framework remove non-matching rows after calculation? | `final_filters()` return value | After `calculate_feature()` |

A FeatureGroup can do either, both, or neither. The two concerns are decoupled.

### `final_filters()` reference

Override this classmethod on your FeatureGroup to control post-calculation row elimination:

```
@classmethod
def final_filters(cls) -> bool | None:
    return None  # default
```

| Return value | Meaning |
|:------------:|---------|
| `None` | Defer to the FilterEngine tied to the ComputeFramework. Most engines (Pandas, PyArrow, Polars, Spark) default to `True` (eliminate rows). Iceberg defaults to `False` (predicate pushdown handles it). |
| `False` | Skip row elimination. Use this when your FeatureGroup fully handles the filter itself. |
| `True` | Force row elimination, even if the FilterEngine would skip it. |

This method does **not** affect whether `features.filters` is populated. Filters are
always available for inline reading.

### Usage patterns

The examples below use pseudocode helpers (`compute_totals`, `build_mask_from_filters`,
`broadcast_sum`) to keep the focus on the filter interaction pattern. Replace these with
your framework-specific logic.

#### Pattern 1: Let the framework handle everything (default)

The most common case. Your FeatureGroup ignores filters entirely, and the framework
removes non-matching rows after calculation. No override needed.

```
class SalesTotal(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data, features: FeatureSet):
        # Just compute; framework handles filtering afterward
        return pa.table({cls.get_class_name(): compute_totals(data)})
```

#### Pattern 2: Inline masking, skip row elimination

Read filters during calculation to conditionally mask values (e.g. null-out non-matching
rows before aggregation), then tell the framework not to remove rows. Useful when
downstream consumers need all rows, but only matching values should contribute to
aggregated results.

Example: "Sum of active sales per region, broadcast back to every row."

```
class MaskedRegionSum(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool:
        return False  # skip row elimination

    @classmethod
    def calculate_feature(cls, data, features: FeatureSet):
        mask = build_mask_from_filters(data, features.filters)
        masked = pc.if_else(mask, data["sales"], None)
        # aggregate masked values, broadcast back to all rows
        return pa.table({cls.get_class_name(): broadcast_sum(masked, data["region"])})
```

Result: all rows preserved, but only matching values contributed to the sum.

#### Pattern 3: Inline masking + row elimination

Read filters for conditional logic **and** request row elimination afterward. Useful when
you need filter-aware computation (masking, weighting, branching) but also want
non-matching rows removed from the final output.

Example: "Sum of active sales per region, only for active rows."

```
class MaskedRegionSumActiveOnly(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool:
        return True  # also eliminate non-matching rows

    @classmethod
    def calculate_feature(cls, data, features: FeatureSet):
        mask = build_mask_from_filters(data, features.filters)
        masked = pc.if_else(mask, data["sales"], None)
        sums = broadcast_sum(masked, data["region"])
        # Return all rows; the framework will remove non-matching ones
        return pa.table({
            cls.get_class_name(): sums,
            "status": data["status"],  # preserve the filter column
        })
```

Result: only matching rows remain, with aggregated values computed from masked data.

#### Pattern 4: Force elimination on a non-eliminating engine

Some engines skip row elimination by default (e.g. Iceberg, which uses predicate pushdown
at scan time). If your FeatureGroup computes derived columns that the scan could not
filter, override `final_filters()` to force elimination:

```
class DerivedIcebergFeature(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool:
        return True  # override Iceberg's default of False
```

### The overlap contract

When a FeatureGroup reads filters inline **and** returns `final_filters() = True`, the
same filters are processed twice: once by your code during calculation, and once by the
framework's FilterEngine afterward.

This is safe as long as you follow one rule: **preserve the filter column with its
original values in your output**.

The framework's row elimination works by matching against the filter column in your
returned data. If your FeatureGroup drops that column, the framework raises a
`ValueError` with a clear message naming the missing column. If your FeatureGroup
modifies the column values, the filter may produce wrong results silently.

```
# Correct: filter column preserved with original values
return pa.table({
    cls.get_class_name(): computed_values,
    "status": original_status_column,   # framework can filter on this
})

# Wrong: filter column modified
return pa.table({
    cls.get_class_name(): computed_values,
    "status": [1, 0, 1, 0],   # mapped to ints; "equal" filter for "active" matches nothing
})

# Wrong: filter column omitted -- raises ValueError at runtime
return pa.table({
    cls.get_class_name(): computed_values,
    # "status" missing; framework raises: "missing filter column 'status'"
})
```

#### Quick reference

| Your FeatureGroup... | `final_filters()` | Reads `features.filters`? | Must preserve filter column? |
|---------------------|:-----------------:|:------------------------:|:---------------------------:|
| Ignores filters | `None` (default) | No | Yes (it is in `input_data`) |
| Handles everything inline | `False` | Yes | No (elimination skipped) |
| Uses inline logic + elimination | `True` | Yes | **Yes** |
| Just forces elimination | `True` | No | Yes |
