# Design Decision: Merge DefaultColumnNames into DefaultOptionKeys

## Date
2025-12-10

## Context
During the implementation of "Clarify Time Concepts Design" (Item 11), a separate `DefaultColumnNames` class was created to hold default column name values, separating them from option keys in `DefaultOptionKeys`.

## Initial Design
```python
# DefaultOptionKeys - option keys for lookup
class DefaultOptionKeys(str, Enum):
    reference_time = "reference_time"
    ...

# DefaultColumnNames - default column values (separate class)
class DefaultColumnNames:
    REFERENCE_TIME = "reference_time"
    TIME_TRAVEL = "time_travel_filter"
```

## Problem
The separation was over-engineering:
- `DefaultOptionKeys.reference_time.value` == `DefaultColumnNames.REFERENCE_TIME` == `"reference_time"`
- Two classes for effectively the same information
- Increased cognitive load for developers
- More files to maintain

## Decision
Merge `DefaultColumnNames` into `DefaultOptionKeys`:

```python
class DefaultOptionKeys(str, Enum):
    """
    Default option keys used to configure mloda feature groups.
    The enum value serves as both the option key and the default column name.

    Time-Related Keys:
    - `reference_time`: Key for the event timestamp column. Value: "reference_time"
    - `time_travel`: Key for the validity timestamp column. Value: "time_travel_filter"
    """
    reference_time = "reference_time"
    time_travel = "time_travel_filter"
    ...
```

## Benefits
1. **Simpler API**: One class instead of two
2. **Less code**: Removed ~20 lines and one file
3. **Clearer intent**: Docstring explains dual purpose
4. **Easier maintenance**: Single source of truth

## Files Changed
- Modified: `default_options_key.py` (added `time_travel`, updated docstring)
- Modified: `time_window/base.py`, `forecasting/base.py`, `global_filter.py`
- Modified: Test files to use `DefaultOptionKeys.reference_time.value`
- Deleted: `default_column_names.py`, `test_default_column_names.py`
- Created: `test_default_options_key.py`

## Usage After Change
```python
# Get default column name
default_time_column = DefaultOptionKeys.reference_time.value  # "reference_time"
default_validity_column = DefaultOptionKeys.time_travel.value  # "time_travel_filter"

# Use as option key
options.get(DefaultOptionKeys.reference_time.value)
```
