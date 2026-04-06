## Filter Pipeline Analysis

This document explains how filters flow through multi-step feature pipelines in mloda,
with concrete data-flow examples for each framework category.

For the user-facing filter API and FeatureGroup authoring patterns, see
[filter_data.md](filter_data.md).


### How filters flow through multi-step pipelines

When a user passes a `GlobalFilter` to `mloda.run_all()`, the framework distributes
filters to FeatureGroups in three stages:

1. **Matching.** For every FeatureGroup in the execution plan, `GlobalFilter.identify_matched_filters()`
   checks each `SingleFilter` against the FeatureGroup's criteria (feature name resolution,
   domain compatibility, compute framework). Only filters whose column is declared in the
   FeatureGroup's `input_data` are matched.

2. **Deep-copy isolation.** Each matched filter is **deep-copied** before it is enriched
   with options, domain, and compute framework from the parent feature. The original
   `SingleFilter` objects in the `GlobalFilter` are never mutated. If two FeatureGroups
   both match the same original filter, they each receive independent copies.

3. **Per-FeatureGroup lifecycle.** Each FeatureGroup runs its own filter lifecycle
   independently:

   ```
   set_filter_engine  ->  calculate_feature()  ->  run_final_filter()
   ```

   The `FeatureSet` attached to each FeatureGroup carries its own set of `SingleFilter`
   instances and its own `FilterEngine` reference. No state is shared between steps.

This means a single `GlobalFilter` with one filter (e.g., `status == "active"`) can be
processed differently by different FeatureGroups in the same pipeline: one may use it
for inline masking, another for row elimination, and a third may ignore it entirely.


### Concrete pipeline flows

All flows use the same input data and filter:

```
Input:
    region | status   | value
    A      | active   |  10
    A      | inactive |  20
    B      | active   |  30
    B      | inactive |  40

Filter: status == "active"
```


#### Flow 1: Eager framework + final filters (Pandas, PyArrow)

The FeatureGroup computes its result without reading filters. After `calculate_feature()`
returns, the framework's `run_final_filter()` applies the filter in memory using boolean
masking (`pc.equal` + `table.filter` for PyArrow, boolean indexing for Pandas).

| Stage | Data |
|-------|------|
| `calculate_feature()` output | `[10, 20, 30, 40]` with `status = [active, inactive, active, inactive]` |
| `run_final_filter()` | Creates mask `[True, False, True, False]`, applies `table.filter(mask)` |
| **Final result** | `[10, 30]` with `status = [active, active]` (2 rows) |

Physical behavior: row elimination happens in memory after the full table is materialized.
The filter engine (`PyArrowFilterEngine`, `PandasFilterEngine`) returns `final_filters() = True`.


#### Flow 2: Lazy framework + final filters (DuckDB, SQLite, Polars, Spark)

The FeatureGroup returns a lazy relation (DuckDB) or lazy frame (Polars/Spark).
`run_final_filter()` calls the relation's `.filter()` method, which appends a
WHERE clause (SQL) or filter node (lazy dataframe) to the query plan. No data moves
until materialization.

| Stage | Data |
|-------|------|
| `calculate_feature()` output | Lazy relation: `SELECT value, status FROM source` |
| `run_final_filter()` | Calls `relation.filter('"status" = ?', ('active',))` |
| Query plan after filter | `SELECT value, status FROM source WHERE "status" = 'active'` |
| **Materialized result** | `[10, 30]` with `status = [active, active]` (2 rows) |

Physical behavior: the optimizer pushes the filter to scan time. The result is identical
to eager row elimination, but no intermediate full-table materialization occurs. The filter
engine (`DuckDBFilterEngine`, `SqliteFilterEngine`, `SparkFilterEngine`, `PolarsFilterEngine`)
returns `final_filters() = True`.


#### Flow 3: Inline masking (conditional aggregation, all rows preserved)

The FeatureGroup reads `features.filters` during `calculate_feature()` and uses the
filter condition to **mask** values (replace non-matching with NULL) rather than
eliminate rows. It returns `final_filters() = False` to prevent post-calculation
row elimination.

| Stage | Data |
|-------|------|
| Read `features.filters` | Extract: `status == "active"` |
| Build mask | `[True, False, True, False]` |
| Apply mask to value column | `masked_value = [10, NULL, 30, NULL]` |
| Aggregate (sum by region, broadcast) | Region A: 10, Region B: 30 |
| **Final result** | `[10, 10, 30, 30]` (4 rows, all preserved) |

Physical behavior: the FeatureGroup handles the filter entirely within its own
calculation logic. The framework skips `run_final_filter()` because
`final_filters() = False`. This pattern is used when downstream consumers need
all rows but only matching values should contribute to aggregated results.


#### Flow 4: Masking AND elimination (same filter, two FeatureGroups)

A real pipeline may need the same filter for two purposes in separate steps:

- **Step 1** (inline masking): A FeatureGroup masks non-matching values, aggregates,
  and preserves all rows.
- **Step 2** (final elimination): A different FeatureGroup uses the framework's
  default row elimination to remove non-matching rows from the final output.

Because `identify_matched_filters()` deep-copies filters independently for each
FeatureGroup, both steps receive the filter and process it through their own lifecycle.

| Stage | Step 1 (inline mask FG) | Step 2 (regular FG) |
|-------|------------------------|-------------------|
| `final_filters()` | `False` | `None` (engine default: `True`) |
| `calculate_feature()` | Reads `features.filters`, masks values, aggregates | Computes raw values, ignores filters |
| `run_final_filter()` | Skipped | Applies row elimination |
| **Result** | `[10, 10, 30, 30]` (4 rows) | `[10, 30]` (2 rows) |

Both steps run within the same `run_all()` call. Each has its own filter copies, its own
filter engine instance, and its own `run_final_filter()` decision. No coordination is
needed between them.

The test `test_mixed_final_and_inline_filters_in_same_run` in
`tests/test_core/test_filter/test_feature_group_final_filters.py` validates this exact
scenario.


### Semantic vs. physical: what `final_filters` actually means

`final_filters()` is a **semantic** flag that controls *when in the pipeline* the filter
is applied. It is not a physical flag about *where* the computation runs.

| Framework category | Examples | `final_filters()` | Physical behavior |
|-------------------|----------|:------------------:|-------------------|
| Eager | Pandas, PyArrow | `True` | Post-hoc filter in memory |
| Lazy (SQL) | DuckDB, SQLite | `True` | `.filter()` adds WHERE to query plan |
| Lazy (dataframe) | Polars, Spark | `True` | `.filter()` adds node to lazy plan |
| Scan-time | Iceberg | `False` | Predicates pushed into scan expressions |

All frameworks that return `True` produce the same logical result (non-matching rows are
absent from the output), but the physical execution differs:

- **Eager frameworks** materialize the full table, then remove rows in a second pass.
- **Lazy SQL frameworks** append a WHERE clause. The database optimizer may push it to
  scan time, but mloda treats it as "final" because the filter is logically applied
  after `calculate_feature()` returns.
- **Lazy dataframe frameworks** add a filter node to the execution plan. The framework
  optimizer decides the physical execution order at materialization time.

Iceberg is the only built-in engine that returns `False`. It translates filters to
PyIceberg expressions (`EqualTo`, `GreaterThanOrEqual`, etc.) and pushes them into
`table.scan(row_filter=...)` at read time. The FeatureGroup never sees unfiltered data.

A FeatureGroup can **override** the engine's default in either direction:

- Return `True` on a FeatureGroup using Iceberg to force post-scan row elimination
  (e.g., for derived columns that the scan predicate cannot express).
- Return `False` on a FeatureGroup using PyArrow to skip row elimination and handle
  the filter inline (e.g., for conditional masking).


### Implications for filtered aggregation plugins

The filter infrastructure already provides the building blocks for conditional
aggregation (the SQL equivalent of `SUM(CASE WHEN condition THEN value END)`):

1. **`features.filters` is always available** inside `calculate_feature()`, regardless
   of the `final_filters()` setting.

2. **Inline masking** (`pc.if_else(mask, value, None)` + aggregation) reproduces
   `CASE WHEN` semantics without a separate code path.

3. **Independent filter lifecycles** mean a conditional aggregation FeatureGroup and a
   row-eliminating FeatureGroup can coexist in the same pipeline without interference.

4. **No new infrastructure is needed.** A filtered aggregation plugin only needs to:
   - Declare the filter column in its `input_data` criteria so the filter is matched.
   - Read `features.filters` in `calculate_feature()` to build a mask.
   - Return `final_filters() = False` to preserve all rows.

This avoids duplicating existing filter operations in a separate
`FilteredAggregationFeatureGroup` and keeps the plugin composable with the rest of
the filter system.
