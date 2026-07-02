# Comparison Contract

Operations that assume ordered or temporal semantics (as-of joins, equi-joins, range/min/max
filters, time-window and forecasting reference columns) validate the columns they touch at the
backend boundary through a shared **comparison contract**. This turns a class of silent-wrongness
bugs (for example joining or comparing a timezone-aware column against a timezone-naive one) into a
clear error instead of a cryptic backend failure or a quietly incorrect result.

## The pieces

- `ComparisonContract` declares the semantics an operation requires: a set of
  `SemanticDimension` values (`ORDERED`, `TEMPORAL`, `NUMERIC`), an optional `unit`, and a
  `tz_policy`.
- `ColumnSemantics` is the framework-neutral view of one column: `is_ordered`, `is_temporal`,
  `is_numeric`, `unit`, `is_tz_aware`. Each compute framework derives it from its own native
  schema (`column_semantics(...)`).
- `ComparisonContract.validate(semantics, column)` checks a single column against the required
  dimensions. `require_compatible(left, right, ...)` checks two columns are comparable, using a
  **strict timezone model**: mixing timezone-aware and timezone-naive temporal columns is an error,
  and known time units must match.

## Where it is applied

| Operation | Check |
|-----------|-------|
| As-of joins | time columns must be ordered (opt-in coercion otherwise); the two time columns must be timezone-compatible |
| Equi-joins (inner/left/right/outer) | when **both** join keys are temporal, they must be timezone-compatible; string / numeric / id keys are never affected |
| Range / min / max filters | when a bound is a native `datetime`, its timezone-awareness must match the filtered column |
| Time-window / forecasting | the reference time column must be temporal and ordered |

The timezone check only fires when both sides are genuinely temporal, so ordinary non-temporal
joins and filters are never rejected.

## Per-engine capability

What each backend can determine from its native schema. `column_semantics` is a required merge- and
filter-engine hook; an engine that does not implement it raises a clear error rather than silently
skipping validation.

| Backend | ordered / temporal / numeric | timezone-awareness | time unit |
|---------|:---:|:---:|:---:|
| pandas | yes | yes | yes |
| polars | yes | yes | yes |
| pyarrow | yes | yes | yes |
| duckdb | yes | yes (`TIMESTAMP WITH TIME ZONE`) | partial |
| sqlite | yes | only for native timestamp columns | only for native timestamp columns |
| spark | yes | no (deferred) | no |
| python_dict | yes (value scan) | yes (value scan) | no |

## Deferred

- **Value inspection on dynamically typed storage.** sqlite stores datetimes as ISO-8601 text, so
  timezone and unit are not visible from its schema; these are read only when the column is a native
  temporal type. A value-scanning fallback is intentionally not implemented yet.
- **Unit enforcement.** The time unit is derived and carried on `ColumnSemantics`, but it is not
  enforced across sides: differing resolutions (for example nanoseconds vs microseconds) are aligned
  natively by the backends, so `require_compatible` does not reject them. A unit is only checked when
  a `ComparisonContract.unit` is explicitly declared (the single-column `validate` path).
