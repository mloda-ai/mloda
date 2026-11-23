# Comparison Contract

Operations that assume ordered or temporal semantics (as-of joins, equi-joins, range/min/max
filters, time-window and forecasting reference columns) validate the columns they touch at the
backend boundary through a shared **comparison contract**. This turns a class of silent-wrongness
bugs (for example joining or comparing a timezone-aware column against a timezone-naive one) into a
clear error instead of a cryptic backend failure or a quietly incorrect result.

The contract decides whether columns may be compared. What a zone means for a computation on a
timezone-aware column (which clock applies, daylight saving lengths, the zone of the result) is the
[time zone computation rule](timezone-computation.md).

## The pieces

- `ComparisonContract` declares the semantics an operation requires: a set of
  `SemanticDimension` values (`ORDERED`, `TEMPORAL`, `NUMERIC`), an optional `unit`, and a
  `tz_policy`.
- `ColumnSemantics` is the framework-neutral view of one column: `is_ordered`, `is_temporal`,
  `is_numeric`, `unit`, `is_tz_aware`, `tz`. `tz` is the zone name; giving it makes the column aware,
  and `is_tz_aware=True` with `tz=None` means aware in an unknown zone. Each merge and filter engine (`BaseMergeEngine` /
  `BaseFilterEngine` subclass) derives it from its framework's native schema via the
  `_column_semantics(data, column)` hook.
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

What each backend can determine from its native schema. `column_semantics` is an opt-in merge- and
filter-engine hook: an engine enables the timezone/unit guard by setting `provides_column_semantics =
True` and implementing the hook. Engines that leave the flag `False` (for example a time-agnostic
custom ComputeFramework) skip the guard entirely and are never forced to implement it. An engine that
opts in but forgets to implement the hook raises a clear error rather than silently skipping
validation. All built-in backends below opt in.

| Backend | ordered / temporal / numeric | timezone-awareness | zone (`tz`) | time unit |
|---------|:---:|:---:|:---:|:---:|
| pandas | yes | yes | native | yes |
| polars | yes | yes | native | yes |
| pyarrow | yes | yes | native | yes |
| duckdb | yes | yes (`TIMESTAMP WITH TIME ZONE`) | recorded zone, else unknown | partial |
| sqlite | yes | only for native timestamp columns | native (Arrow type) | only for native timestamp columns |
| spark | yes | only with a recorded zone | recorded zone from the transfer, else not reported | no |
| python_dict | yes (value scan) | yes (value scan) | tzinfo of the first value | no |

## Deferred

- **Value inspection on dynamically typed storage.** sqlite stores datetimes as ISO-8601 text, so
  timezone and unit are not visible from its schema; these are read only when the column is a native
  temporal type. A value-scanning fallback is intentionally not implemented yet.
- **Unit enforcement.** The time unit is derived and carried on `ColumnSemantics`, but it is not
  enforced across sides: differing resolutions (for example nanoseconds vs microseconds) are aligned
  natively by the backends, so `require_compatible` does not reject them. A unit is only checked when
  a `ComparisonContract.unit` is explicitly declared (the single-column `validate` path).
