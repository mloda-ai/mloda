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
  `is_numeric`, `unit`, `is_tz_aware`, `tz`. Each merge and filter engine (`BaseMergeEngine` /
  `BaseFilterEngine` subclass) derives it from its framework's native schema via the
  `_column_semantics(data, column)` hook.
- `ComparisonContract.validate(semantics, column)` checks a single column against the required
  dimensions. `require_compatible(left, right, ...)` checks two columns are comparable, using a
  **strict timezone model**: mixing timezone-aware and timezone-naive temporal columns is an error,
  while two aware columns remain comparable even when their zones differ.

`tz` carries the native zone label, such as `Europe/Berlin`, `UTC`, or a fixed-offset label.
Supplying a known zone implies `is_tz_aware=True`. `tz=None` means no zone identity is known;
an aware backend can still report `is_tz_aware=True` with `tz=None`. The existing fifth positional
`is_tz_aware` argument and its keyword form retain that behavior. Check awareness separately from
zone identity: unknown does not mean naive.

Labels retain the backend's spelling; aliases and fixed-offset labels are not normalized across
backends. Python datetimes and parsed ISO strings use `str(tzinfo)`, for example `UTC+05:30`;
Arrow retains its schema label, which may instead be `+05:30`. Callers can inspect known labels to
check zone preservation, but the comparison contract does not require equal zone labels.

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

What each backend can determine from its native schema or existing value-inspection path.
`column_semantics` is an opt-in merge- and
filter-engine hook: an engine enables the timezone/unit guard by setting `provides_column_semantics =
True` and implementing the hook. Engines that leave the flag `False` (for example a time-agnostic
custom ComputeFramework) skip the guard entirely and are never forced to implement it. An engine that
opts in but forgets to implement the hook raises a clear error rather than silently skipping
validation. All built-in backends below opt in.

| Backend | ordered / temporal / numeric | timezone-awareness | zone (`tz`) | time unit |
|---------|:---:|:---:|:---:|:---:|
| pandas | yes | yes | `DatetimeTZDtype.tz` or Arrow timestamp schema | yes |
| polars | yes | yes | `Datetime.time_zone` for eager and lazy frames | yes |
| pyarrow | yes | yes | timestamp schema's `tz` | yes |
| duckdb | yes | yes (`TIMESTAMP WITH TIME ZONE`) | recorded `DuckdbRelation.time_zone(column)`; otherwise unknown | partial |
| sqlite | yes | native timestamps or ISO text inspection | native timestamp zone or a uniform sampled ISO offset | only for native timestamp columns |
| spark | yes | no (deferred) | unknown | no |
| python_dict | yes (value scan) | yes (value scan) | common native datetime zone or a uniform sampled ISO offset | no |

Typed timestamp columns retain their schema zone even when empty or all-null. Bare DuckDB
`TIMESTAMPTZ` and wrappers without recorded source metadata report aware with an unknown zone;
the session timezone does not establish the original column's zone. Reading recorded wrapper
metadata does not execute or convert the relation. Spark keeps its existing deferred-awareness
behavior and reports no zone.

Python dictionaries retain a native datetime zone only when all non-null values agree on its label,
including across daylight-saving offsets. ISO text inspection uses its existing bounded sample
(100 non-null values by default). Mixed aware zones or offsets retain awareness but have no single
known zone; an all-null untyped sample cannot establish one. ISO offsets cannot recover an IANA zone.

## Deferred

- **Unit enforcement.** The time unit is derived and carried on `ColumnSemantics`, but it is not
  enforced across sides: differing resolutions (for example nanoseconds vs microseconds) are aligned
  natively by the backends, so `require_compatible` does not reject them. A unit is only checked when
  a `ComparisonContract.unit` is explicitly declared (the single-column `validate` path).
