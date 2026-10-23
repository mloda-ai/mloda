# Feature Group Compute Framework Integration

## Overview

One of mloda's key strengths is its ability to decouple feature definitions from specific computation technologies. This document explains how feature groups integrate with different compute frameworks.

## Core Concepts

### Compute Framework Specification

Feature groups specify which compute frameworks they support through the `compute_framework_rule` method:

```py
@classmethod
def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
    """Define the compute frameworks this feature group supports."""
    return {PandasDataFrame}  # Support only Pandas
    # Or return True to support all available compute frameworks
```

### Declaring an Operation Unsupported on a Framework

`compute_framework_rule` is a static, class-level set: it cannot say "this
feature group runs on SQLite in general, but *this particular operation* is
unsupported there." For that, override `supports_compute_framework`, a
per-feature hook evaluated at match time:

```py
@classmethod
def supports_compute_framework(cls, feature_name, options, compute_framework) -> bool:
    """Reject an operation on a specific framework. Default returns True."""
    if compute_framework is SqliteFramework and is_median_op(feature_name, options):
        return False  # median is unsupported on SQLite
    return True
```

Returning `False` removes that framework from the candidate set **for this
feature only**:

- If another framework can still run the operation, the matcher routes around
  the rejected one silently (no error).
- If the only remaining candidate is the rejected framework (for example, the
  user pinned the feature to it), resolution fails with the standard
  "No feature groups found" error, which names the near-miss feature group and
  why it dropped, one line per eliminated candidate under a
  "Feature group(s) eliminated while matching '...'" block, e.g.

```
No feature groups found for feature name: 'X'.
Feature group(s) eliminated while matching 'X':
  - MedianFeatureGroup (compute framework pin): pinned compute framework 'SqliteFramework' is not among its supported ['DuckDBFramework', 'PandasDataFrame']
Use resolve_feature(name, options=...) to debug feature resolution.
For troubleshooting guide, see: https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/
```

The rejected framework is named as a near-miss with its reason rather than
vanishing into a generic "unknown feature" message. Prefer this hook over
raising a generic error from inside `calculate_feature`: the rejection happens
during planning rather than at compute time, and the message is built for you.

The debug inspector `resolve_feature(name)` reflects the hook too, because it
runs the **same matcher over the same candidate universe as the engine** (it
delegates to `IdentifyFeatureGroupClass`). Its `ResolvedFeature` result carries
`supported_compute_frameworks` and `unsupported_compute_frameworks` (evaluated
under default options unless you pass `options=`, see
[Discover Plugins](discover-plugins.md)).

Two distinct gaps both resolve to `feature_group=None` rather than appearing
runnable:

- **Unsupported on every framework.** The feature matches a group, but
  `supports_compute_framework` rejects every installed framework. Resolution
  fails with the capability error above.
- **Uninstalled framework.** The only framework a matching group declares is not
  installed (its backend library is absent, so `is_available()` is `False`). The
  candidate universe drops unavailable frameworks before matching, so the group
  maps to an empty framework set and resolution fails closed with the ordinary
  `No feature groups found for feature name: '<name>'.` error. It does **not**
  read as runnable on a framework you cannot actually run.

### Declaring capability per subtype

This section covers ordinary feature groups; operation groups declare it on their implementations (`supported`).

Families with multiple subtypes declare per-backend capability as data
instead of a hand-written `supports_compute_framework`.

**The data provider** declares the dimension once with `SUBTYPES`:

```py
class RankFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    SUBTYPES = SubtypeDeclaration(
        key="rank_type",
        parametric_families={"ntile": "N-tile bucketing"},
        supported={"PythonDictFramework": {"dense", "ordinal"}},
    )
    PREFIX_PATTERN = r".*__(?P<rank_type>[\w]+)_rank$"
    PROPERTY_MAPPING = {
        "rank_type": property_spec(
            "Rank subtype.",
            strict=True,
            allowed_values={"dense": "Dense ranking", "ordinal": "Ordinal ranking"},
        ),
    }
```

Two shapes, enforced at class definition; a half declaration fails at import:

- Shape A: `key` names a `PROPERTY_MAPPING` key with an enumerable value
  space; `parametric_families` join the universe by family name.
- Shape B (multi-axis families collapsed into one subtype id): declare the
  universe explicitly with a resolver:

```py
def resolve_window(feature_name: str, options: Options) -> str | None:
    return options.get("window_function")

SUBTYPES = SubtypeDeclaration(universe={"median", "sum"}, resolver=resolve_window)
```

`supported` is a sparse per-framework override: frameworks absent from it
keep the full universe. A key naming no declared framework is a silent no-op
for matching but raises from `subtype_support_matrix()`, so a typo surfaces
as `subtype_error` in the audit. The derived
`supports_compute_framework` canonicalizes parametric instances (`ntile_2`
becomes `ntile`) and gates declared subtypes by the declaration; undeclared
subtypes and features without one stay open.

**The data steward** audits capability via `subtype_support_matrix()`
(supported subtypes per framework from `compute_framework_definition()`;
empty for abstract bases; for an operation group read from its implementations) and `get_feature_group_docs()` (`subtype_key`,
`subtypes`, `parametric_subtypes`, `subtype_support`, `subtype_error`). A
hand-overridden `supports_compute_framework` yields no declared matrix; the
misfit surfaces as `subtype_error`.

**The data user** sees the outcome on `resolve_feature(name, options=...)`:
`subtype` (e.g. `ntile_2`) and `subtype_family` (`ntile`, parametric
instances only).

### Empty Results

The contract is: a **final** requested feature must return a *schema-bearing*
result, meaning at least one column. **Zero rows is a valid result; zero columns
is not.** A filter that excludes every row, a join with no matches, or a time
window with no events all return a well-typed frame with the right columns and
no rows, and mloda passes these through unchanged.

The error fires only when a final requested result carries *no schema at all*. If
`calculate_feature` produces a result with no columns,
`ComputeFramework.run_validate_output_features` raises:

```
EmptyResultError: Result carries no schema (no columns): <FeatureGroupClassName>. ...
```

`EmptyResultError` is a `ValueError` subclass and is importable from the public
API: `from mloda.provider import EmptyResultError`. Intermediate feature groups
(those whose output feeds another feature group rather than the caller directly)
are never subject to this check.

#### The schema-presence gate

The guard detects a missing schema via the framework's existing
`ComputeFramework.extract_column_names(data) -> set[str]` (a classmethod): an empty set
means no schema, which is the error condition. Every framework already implements
this off schema metadata, so it works on a zero-row frame and costs nothing extra
(no row scan, collect, or count). No per-framework opt-in is needed when you
implement a new compute framework, as long as `extract_column_names` returns the
columns for a zero-row frame. Feature groups can call
`<Framework>.extract_column_names(data)` without an instance to resolve available columns.

There is one representational caveat. The schema-bearing frameworks (PyArrow,
Pandas, Polars, DuckDB, SQLite, Spark, Iceberg) carry their schema as metadata
even at zero rows, so a zero-row result keeps its columns and passes. The
PythonDict framework represents data as a columnar `dict[str, list]`, where the
schema is the set of keys and is present even at zero rows: `{"col": []}` is a
valid schema-bearing zero-row frame, while `{}` (zero columns) is the only
schema-less value. Emptiness is judged purely on schema presence, with no
opt-in: a zero-column result raises `EmptyResultError` uniformly on every
framework. One consequence: on a schema-less result (a zero-column frame),
column selection returns the result as is, so a misspelled requested column on
schema-less data does not produce a "column not found" error.

#### Filter column validation

`_validate_filter_columns` skips its column-presence and dtype checks only when
the data is the schema-less empty result (in practice PythonDict's `{}`).
Filtering an empty result is a no-op, so neither the column check nor row
elimination has anything to do. Data on which the framework cannot see columns
but that is not an empty result still fails the missing-filter-column check
loudly.

### Column name case sensitivity

mloda matches column and feature names exactly (case-sensitive) in every lookup, match and presence check; engines that resolve names differently add the rules below.

-   DuckDB and SQLite compare identifiers ignoring ASCII case only (`val` and `VAL` are one column; `é` and `É` are two), and mloda's checks for new and helper column names fold case the same way. A join, as-of merge, union, append, `from_arrow` or `from_dict` whose output names differ only in case, or repeat a name exactly, raises `ValueError`; rename one side upstream.
-   On those engines, mask primitives, the `partition_by` / `order_by` columns of `with_row_number` / `window`, merge keys, as-of `by` and time columns, `select` names and SQLite `order` names must be exact column names. SQLite also accepts its `rowid` / `oid` / `_rowid_` pseudo-columns in `select`, `order`, `with_row_number` and `window`. Raw SQL fragments (`filter` conditions, `project` expressions, the `window` function, DuckDB `order`) bind through the engine and stay case-insensitive; on SQLite an unresolvable quoted name there is read as a string literal, not an error.
-   Spark column resolution follows `spark.sql.caseSensitive` (default `false`, folding like Python's `str.lower`, Unicode included). The Spark as-of merge and adding a feature column in `transform` raise `ValueError` on names that collide under that setting; Spark joins, union and append are not checked. Spark as-of time columns must match exactly; other Spark lookups (keys, masks, filters) bind through Spark's resolver.

### Row count for observability

`ComputeFramework._row_count(self, data) -> int | None` supplies the row counts extenders read
off `HookContext` (`rows_in`/`rows_out`) on the `FEATURE_GROUP_CALCULATE_FEATURE` and
`INPUT_DATA_LOAD` hooks (see [Extender](../chapter1/extender.md)). The default,
`HookContext.row_count(data)`, is a best-effort `len(data)` that already counts a columnar
`dict[str, list]` by its first column, so `PythonDictFramework` needs no override. Override
`_row_count` only when the native type's `__len__` is missing, wrong, or would materialize or
query the data: `DuckDBFramework` and `SqliteFramework` both return `None` for their relation
objects, since `len()` on either runs a real `COUNT(*)` query rather than reading metadata.

### Timezone and unit validation (merge and filter engines)

A custom compute framework's merge and filter engines can opt into the
[comparison contract](comparison-contract.md), which rejects incompatible
timezone/unit combinations in equi-joins, as-of joins, and datetime filter bounds.
The guard is **opt-in**: set `provides_column_semantics = True` on your
`BaseMergeEngine` / `BaseFilterEngine` subclass and implement
`_column_semantics(data, column)` to report the column's native semantics. Leave the
flag at its default `False` (for a framework with no temporal intent) and the guard
is skipped entirely, so you are never forced to implement the hook. An engine that
opts in but forgets the hook raises a clear error rather than silently skipping
validation. As-of joins always require `_column_semantics` regardless of the flag,
since ordered time columns are intrinsic to the operation.

### Operation Groups: one group, one Implementation per framework

An operation is written once as an `OperationFeatureGroup` (naming, options, input features) and served by one `Implementation` per compute framework. Both come from `mloda.provider`. `AggregatedFeatureGroup`, `TimeWindowFeatureGroup`, `MissingValueFeatureGroup` and `TextCleaningFeatureGroup` are built this way.

```py
class MyOperation(OperationFeatureGroup):
    ...  # naming, PROPERTY_MAPPING, input_features

class PandasMyOperation(OperationImplementation[MyOperation]):
    feature_group = MyOperation
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[MyOperation], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        ...  # group is the matched feature group class
```

Operation implementations subclass `OperationImplementation` (a plain `Implementation` naming an operation group raises `TypeError`); loaders keep `Implementation`.

- **Framework set**: the frameworks that have an implementation (a framework subclass is served by its parent's implementation), intersected with `compute_framework_rule()`. A group with no implementation has no framework and never matches.
- **Shared loop**: put the shared per-feature loop on the group as a classmethod that takes the per-framework pieces as keyword-only callables (`AggregatedFeatureGroup.aggregate(data, features, perform=..., add_column=..., columns=...)`). Each implementation's `calculate` hands its own methods over through the `group` it is called with, never through `cls.feature_group`, so a subclass of the group is served with its overrides. An `Implementation` subclass that leaves `calculate` abstract is ignored when implementations are loaded.
- **Capability**: `supported` (a frozenset of subtypes, `None` means all) and the classmethod `accepts(feature_name, options)` (default `True`) decide whether an implementation can run a feature. A framework whose implementation rejects it is not a candidate for that feature.
- **Divergence**: `divergent` maps a subtype to a short reason when the implementation's result differs from the group's reference contract (for example `{"sum": "skips NaN in sum"}`). The contract is documented on the group; implementations that follow it declare nothing. Divergence is declared per operation, not detected from the data: a NaN-free column still prefers the matching implementation. Polars `max` is declared divergent because its multi-column form propagates NaN.
- **Loading**: an operation group's implementations are found by import, so call `PluginLoader.all()` (or import the implementation modules) before running.
- **Hooks**: an operation group declares no column-wise hooks; the loop's keyword-only callables are its contract and mypy checks the `aggregate`, `window`, `impute` or `clean` call inside each implementation's `calculate`.

### Choosing among implementations

When several frameworks can run a step, the choice order is: frameworks the step must run on (pins, a single allowed framework), then an implementation that matches the reference contract before one that differs, then fewer conversions, then converting later, then the run's `compute_frameworks` list order, then the default order. The recorded reason says what happened:

- `prefers a matching version`: a divergent implementation was skipped although it would otherwise have won.
- `reads in place`: a source read from a connection in the collection costs one conversion when served off a framework that accepts that connection, so the work stays in the database ahead of list order.
- `<reason>; runs a version that differs: skips NaN in sum`: the step runs a divergent implementation (only option, pinned, or no matching alternative).

### Pinning an implementation

```py
Feature("sales__sum_aggr", implementation="PandasAggregation")
Feature("sales__sum_aggr", implementation=PandasAggregation)
Feature("x__sum_aggr__max_aggr", implementation=["PandasAggregation", "PyArrowAggregation"])
```

`implementation=` takes a class name (or `module:qualname`), an `Implementation` class, or a list of either. One entry pins every step of that group in the feature's chain; a list of N entries pins N steps of the same group in input order (inner step first). The pin is part of the feature's identity and is not passed to input features. In a feature config use the `implementation` key. Pinned steps get the reason `pinned`.

An unknown name, a name shared by two loaded classes, an implementation that is disabled, denied or whose framework is unavailable, one that conflicts with `compute_framework=`, a wrong entry count, or two requests pinning one step differently raises `ImplementationPinError` (a `ValueError`) naming the feature and the class.

### Migrating from per-framework classes

- `PandasAggregatedFeatureGroup`, `PyArrowAggregatedFeatureGroup` and `PolarsLazyAggregatedFeatureGroup` are replaced by `AggregatedFeatureGroup` with `PandasAggregation`, `PyArrowAggregation` and `PolarsLazyAggregation`.
- The per-framework time window, missing value and text cleaning classes are replaced by `TimeWindowFeatureGroup` (`PandasTimeWindow`, `PyArrowTimeWindow`), `MissingValueFeatureGroup` (`PandasMissingValueImputation`, `PyArrowMissingValueImputation`, `PythonDictMissingValueImputation`) and `TextCleaningFeatureGroup` (`PandasTextCleaning`, `PythonDictTextCleaning`).
- Implementations are found by import: import the implementation modules or call `PluginLoader.all()`.
- An allow or deny list that names an old class must name the group or the implementation instead.
- Plan locks recorded before the move no longer match.
- A subclass of an old per-framework class must subclass the group (or the implementation) instead.
- Time window: Pandas skips NaN in sum, avg, mean, std, var and count, which diverges from the reference. With Pandas and PyArrow both enabled those windows run on PyArrow (NaN propagates, slower); pin `implementation="PandasTimeWindow"` to keep the Pandas result.
- PythonDict text cleaning turns NaN into an empty string, like Pandas.
- PyArrow and PythonDict imputation treat NaN as missing (skipped in statistics and filled) and break `mode` ties by the smallest value, like Pandas; string ties change too.
- PyArrow `std` and `var` windows use the sample statistic like Pandas, so a one-row window is null for a single column and NaN for a multi-column feature.
- `feature_group="<old per-framework class>"` becomes `implementation="<implementation>"`, or `compute_framework=...`, for every moved family.
- A steward disables one implementation with `PluginCollector.add_disabled_implementation_classes`.
- With Pandas and PyArrow both enabled, a sum now runs on PyArrow (it matches the reference contract; Pandas skips NaN) and may add a conversion. Pin `implementation="PandasAggregation"` to keep the old behaviour.
- A provider customizes an implementation by overriding its methods (for example `_perform_aggregation`, `_perform_window_operation`, `_perform_imputation`, `_apply_operation`) on an implementation subclass, and the loop by overriding the group's loop (`aggregate`, `window`, `impute`, `clean`) on a group subclass.

## Framework Selection Process

mloda picks one compute framework per step group (the features of one FeatureGroup that run as one step) in one planning step, before link joins are resolved.

- **Allowed set**: the framework must fit the FeatureGroup (`compute_framework_rule`, `supports_compute_framework`), any pin via `Feature(compute_framework=...)`, and the run's enabled frameworks.
- **Connection skip**: a `REQUIRED` framework (DuckDB, SQLite) is a candidate for a feature, roots included, when the DataAccessCollection resolves a connection for it: by the feature's `connection_handle` option, or as the single matching connection. Two matching connections without a handle drop it, with a reason naming the handles. A non-root feature can also use the connection a parent carries, so the framework is kept when every parent can run on it; parents carrying different connections keep the step off that framework. Connections come only from the DataAccessCollection. A connection alone does not select DuckDB or SQLite: pin the feature or restrict the run with `compute_frameworks=[DuckDBFramework]`. A source read from that connection (`DuckdbFG`) is the exception, see `reads in place` below.
- **Bound connection**: the resolved connection is bound to every step that needs it, roots included. Parents on two different handles never share a DuckDB or SQLite consumer; a consumer pinned or restricted to that framework fails at planning naming both handles.
- **Joins**: a link's child runs on one of its two sides. RIGHT joins run on the right side's framework, unless the consumers cannot run there, then on the left side's (only for links joining two different FeatureGroups). APPEND and UNION on the left. Children of one link agree on the side.
- **Filters**: a filter pinned to a framework moves its host feature onto that framework.
- **Conversions**: a transformer chain must exist between every parent and child on different frameworks.
- **Choice among valid plans**: lowest cost first (one point per conversion, plus one point when a source read from a connection in the DataAccessCollection runs off a framework that accepts that connection, ranked before any conversion; reason `reads in place`), then the plan that converts later (a conversion deeper in the feature graph, by longest path from a root, of the features that convert, or at the end into `output_framework`), then the order of the run's `compute_frameworks` list, then the default order. The default order puts Pandas, Polars, PyArrow before `SELF_MANAGED` frameworks (Spark, Iceberg), before `REQUIRED` ones (DuckDB, SQLite), then class name.
- **Reason**: each step records why it got its framework: `pinned`, `only allowed framework`, `rules exclude preferred frameworks`, `saves N conversion(s)`, `converts later`, `your list order` or `default order`. "saves N conversions" counts the conversions added by moving only that step to a preferred framework.
- **Plans equal in cost and lateness**: blocks are settled in a fixed order (by FeatureGroup and feature names), each taking the most preferred framework still possible.
- **Both ways**: a Polars-only consumer pulls its unrestricted source onto Polars, and a Polars-only source pulls its unrestricted consumers, so no transform step is needed.
- **No valid plan**: planning raises an error naming the features and their allowed sets.

List order is a tie-break after cost, not "first listed wins". To force a framework, pin the feature or restrict the run's list.

`Feature.get_compute_framework()` returns the chosen framework. On a feature that allows several frameworks and was never planned, it raises.

Framework authors declare the connection rule by overriding `connection_requirement()` (default `ConnectionRequirement.NONE`). A `REQUIRED` framework also overrides `connection_of(data)` to return the connection its native data carries, so a step after a parent on the same framework needs no connection of its own.

## Appending Results

Override `_append_columns(data, columns)` to append a non-root dict result to the frame. The default returns `None`, which replaces the frame. Frameworks holding a lazy plan with unstable row order override the classmethod `_positional_append_needs_pinned_frame` to return `True`, which appends only onto a frame mloda built itself.

A single returned column (a list or Series) follows the same rule: on such a framework `transform` raises a `ValueError` naming the feature unless the frame is pinned. The fix is to return the column inside the frame the feature group received, in the framework's own data type. A frame counts as pinned when it was built from a dict, converted by a transformer in a step or at a framework transfer, appended onto a pinned frame, filtered from one, or passed unchanged through a same-framework hop from a pinned frame. The output of a framework's own `transform()` for a non-native result is treated as pinned too, so it must not reorder rows. A feature group that returns its own native frame leaves it unpinned for the next feature group on that frame.

## Data Transformation

When data needs to move between compute frameworks:

1. The `transform` method converts data between frameworks
2. Each framework defines how to transform data to and from other frameworks
3. The system automatically handles these transformations when needed

For more details on how data transformation works between compute frameworks, see [Framework Transformers](framework-transformers.md).

## Framework Notes

- **DuckDB**: a step gets its connection from the DataAccessCollection (select one with the `connection_handle` option), or from the relation its parents hand in; a connection object in a feature option raises at planning. mloda validates it and pins its session timezone to UTC but never opens or closes it. A transform into DuckDB or SQLite needs a matching connection in the DataAccessCollection, otherwise planning (prepare, explain, run_all) fails with an error naming the step. An `output_framework` of DuckDB or SQLite needs one only when a requested step runs on another framework and its result is converted. It runs in SYNC mode only, so its steps stay in the parent process under a MULTIPROCESSING run.
- **Spark**: requires PySpark and a Java 17+ runtime (`JAVA_HOME`). mloda can auto-create a local `SparkSession`; for production, supply a configured one through the data access collection. The session stays in the parent process, so Spark steps run in SYNC or THREADING mode, never in a multiprocessing worker; Spark's own distributed processing covers scale-out.
- **Iceberg**: needs a catalog supplied through the data access collection. The catalog stays in the parent process, so Iceberg steps run in SYNC or THREADING mode, never in a multiprocessing worker.

A run requesting only `{ParallelizationMode.MULTIPROCESSING}` drops these frameworks at setup; a run left with no usable framework, or a feature pinned to a dropped one, raises. Request `{SYNC, MULTIPROCESSING}` or `{THREADING, MULTIPROCESSING}` to combine them with worker-dispatched frameworks.

## SQL Relation Helpers (DuckDB / SQLite)

The DuckDB and SQLite frameworks expose a relation object (`DuckdbRelation`, `SqliteRelation`) inside `calculate_feature`. Both share the same helper surface so a feature group can be written once against either.

### Result type

A requested feature computed on DuckDB or SQLite comes back as a `pa.Table`, fetched in full when the result is selected. With `output_framework=DuckDBFramework` or `output_framework=SqliteFramework`, a step that ran on that framework returns the relation it computed, on the connection the step used. The relation is lazy: every read runs the query again, the connection must stay open, and SQLite leaves one temporary view on it. A result converted from another framework is loaded as a new relation onto the matching connection in the DataAccessCollection. A run whose requested steps all run on the output framework needs no such connection.

Framework authors: `select_native_data_by_column_names` returns the selection in the framework's own type (the default delegates to `select_data_by_column_names`). A framework whose `select_data_by_column_names` returns another framework's data type must also override `result_framework()` to name that framework. With that framework as `output_framework`, a run calls `select_native_data_by_column_names` instead of `select_data_by_column_names`, so a subclass that customizes one should customize the other.

### Reading column types

The `.types` property returns column types aligned with `.columns`. The element type differs by backend:

- `DuckdbRelation.types` returns DuckDB-native dtype objects.
- `SqliteRelation.types` returns PyArrow `pa.DataType` objects (from propagated hints, falling back to SQLite affinity inference).

```py
relation.columns   # ["user_id", "amount"]
relation.types     # backend-specific dtype objects, same order as columns
```

`SqliteRelation.to_arrow_table()` types a column by its propagated hint, else by its non-null values (int and float mixed gives float64, any other mix gives string, with numbers as text).
A computed column without a non-null value takes the type its SQL expression yields for two placeholder rows, so its type is the same with and without rows unless the expression's type depends on the values; a projection reading `rowid` cannot be probed and is string.
A relation mloda did not build falls back to the declared column affinity when it has no rows. In a table, a typeless or `NUMERIC`-affinity column (`NUMERIC`, `DECIMAL`, `BOOLEAN`, `DATE`) is then string, like an all-null column; with rows, its values decide. The relation's `select`, `filter`, `limit`, `order`, `join`, and `append_columns` results keep these types, as do union, append, and as-of merge results. Any other view reports the affinity SQLite gives its columns, so a typeless column there is large_binary when it has no rows.
A column Arrow cannot build (for example bytes that are not UTF-8 in a text column, or text that is not a date in a date column) raises a `ValueError` naming it.

### Window functions

`with_row_number` appends a `ROW_NUMBER()` column; `window` appends an arbitrary window expression. Both quote every identifier and raise `ValueError` if the new `alias` collides with an existing column (the comparison ignores ASCII case) or if a `partition_by` / `order_by` column is not an exact column name.

```py
from mloda_plugins.compute_framework.base_implementations.sql.sql_window import (
    OrderBy,
    WindowFrame,
    Preceding,
    CurrentRow,
)

# ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts)
ranked = relation.with_row_number(
    "rn",
    partition_by=["user_id"],
    order_by=[OrderBy("ts")],
)

# SUM(amount) OVER (PARTITION BY user_id ORDER BY ts ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
rolling = relation.window(
    "SUM(amount)",
    "amount_rolling",
    partition_by=["user_id"],
    order_by=[OrderBy("ts")],
    frame=WindowFrame(kind="rows", start=Preceding(2), end=CurrentRow()),
)
```

`order_by` accepts plain column-name strings or `OrderBy(column, descending=..., nulls="first"|"last")`. The `func` passed to `window` is inlined verbatim as raw SQL, so never build it from user-controlled input.

!!! warning "SQLite version requirement"
    On SQLite, `with_row_number` and `window` require SQLite >= 3.28.0; using `NULLS` placement in `order_by` additionally requires SQLite >= 3.30.0. Both raise `ValueError` on older runtimes. DuckDB has no such gate.
