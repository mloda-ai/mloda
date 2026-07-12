# Feature Group Compute Framework Integration

## Overview

One of mloda's key strengths is its ability to decouple feature definitions from specific computation technologies. This document explains how feature groups integrate with different compute frameworks.

## Core Concepts

### Compute Framework Specification

Feature groups specify which compute frameworks they support through the `compute_framework_rule` method:

``` python
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

``` python
@classmethod
def supports_compute_framework(cls, feature_name, options, compute_framework) -> bool:
    """Reject an operation on a specific framework. Default returns True."""
    if compute_framework is SQLiteFramework and is_median_op(feature_name, options):
        return False  # median is unsupported on SQLite
    return True
```

Returning `False` removes that framework from the candidate set **for this
feature only**:

- If another framework can still run the operation, the matcher routes around
  the rejected one silently (no error).
- If the only remaining candidate is the rejected framework (for example, the
  user pinned the feature to it), resolution fails with a dedicated,
  actionable error that names the unsupported and supported frameworks, e.g.
  `Unsupported compute framework(s) for feature 'X': ['SQLiteFramework']. Supported on: ['DuckDBFramework', 'PandasDataFrame'].`

This is distinct from the "no feature group found" error, so an unsupported
operation is no longer indistinguishable from an unknown feature. Prefer this
hook over raising a generic error from inside `calculate_feature`: the
rejection happens during planning rather than at compute time, and the message
is built for you.

The debug inspector `resolve_feature(name)` reflects the hook too: its
`ResolvedFeature` result carries `supported_compute_frameworks` and
`unsupported_compute_frameworks` (evaluated under default options, since the
inspector does not know the caller's `Options`), and a feature that matches a
group but is unsupported on every installed framework resolves to
`feature_group=None` with a capability error rather than appearing runnable.

### Declaring Capability per Subtype

When a family's operations are enumerated by one `PROPERTY_MAPPING` key (the
aggregation type, the scaler type, ...), name that key with `SUBTYPE_KEY` and
declare which subtypes each framework supports:

``` python
class StatFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    SUBTYPE_KEY = "stat_type"
    PREFIX_PATTERN = r".*__([\w]+)_stat$"
    PROPERTY_MAPPING = {
        "stat_type": property_spec(
            "statistic to compute",
            strict=True,
            allowed_values={"sum": "Sum of values", "median": "Median value"},
        ),
    }

    @classmethod
    def supported_subtypes(cls, compute_framework):
        if compute_framework is SQLiteFramework:
            return frozenset({"sum"})  # median is unsupported on SQLite
        return cls.subtype_universe()
```

`subtype_universe()` derives from the key's declared values, and
`supports_compute_framework` is then derived from both: no override needed. A
subtype outside the universe (a parametric one such as `ntile_2`) stays allowed,
so the hook never double-gates what matching already rejects. Families that
flatten several axes into one subtype override `subtype_universe()` themselves.

The declaration is enumerable without probing: `subtype_support_matrix()` returns
framework class name -> supported subtypes (and raises if a framework claims a
subtype outside the universe), and `get_feature_group_docs()` exposes the same
data as `subtype_key`, `subtypes` and `subtype_support`. `resolve_feature(name)`
reports the concrete feature's `subtype`.

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
`ComputeFramework._extract_column_names(self, data) -> set[str]`: an empty set
means no schema, which is the error condition. Every framework already implements
this off schema metadata, so it works on a zero-row frame and costs nothing extra
(no row scan, collect, or count). No per-framework opt-in is needed when you
implement a new compute framework, as long as `_extract_column_names` returns the
columns for a zero-row frame.

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

### Framework-Specific Implementations

Feature groups follow a layered architecture:
- Base class defines the interface and common functionality
- Framework-specific classes implement the actual calculations

```
FeatureGroup
  └── BaseFeatureGroup (e.g., ClusteringFeatureGroup)
        ├── PandasImplementation
        ├── PyArrowImplementation
        └── PythonDictFrameworkImplementation
```

## Implementation Pattern

### 1. Base Class

The base class defines the interface and common functionality:

``` python
class MyFeatureGroup(FeatureGroup):
    """Base class for MyFeatureGroup."""
    
    def input_features(self, options, feature_name):
        # Common logic for extracting input features
        
    @classmethod
    def calculate_feature(cls, data, features):
        # This will be overridden by framework-specific implementations
        raise NotImplementedError()
```

### 2. Framework-Specific Implementation

Each framework-specific implementation:
- Specifies which compute frameworks it supports
- Implements the calculation logic for that framework

``` python
class PandasMyFeatureGroup(MyFeatureGroup):
    @classmethod
    def compute_framework_rule(cls):
        """Define supported compute frameworks."""
        return {PandasDataFrame}
    
    @classmethod
    def calculate_feature(cls, data, features):
        """Implement calculation using pandas."""
        # Pandas-specific implementation
```

## Framework Selection Process

When a feature is requested:

1. The system identifies the appropriate feature group
2. It checks which compute frameworks are supported by:
   - The feature definition
   - The feature group
   - The mloda request
3. It selects a compatible compute framework
4. It uses the framework-specific implementation for calculations

## Data Transformation

When data needs to move between compute frameworks:

1. The `transform` method converts data between frameworks
2. Each framework defines how to transform data to and from other frameworks
3. The system automatically handles these transformations when needed

For more details on how data transformation works between compute frameworks, see [Framework Transformers](framework-transformers.md).

## Example

For a clustering feature group:

``` python
# Base class (framework-agnostic)
class ClusteringFeatureGroup(FeatureGroup):
    def input_features(self, options, feature_name):
        # Extract source features from feature name
        
    @classmethod
    def calculate_feature(cls, data, features):
        # This will be overridden by framework-specific implementations

# Pandas implementation
class PandasClusteringFeatureGroup(ClusteringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls):
        return {PandasDataFrame}
    
    @classmethod
    def calculate_feature(cls, data, features):
        # Pandas-specific clustering implementation
        
# PyArrow implementation
class PyArrowClusteringFeatureGroup(ClusteringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls):
        return {PyArrowTable}
    
    @classmethod
    def calculate_feature(cls, data, features):
        # PyArrow-specific clustering implementation
```

For an aggregated feature group with Polars support:

``` python
# Base class (framework-agnostic)
class AggregatedFeatureGroup(FeatureGroup):
    def input_features(self, options, feature_name):
        # Extract source features from feature name
        
    @classmethod
    def calculate_feature(cls, data, features):
        # This will be overridden by framework-specific implementations

# Polars Lazy implementation
class PolarsLazyAggregatedFeatureGroup(AggregatedFeatureGroup):
    @classmethod
    def compute_framework_rule(cls):
        return {PolarsLazyDataFrame}
    
    @classmethod
    def calculate_feature(cls, data, features):
        # Polars lazy-specific aggregation implementation
        # Uses lazy evaluation for query optimization
```

Note that Polars supports both eager (`PolarsDataFrame`) and lazy (`PolarsLazyDataFrame`) evaluation modes, allowing you to choose the appropriate strategy based on your performance requirements.

For an analytical feature group with DuckDB support:

``` python
# Base class (framework-agnostic)
class AnalyticalFeatureGroup(FeatureGroup):
    def input_features(self, options, feature_name):
        # Extract source features from feature name
        
    @classmethod
    def calculate_feature(cls, data, features):
        # This will be overridden by framework-specific implementations

# DuckDB implementation
class DuckDBAnalyticalFeatureGroup(AnalyticalFeatureGroup):
    @classmethod
    def compute_framework_rule(cls):
        return {DuckDBFramework}
    
    @classmethod
    def calculate_feature(cls, data, features):
        # DuckDB-specific analytical implementation
        # Uses SQL-like operations for complex analytics
        # Example: data.aggregate("column", "sum").df()
        return data.aggregate("value_column", "sum")
```

**Important**: DuckDB feature groups require a connection object to be available. The framework will automatically handle connection management, but ensure your data access collection includes the necessary connection information.

For a distributed processing feature group with Spark support:

``` python
# Base class (framework-agnostic)
class DistributedFeatureGroup(FeatureGroup):
    def input_features(self, options, feature_name):
        # Extract source features from feature name
        
    @classmethod
    def calculate_feature(cls, data, features):
        # This will be overridden by framework-specific implementations

# Spark implementation
class SparkDistributedFeatureGroup(DistributedFeatureGroup):
    @classmethod
    def compute_framework_rule(cls):
        return {SparkFramework}
    
    @classmethod
    def calculate_feature(cls, data, features):
        # Spark-specific distributed processing implementation
        # Uses Spark DataFrame operations for scalable processing
        # Example: data.groupBy("category").agg({"value": "sum"})
        return data.groupBy("category_column").agg({"value_column": "sum"})
```

**Important**: Spark feature groups require PySpark installation and Java 8+ environment with JAVA_HOME configured. The framework can auto-create a local SparkSession if none is provided, but for production use, you should provide a configured SparkSession through the data access collection. Spark uses its own distributed processing capabilities instead of mloda's framework inherent multiprocessing.

## SQL Relation Helpers (DuckDB / SQLite)

The DuckDB and SQLite frameworks expose a relation object (`DuckdbRelation`, `SqliteRelation`) inside `calculate_feature`. Both share the same helper surface so a feature group can be written once against either.

### Reading column types

The `.types` property returns column types aligned with `.columns`. The element type differs by backend:

- `DuckdbRelation.types` returns DuckDB-native dtype objects.
- `SqliteRelation.types` returns PyArrow `pa.DataType` objects (from propagated hints, falling back to SQLite affinity inference).

``` python
relation.columns   # ["user_id", "amount"]
relation.types     # backend-specific dtype objects, same order as columns
```

### Window functions

`with_row_number` appends a `ROW_NUMBER()` column; `window` appends an arbitrary window expression. Both quote every identifier and raise `ValueError` if the new `alias` collides with an existing column.

``` python
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
