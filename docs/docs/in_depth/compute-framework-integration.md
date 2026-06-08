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

### Allowing Empty Results

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

`EmptyResultError` is a `ValueError` subclass. Intermediate feature groups
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
PythonDict framework represents data as `list[dict]`, where the schema lives in
each row's keys, so its only empty value is `[]`, which has no columns. A
PythonDict result that is empty is therefore always treated as schema-less and
must opt in via `allow_empty_result()` (see below) to be accepted.

#### Opting in to empty results

Override `allow_empty_result` on a feature group to declare that a schema-less
empty result is a legitimate outcome for that group:

``` python
class KnowledgeGraphFeatureGroup(FeatureGroup):
    @classmethod
    def allow_empty_result(cls) -> bool:
        """Zero matches is a valid answer for graph traversals."""
        return True
```

When `allow_empty_result()` returns `True`, the result flows through to the
caller unchanged regardless of schema. Typical use cases include knowledge
graphs, search, agent memory, and authorization filters, especially on the
PythonDict framework where an empty match cannot carry a schema.

#### Filter column validation

As a side effect, when `allow_empty_result()` is `True`, `_validate_filter_columns`
also skips its column-presence check on schema-less data. This prevents spurious
"column not found" errors when a filter is applied to legitimately empty output
that carries no columns to validate against.

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
