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
