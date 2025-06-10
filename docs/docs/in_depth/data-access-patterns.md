# Data Access Patterns: BaseInputData vs MatchData

## Overview

mloda provides two distinct but complementary patterns for data access: **BaseInputData** and **MatchData**. While they may appear similar at first glance, they serve different purposes and are used in different contexts within the framework.

This document clarifies the differences between these concepts and their respective use cases. For practical examples of data access methods, see [Data Access Overview](access-feature-data.md).

## BaseInputData Pattern

### Purpose
BaseInputData is an **abstract base class** that defines how feature groups **load and access data**. It's the foundation for data loading mechanisms in mloda.

### Key Characteristics
- **Data Loading Focus**: Primarily concerned with how to load data from various sources
- **Feature Group Integration**: Used by feature groups through the `input_data()` method
- **Inheritance-Based**: Concrete implementations inherit from BaseInputData
- **Scope Management**: Supports both global and feature-specific data access scopes
- **Universal Usage**: Used by all feature groups that need to load data

### Use Cases
- Reading files (CSV, JSON, Parquet, etc.)
- Connecting to databases
- Creating synthetic/test data
- Loading data from APIs
- Managing data dependencies between features

For detailed examples of these use cases, see the [data access documentation](access-feature-data.md).

### Example Implementation
``` python
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData

class ReadFileFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ReadFile()  # BaseInputData implementation
    
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        if reader is not None:
            data = reader.load(features)
            return data
        raise ValueError("Reading file failed.")
```

### Common BaseInputData Implementations
- **ReadFile**: For file-based data loading (see [access-feature-data](access-feature-data.md#global-scope-data-access))
- **DataCreator**: For generating synthetic data (see [access-feature-data](access-feature-data.md#data-creator))
- **ApiInputData**: For runtime data injection (see [access-feature-data](access-feature-data.md#apidata))
- **DatabaseConnector**: For database connections

## MatchData Pattern

### Purpose
MatchData is a **specialized matching mechanism** specifically designed for feature groups that require **framework connection objects**. It determines which data access method should be used when stateful connections are needed.

### Key Characteristics
- **Connection-Specific**: Only used for feature groups that need framework connection objects
- **Matching Logic**: Determines which data source matches when connections are involved
- **Scope Resolution**: Resolves conflicts between feature-scope and global-scope data access for stateful frameworks
- **Limited Usage**: Only applies to specific compute frameworks (like DuckDB) that require persistent connections

### When MatchData is Used
MatchData is **only** used in these specific scenarios:
- Feature groups that work with **stateful compute frameworks** (e.g., DuckDB)
- When **framework connection objects** are required
- For data sources that need **persistent connections** (databases, connection pools)

### Use Cases
- Matching DuckDB features to appropriate DuckDB connections
- Routing features to specific database connections based on credentials
- Resolving data access when multiple connection objects are available
- Enabling flexible connection configuration for stateful frameworks

### Example Implementation
``` python
from mloda_core.abstract_plugins.components.match_data.match_data import MatchData

class DuckDBFeatureGroup(AbstractFeatureGroup, MatchData):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        # Logic to determine if this matcher handles DuckDB connections
        if framework_connection_object and isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        if data_access_collection and data_access_collection.has_duckdb_connections():
            return data_access_collection.get_duckdb_connection()
        return None
```

## Key Differences

| Aspect | BaseInputData | MatchData |
|--------|---------------|-----------|
| **Primary Purpose** | Data loading and access | Connection object matching for stateful frameworks |
| **When Used** | All feature groups that load data | Only feature groups requiring framework connection objects |
| **Scope** | Universal data access pattern | Specialized for stateful compute frameworks |
| **Usage Pattern** | `input_data()` method in feature groups | Multiple inheritance: `AbstractFeatureGroup, MatchData` |
| **Connection Dependency** | Works with or without connections | Specifically designed for connection objects |
| **Framework Support** | All compute frameworks | Only stateful frameworks (DuckDB, database connections) |

## How They Work Together

BaseInputData and MatchData serve **different purposes** and are used in **different scenarios**:

### BaseInputData Workflow
1. **Feature groups** define their data loading strategy via `input_data()` method
2. **BaseInputData implementations** handle the actual data loading
3. Works with **all compute frameworks** (stateful and stateless)

### MatchData Workflow (Connection-Specific)
1. **Only used** when feature groups need **framework connection objects**
2. **MatchData** determines which connection object to use for stateful frameworks
3. **Only applies** to specific compute frameworks like DuckDB that require persistent connections

### Combined Usage Example
``` python
class DuckDBAnalyticsFeature(AbstractFeatureGroup, MatchData):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        # BaseInputData for general data loading
        return ReadFile()
    
    @classmethod
    def match_data_access(cls, feature_name: str, options: Options, 
                         data_access_collection: Optional[DataAccessCollection] = None,
                         framework_connection_object: Optional[Any] = None) -> Any:
        # MatchData for connection object matching
        if framework_connection_object and isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        return None
```

## Practical Examples

### Scenario 1: Standard File Processing (BaseInputData Only)

**Use Case**: Reading CSV files with Pandas
**Pattern**: Only BaseInputData is needed

``` python
class CsvProcessingFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ReadFile()  # BaseInputData handles file reading
```

### Scenario 2: DuckDB Analytics (BaseInputData + MatchData)

**Use Case**: Analytics with DuckDB requiring connection objects
**Pattern**: Both BaseInputData and MatchData are needed

``` python
class DuckDBAnalyticsFeature(AbstractFeatureGroup, MatchData):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ReadFile()  # BaseInputData for data loading
    
    @classmethod
    def match_data_access(cls, ...):
        # MatchData for connection matching
        return appropriate_duckdb_connection
```

For database connection patterns, see [Framework Connection Object](framework-connection-object.md).

### Scenario 3: In-Memory Processing (BaseInputData Only)

**Use Case**: Creating synthetic data with Pandas
**Pattern**: Only BaseInputData is needed

``` python
class SyntheticDataFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"synthetic_data"})  # BaseInputData for data creation
```

## Integration with Other Concepts

### Compute Frameworks
- **BaseInputData**: Works with all compute frameworks
- **MatchData**: Only works with stateful frameworks requiring connection objects

For more information, see:
- [Compute Frameworks](../chapter1/compute-frameworks.md)
- [Framework Connection Object](framework-connection-object.md)
- [Compute Framework Integration](compute-framework-integration.md)

### Feature Groups
These patterns are fundamental to how feature groups access data. For more details, see:
- [Feature Groups](../chapter1/feature-groups.md)
- [Feature Group Matching](feature-group-matching.md)

## Best Practices

### When to Use BaseInputData Only
- Working with stateless compute frameworks (Pandas, PyArrow, Polars)
- File-based data loading
- API data injection
- Synthetic data generation
- Most standard data processing scenarios

### When to Use BaseInputData + MatchData
- Working with stateful compute frameworks (DuckDB)
- Database connections requiring persistent state
- Connection pooling scenarios
- When framework connection objects are required

### Design Considerations
- **BaseInputData**: Focus on robust data loading, error handling, and performance
- **MatchData**: Focus on accurate connection matching and state management
- **Integration**: Use MatchData only when framework connection objects are actually needed

## Related Documentation

- **[(Feature) data](access-feature-data.md)** - Comprehensive guide to data access in mloda
- **[Framework Connection Object](framework-connection-object.md)** - Managing stateful connections (essential for understanding MatchData)
- **[Feature Groups](../chapter1/feature-groups.md)** - Introduction to feature groups
- **[Compute Frameworks](../chapter1/compute-frameworks.md)** - Overview of compute framework system
- **[Feature Group Matching](feature-group-matching.md)** - How features are matched to implementations

## Summary

BaseInputData and MatchData serve **different and specialized roles** in mloda's data access architecture:

- **BaseInputData** is the **universal pattern** for data loading - used by all feature groups that need to load data
- **MatchData** is a **specialized pattern** for connection object matching - only used by feature groups that require framework connection objects

**Key Understanding:**
- **Most feature groups** only use BaseInputData
- **MatchData is only needed** when working with stateful compute frameworks like DuckDB
- **They are not alternatives** - they solve different problems in different contexts

Understanding this distinction is crucial for:
- Choosing the right pattern for your use case
- Implementing feature groups correctly
- Working with stateful vs stateless compute frameworks
- Leveraging mloda's connection management capabilities

This separation allows mloda to provide both universal data access (BaseInputData) and specialized connection management (MatchData) while keeping the complexity contained to only those scenarios that actually need it.
