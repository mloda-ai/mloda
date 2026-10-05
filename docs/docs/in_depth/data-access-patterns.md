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
- Reading files, documents and databases (through format FeatureGroups)
- Connecting to databases
- Creating synthetic/test data
- Loading data from APIs
- Managing data dependencies between features

For detailed examples of these use cases, see the [data access documentation](access-feature-data.md).

### Example Implementation
```py
from typing import Any

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet

class SyntheticFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"synthetic"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"synthetic": [1, 2, 3]}
```

### Common Implementations
- **Format FeatureGroups**: one FeatureGroup per source format, claiming features through `CLAIM_ROUTES` instead of `input_data()`:
    - `ReadFileFG` with `CsvFG`, `ParquetFG`, `JsonFG`, `FeatherFG`, `OrcFG` (see [access-feature-data](access-feature-data.md#global-scope-data-access))
    - `ReadDBFG` with `SqliteFG`
    - `ReadDocumentFG` with `TextFG`, `PyFG`, `MarkdownFG`, `YamlFG`, `JsonDocumentFG`; documents skip suffixes owned by file format groups by default
- **DataCreator**: For generating synthetic data (see [access-feature-data](access-feature-data.md#data-creator))
- **ApiInputData**: For runtime data injection (see [access-feature-data](access-feature-data.md#apidata))

### Writing a format FeatureGroup

Subclass the base that fits your source and write only what it asks for. The base owns the matching: claim routes, source discovery, pointers, `data_access_handle`, `column_to_file`, per-run caching, ambiguity and missing-column errors, and the extender hook around the load.

- **FormatFeatureGroup** (any source): declare `CLAIM_ROUTES`, implement `find_sources` and `load_neutral`; optionally `columns`, `has_column`, `ambiguity_fix`, `describe_columns`, `count_rows`.
- **ReadFileFG**: implement `suffixes()` and `column_names(path)`. It inherits file and folder discovery, the pinned-file rule, and a default `load_neutral` that hands a `FileSource` to the compute framework.
- **ReadDocumentFG**: implement `suffixes()`; optionally `read_text` and `handover_suffixes`. It inherits the three names `<Group>`, `<Group>~source` and `<Group>~file_type`.
- **ReadDBFG**: implement `is_valid_credentials`, `database_identity`, `connect`, `list_tables`, `table_columns` and `produce_rows`. `is_valid_credentials` must never raise. `database_identity` must not contain credential values. To check credentials against `PropertySpec`s, use `validate_property_values` and convert its error:


- **ReadDBFG**: implement `is_valid_credentials`, `database_identity`, `connect`, `list_tables`, `table_columns`, and `produce_rows`. `is_valid_credentials` must never raise. To check credentials against `PropertySpec`s, use `validate_property_values` and convert its error:

    ```python
    from typing import Any

    from mloda.provider import PropertySpec, PropertyValidationError, validate_property_values

    CREDENTIAL_SPECS = {"host": PropertySpec("Database host")}


    def is_valid_credentials(credentials: dict[str, Any]) -> bool:
        try:
            validate_property_values(credentials, CREDENTIAL_SPECS, closed_world=True)
        except PropertyValidationError:
            return False
        return True


    assert is_valid_credentials({"host": "db"})
    assert not is_valid_credentials({"host": "db", "password": "secret"})
    ```

Override `data_access_identity` to publish a richer identity than the default. Run the contract mixins in mloda's `tests/mixins/reader_feature_groups/` against your group; they pin the behavior every format group shares.


CSV inference semantics are defined by pyarrow's default CSV reader; the stdlib reader behind PythonDict follows it. Types are inferred per column: null tokens (pyarrow's default set, e.g. `NA`, `NaN`, `null`) become `None` in an int/float/bool column but stay literal text in a string column, a column of only empty cells and/or null tokens is all-`None`, and an int column with a value outside signed int64 range degrades entirely to float.

The stdlib reader does not yet cover pyarrow's full surface. Where they differ, a column pyarrow types stays a string column in PythonDict: dates, timestamps and times, whitespace-padded numbers (`" 1 "`), `inf`/`infinity`, uppercase `NAN` (the float value, not the `NaN` null token), hex literals (`0x1f`), and a `true`/`1` mix (pyarrow reads `1`/`0` as bools too).

### Column discovery and row counts

`describe_columns(match)` maps column name to `DataType` (`None` where unknown) and raises `NotImplementedError` when the group cannot enumerate columns, `ImportError` when a backend it needs is missing, and `OSError` or `ValueError` for an unreadable source. `ReadFileFG` lists the names from `column_names`; `ParquetFG`, `FeatherFG` and `OrcFG` report the types stored in the file's schema, `JsonFG` the types pyarrow infers while parsing, and `SqliteFG` SQLite's declared (unenforced) column types. A document group reports its three names as strings.

`count_rows(match, compute_framework)` returns the row count of the source without loading it, or `None` when only a read can tell (the default). `ParquetFG`, `OrcFG` and `FeatherFG` count from file metadata for every compute framework; `CsvFG` only under PythonDict, since other frameworks read CSV through pyarrow, whose row split can differ; a document group counts one. The count is of the source itself, not of the step's output: filters, extenders and feature-group logic may change what the step reports. Both are called on the group with the `SourceMatch`; a resolved plan does not expose them.

### Pointing a feature at a source

- `options={"CsvFG": path}` points the group at one file (or folder). A subclass also answers to its parent's name, so `Feature("x", options={"CsvFG": path})` reaches a `CsvFG` subclass.
- `options={"SqliteFG": Credential(sqlite="/x.db")}` points a database group at one database. Prefer `Credential` for secrets: a plain-dict pointer shows its values in `str(options)`.
- `Feature(..., feature_group=CsvFG)` scopes resolution to that group (and its subclasses) without choosing a source.
- `data_access_handle` only narrows: it picks one of the sources the `DataAccessCollection` holds and never points a group at a source the collection lacks.
- `column_to_file` pins columns to files in the `DataAccessCollection`; a pinned file that cannot serve the request declines instead of falling back (see [access-feature-data](access-feature-data.md)).

The matched pair lives on `Feature.input_data_match`, never in the options, and is handed to the loader at load time. The source may hold credentials, so use `PlanStep.data_access_identity` (or `Group.data_access_identity(match)` outside a plan) for display and logs. If your own error text may carry a credential, scrub it with `mloda.provider.scrub_credentials` before logging.

### Non-file sources such as HTTP

Write a plain `FeatureGroup` that claims a feature only when it is pointed at (`options={"MyApiFG": url}` or `feature_group=MyApiFG`), under the pointed-only route of the format FeatureGroup matcher contract. `govdata` is the example. `ApiInputData` injects in-memory data passed through the API request and is not an HTTP client.

### Resolution of a pointed group vs feature-group resolution

Source selection runs inside [feature-group resolution](feature-group-matching.md): the group's matcher is the criteria gate, so a group is accessible, scoped and framework-checked like any other feature group. Several sources found for one name in one group are an error naming them and the fix; two groups finding the same column is ordinary feature-group ambiguity (fix with `feature_group=`, a pointer or `data_access_handle`). See [Resolution errors](troubleshooting/feature-group-resolution-errors.md).

### Declining with an attributable reason

A format group that owns a source but cannot serve the requested feature records why it declined; the reason then appears in the near-miss block of the "No feature groups found" error message, labeled `(input data)`. `record_match_rejection` is exported via `mloda.provider`. A custom file format group owns its own suffix and lists its columns; a `ValueError` from `column_names` declines the file with a recorded reason, for example a required schema marker in the header:

```python
from typing import Any

from mloda.provider import FeatureSet, ReadFileFG

class SensorCsvFG(ReadFileFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".sensorcsv",)

    @classmethod
    def column_names(cls, path: str) -> list[str]:
        with open(path, encoding="utf-8") as handle:
            header = handle.readline()
        if "#sensor-schema" not in header:
            raise ValueError("its header lacks the #sensor-schema marker")
        return header.replace("#sensor-schema", "").strip().split(",")

    @classmethod
    def load_neutral(cls, match: Any, features: FeatureSet) -> Any: ...
```

The recorded decline renders as a near-miss line of the resolution failure:

```
  - SensorCsvFG (input data): SensorCsvFG matched /data/run1.sensorcsv but could not read its columns: its header lacks the #sensor-schema marker
```

The recorded decline renders as a near-miss line of the resolution failure:

```
  - SensorCsvFG (input data): SensorCsvFG matched /data/run1.sensorcsv but could not read its columns: its header lacks the #sensor-schema marker
```

Rules for format group authors:

- Record only when ownership is established but the content fails: right suffix but a missing column, valid credentials but a declined feature. Plain non-matches (wrong suffix, invalid credentials) stay silent.
- Never raise to decline: a hook that raises anything but `OSError`, `ValueError` or `ImportError` from `column_names` (or an unmarked `NotImplementedError` where a base documents it) ends matching for that candidate. A raise marked with `escalate_match_abort` propagates. Record, then return a falsy value.
- Recording outside an engine-opened window is a no-op, so groups stay usable standalone.
- Recorded reasons are discarded at the enclosing candidate level: when the group ultimately matches, or, for unowned recordings, when the feature group matches by another rule. An owned veto instead gates the name-based rules (see the paragraph below). Only a decline surfaces them.
- Name the group and the concrete source in the reason, as the example does. The first recording per owner wins, so a later reason under a name already used in the same window is dropped.
- A `ReadFileFG` subclass that cannot enumerate columns (a `column_names` raising `NotImplementedError` or `ImportError`) declines a chain- or column-separated feature name while matching. A `ReadDocumentFG` subclass declines any name other than its three declared names. An explicit `column_to_file` pin is exempt.
- In `ReadFileFG` matching, an unpinned file whose columns cannot be read (`OSError`, `ValueError`) is declined with a recorded reason, so a shipped file group needs the file to be readable when features resolve. A pinned or pointed file that cannot be read aborts with the missing-column error instead of falling back to another file.

`ReadFileFG` column validation and the `ReadDBFG` catalog check already record automatically; a custom group only needs this for its own decline points.

A veto recorded while the user explicitly addressed the group (an option key equal to its `data_access_name()`) gates the candidate's name-based match rules: the feature group fails at resolution with that reason instead of resolving by name and crashing at load time. A content decline on that path gates the same way: if the addressed group records a decline and its probe still matches nothing, the recording counts as owned. A decline followed by a match on another input of the same probe stays discarded as usual. A pinned group is final: when it declines, neither another group nor the DataAccessCollection serves the feature. Since group options forward to input features, a pointer on a derived feature binds its inputs too (see [Resolution errors](troubleshooting/feature-group-resolution-errors.md#pointers-on-derived-features)). A pinned group that matches nothing without a recorded reason is reported under its own name. An unowned decline on the global probe stays near-miss material only, and the MatchData rule is not gated.

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
```py
from mloda.provider import MatchData, FeatureGroup

class DuckDBFeatureGroup(FeatureGroup, MatchData):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        # Logic to determine if this matcher handles DuckDB connections
        if framework_connection_object and isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        if data_access_collection is not None:
            return data_access_collection.resolve(
                "connection",
                predicate=lambda c: isinstance(c, duckdb.DuckDBPyConnection),
                hint=options.get("data_access_handle"),
            )
        return None
```

### Connection Forwarding
- The connection value forwards to input features like any other option, including across a same-class chain (an upstream feature resolved by the same MatchData class as its consumer).
- Both entry points (`global_scope_data_access` and `feature_scope_data_access`) mark the class-name key non-forwarded for pickling, without blocking normal option flow.
- Pickling an `Options` with a marked key (multiprocessing preflight or a worker handoff) drops that key from the snapshot, so MatchData features stay safe in mixed SYNC/MULTIPROCESSING runs even though the connection itself can't be pickled. In-process reads are unaffected.

## Key Differences

| Aspect | BaseInputData | MatchData |
|--------|---------------|-----------|
| **Primary Purpose** | Data loading and access | Connection object matching for stateful frameworks |
| **When Used** | All feature groups that load data | Only feature groups requiring framework connection objects |
| **Scope** | Universal data access pattern | Specialized for stateful compute frameworks |
| **Usage Pattern** | `input_data()` method in feature groups | Multiple inheritance: `FeatureGroup, MatchData` |
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
```py
class DuckDBAnalyticsFeature(FeatureGroup, MatchData):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        # BaseInputData for general data loading
        return DataCreator({"analytics_input"})

    @classmethod
    def match_data_access(cls, feature_name: str, options: Options,
                         data_access_collection: DataAccessCollection | None = None,
                         framework_connection_object: Any | None = None) -> Any:
        # MatchData for connection object matching
        if framework_connection_object and isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        return None
```

## Practical Examples

### Scenario 1: Standard File Processing (Format FeatureGroup Only)

**Use Case**: Reading CSV files with Pandas
**Pattern**: Neither is needed; `CsvFG` claims the feature when a file has the column

```py
Feature("id", options={"CsvFG": "data/people.csv"})
```

### Scenario 2: DuckDB Analytics (BaseInputData + MatchData)

**Use Case**: Analytics with DuckDB requiring connection objects
**Pattern**: Both BaseInputData and MatchData are needed

```py
class DuckDBAnalyticsFeature(FeatureGroup, MatchData):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"analytics_input"})  # BaseInputData for data creation

    @classmethod
    def match_data_access(cls, ...):
        # MatchData for connection matching
        return appropriate_duckdb_connection
```

For database connection patterns, see [Framework Connection Object](framework-connection-object.md).

### Scenario 3: In-Memory Processing (BaseInputData Only)

**Use Case**: Creating synthetic data with Pandas
**Pattern**: Only BaseInputData is needed

```py
class SyntheticDataFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
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
- Runtime data injection and synthetic data (files and databases use format FeatureGroups)
- mloda data injection
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
