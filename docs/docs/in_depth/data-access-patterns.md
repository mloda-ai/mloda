# Data Access Patterns: Format FeatureGroups and MatchData

## Overview

mloda reads data through **format FeatureGroups** (files, documents, databases and custom sources). **MatchData** is a separate, narrow pattern for feature groups that need framework connection objects. `BaseInputData` is only the marker base of `DataCreator` and `ApiInputData`, the return type of `input_data()`; it carries no reader behavior.

For practical examples of data access methods, see [Data Access Overview](access-feature-data.md).

## Data Sources

- **Format FeatureGroups**: one FeatureGroup per source format, claiming features through `CLAIM_ROUTES` instead of `input_data()`:
    - `ReadFileFG` with `CsvFG`, `ParquetFG`, `JsonFG`, `FeatherFG`, `OrcFG` (see [access-feature-data](access-feature-data.md#global-scope-data-access))
    - `ReadDBFG` with `SqliteFG`
    - `ReadDocumentFG` with `TextFG`, `PyFG`, `MarkdownFG`, `YamlFG`, `JsonDocumentFG`; documents skip suffixes owned by file format groups by default
- **DataCreator**: For generating synthetic data (see [access-feature-data](access-feature-data.md#data-creator))
- **ApiInputData**: For runtime data injection (see [access-feature-data](access-feature-data.md#apidata))

## Writing a format FeatureGroup

Subclass the base that fits your source and write only what it asks for. The base owns the matching: claim routes, source discovery, pointers, `data_access_handle`, `column_to_file`, per-run caching, ambiguity and missing-column errors, and the extender hook around the load.

A format group can also describe its data through `declared_attributes`, see [Declared Attributes and Input Requirements](feature-group-matching.md#declared-attributes-and-input-requirements).

- **FormatFeatureGroup** (any source): declare `CLAIM_ROUTES`, implement `find_sources` and `load_neutral`; optionally `columns`, `has_column`, `ambiguity_fix`, `describe_columns`, `count_rows`. `query_text` is reserved: a group whose routes do not require it declines, and aborts when pointed.
- **ReadFileFG**: implement `suffixes()` and `column_names(path)`. It inherits file and folder discovery, the pinned-file rule, and a default `load_neutral` that hands a `FileSource` to the compute framework; the stock `FileSource` transformers read csv, parquet, json, feather and orc.
- **ReadDocumentFG**: implement `suffixes()`; optionally `read_text` and `handover_suffixes`. It inherits the three names `<Group>`, `<Group>~source` and `<Group>~file_type`.
- **ReadDBFG**: implement `is_valid_credentials`, `database_identity`, `connect`, `list_tables`, `table_columns` and `produce_rows`. `is_valid_credentials` must never raise. `database_identity` must not contain credential values. `login_key(credentials)` returns a credential-free hashable where equal keys mean the same login (default: a keyed digest of the credentials without `table_name`, with values that are not plain scalars compared by equality); two logins on one source are told apart by `data_access_handle`. Override `login_key` when credential fields that do not change grants (timeouts, application names) differ between handles. `data_access_identity` names the dataset, not the login. The table catalog is cached per run per login (`login_key`) and preset table; with a preset `table_name`, only that table's columns are read. `rechecked_filters(features)` gives the filters a read may push into its query, since the framework applies them again after the load. A subclass whose calculation uses the loaded rows beyond loading them (ranks, totals) overrides `rechecked_filters` to return `[]`. `SqliteFG` pushes exact numeric and text-equality shapes into a parameterized WHERE and leaves everything else to the post-read filter, so a run that succeeds today returns the same rows; it pushes only when each untyped column holds one storage class, and a typed filter column only when its type matches the declared affinity (INT64, DOUBLE, STRING). To check credentials against `PropertySpec`s, use `validate_property_values` and convert its error:

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

- **ReadDBFG query route**: opt in with `CLAIM_ROUTES = (*ReadDBFG.CLAIM_ROUTES, ReadDBFG.QUERY_ROUTE)` and implement `produce_query_rows(connection, query_text, features)`. A query-only group lists just `(ReadDBFG.QUERY_ROUTE,)` and stubs the table hooks. The source is a hash of the query text, so textually different queries are different sources. Without a pointer the route claims any leaf name for each matching credential in the DataAccessCollection, and it switches off the table route for that feature. A derived name such as `price__sum_aggr` goes to the group that computes it; `feature_group=<Group>` overrides that.

A format group's `PROPERTY_MAPPING` is merged over the declarations of every class in its MRO, mixins included, even when the group declares none of its own: own keys win and base keys cannot be dropped, so `PROPERTY_MAPPING = None` on a format group raises `TypeError`. Class definition is checked: a `ClaimRoute` with a wrongly typed field, a file or document route whose `source_kind` is not `"file"` or `"folder"`, and `suffixes()` that does not return a non-empty tuple of non-empty strings all raise `TypeError`. A concrete group's `suffixes()` runs at class definition.

A format group reports `match.source` as its `data_access_identity`; override it to publish a richer identity. Run the contract mixins in mloda's `tests/mixins/reader_feature_groups/` against your group; they pin the behavior every format group shares.

CSV inference semantics are defined by pyarrow's default CSV reader; the stdlib reader behind PythonDict follows it. Types are inferred per column: null tokens (pyarrow's default set, e.g. `NA`, `NaN`, `null`) become `None` in an int/float/bool column but stay literal text in a string column, a column of only empty cells and/or null tokens is all-`None`, and an int column with a value outside signed int64 range degrades entirely to float.

#### Loaders and compute frameworks

A database implementation receives `match.access` as a `DBTable` (or a `DBQuery` on the query route), both from `mloda.provider`. A subclass that changes parsing (for example a semicolon CSV overriding only `column_names`) must also override `load_neutral`, otherwise the inherited neutral form and the inherited implementations still parse the parent format.

A per-framework fast path is an `Implementation` (from `mloda.provider`): set `feature_group` (a `FormatFeatureGroup` subclass) and `compute_framework`, and write a classmethod `calculate(group, match, features, framework)` that returns the data. At plan build mloda picks one per step over the governed set, walking the group MRO, then the framework MRO, most specific pair first; the serving subclass is passed as `group`. A class that overrides `load_neutral` cuts off implementations inherited from above it, and a subclass of an implementation replaces it. Two implementations of one exact pair where neither subclasses the other stop the environment build with `EnvironmentPreconditionError`. Without an implementation the group loads `load_neutral` and a transformer converts. An implementation must stay referenced: define it at module level, because an unreferenced class is garbage-collected and no longer found. Disable one with `PluginCollector.add_disabled_implementation_classes`. Calls made without an engine (`count_rows`) see every loaded implementation, not the governed set. `Implementation.version()` hashes the class like a FeatureGroup's. `PlanStep`, the plan lock and `HookContext.implementation` record which implementation ran.

The stdlib reader does not yet cover pyarrow's full surface. Where they differ, a column pyarrow types stays a string column in PythonDict: dates, timestamps and times, whitespace-padded numbers (`" 1 "`), `inf`/`infinity`, uppercase `NAN` (the float value, not the `NaN` null token), hex literals (`0x1f`), and a `true`/`1` mix (pyarrow reads `1`/`0` as bools too).

### Column discovery and row counts

`describe_columns(match)` maps column name to `DataType` (`None` where unknown) and raises `NotImplementedError` when the group cannot enumerate columns, `ImportError` when a backend it needs is missing, and `OSError` or `ValueError` for an unreadable source. `ReadFileFG` lists the names from `column_names`; `ParquetFG`, `FeatherFG` and `OrcFG` report the types stored in the file's schema, `JsonFG` the types pyarrow infers while parsing, and `SqliteFG` SQLite's declared (unenforced) column types. A document group reports its three names as strings.

`count_rows(match, compute_framework)` returns the row count of the source without loading it, or `None` when only a read can tell (the default). `ParquetFG`, `OrcFG` and `FeatherFG` count from file metadata for every compute framework; `CsvFG` only under PythonDict, since other frameworks read CSV through pyarrow, whose row split can differ; a document group counts one. The count is of the source itself, not of the step's output: filters, extenders and feature-group logic may change what the step reports. Both are called on the group with the `SourceMatch`; from a prepared session use `session.describe_columns(step)` and `session.count_rows(step)` (see the mloda API page).

### Pointing a feature at a source

- `options={"CsvFG": path}` points the group at one file (or folder). The class object works as the key too (`options={CsvFG: path}`), and a `data_access_name()` override does not add an alias pointer: only class names point. A subclass also answers to its parent's name, so `Feature("x", options={"CsvFG": path})` reaches a `CsvFG` subclass. A pointer makes the other format groups skip the collection, and a pointer at a group that is not accessible in the run raises, naming the filter that dropped it.
- `options={"SqliteFG": Credential(sqlite="/x.db")}` points a database group at one database. A plain-dict pointer works too: the matched options hold it as a `RegisteredCredential`, so `str(options)` masks its values. Prefer `Credential` for secrets: a pointer forwarded to a consumer prints raw there.
- `Feature(name, options={"query_text": ..., "<Group>": Credential(...)})` runs a query on a database group that opts into `ReadDBFG.QUERY_ROUTE`; a group without the route declines when unpointed and aborts when pointed.
- `Feature(..., feature_group=CsvFG)` scopes resolution to that group (and its subclasses) without choosing a source.
- `data_access_handle` only narrows: it picks one of the sources the `DataAccessCollection` holds and never points a group at a source the collection lacks.
- `column_to_file` pins columns to files in the `DataAccessCollection`; a pinned file that cannot serve the request aborts instead of falling back (see [access-feature-data](access-feature-data.md)).

The matched pair lives on `Feature.input_data_match`, never in the options, and is handed to the loader at load time. The source may hold credentials, so use `PlanStep.data_access_identity` (or `Group.data_access_identity(match)` outside a plan) for display and logs. If your own error text may carry a credential, scrub it with `mloda.provider.scrub_credentials` before logging.

### Non-file sources such as HTTP

Write a `FormatFeatureGroup` subclass with a pointed-only `NamePolicy.OPEN` claim route: it claims a name only when pointed at (`options={"MyApiFG": url}` or `feature_group=MyApiFG`). Implement `find_sources` and `load_neutral` to fetch and return the data. A plain `FeatureGroup` is not a pointer target. A key that differs from a loaded format group's name only by case raises a `FormatPointerError`; a key naming a group that is not loaded is an unknown option key and is ignored. `ApiInputData` injects in-memory data passed through the API request and is not an HTTP client.

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

Rules for format group authors:

- Record only when ownership is established but the content fails: right suffix but a missing column, valid credentials but a declined feature. Plain non-matches (wrong suffix, invalid credentials) stay silent.
- Never raise to decline: a hook that raises anything but `OSError`, `ValueError` or `ImportError` from `column_names` (or an unmarked `NotImplementedError` where a base documents it) ends matching for that candidate. A raise marked with `escalate_match_abort` propagates. Record, then return a falsy value.
- Recording outside an engine-opened window is a no-op, so groups stay usable standalone. In tests, use `collect_match_rejections` (see [Feature Group Matching](feature-group-matching.md)).
- Recorded reasons are discarded at the enclosing candidate level: when the group ultimately matches, or when the feature group matches by another rule. Only a decline surfaces them.
- Name the group and the concrete source in the reason, as the example does. The first recording per owner wins, so a later reason under a name already used in the same window is dropped; only the missing-column summary replaces a reason the same group recorded during its own probe.
- A `ReadFileFG` subclass whose `column_names` raises `NotImplementedError` is eliminated for every feature name. One that raises `ImportError` declines any name, with a recorded reason. A file pinned with `column_to_file` aborts with a missing-column message ("column ... is in none of the sources", or "could not list the columns of any of its sources" when none can list). A `ReadDocumentFG` subclass declines any name other than its three declared names.
- In `ReadFileFG` matching, an unpinned file whose columns cannot be read (`OSError`, `ValueError`) is declined with a recorded reason, so a shipped file group needs the file to be readable when features resolve. A pinned or pointed file that cannot be read aborts with the missing-column error instead of falling back to another file.

`ReadFileFG` column validation and the `ReadDBFG` catalog check already record automatically; a custom group only needs this for its own decline points.

A `feature_group=` scope is final: only the scoped group and its subclasses are candidates. A class-name pointer is final against other format groups and the DataAccessCollection. A pointed group can end its match in two ways:

- It aborts, for example on a `query_text` it cannot serve, a refusal at INPUT_DATA_MATCH, or a missing column. These aborts are deferred: they yield to a candidate that computes the feature from input features, or to a subclass of the pointed group that takes it over. The several-sources ambiguity abort is not deferred and always raises.
- It declines, for example on a PROPERTY_MAPPING value or required-key failure, no source found, or a name the group does not declare. A plain FeatureGroup that claims the name by its own rules still serves it.

Since group options forward to input features, a pointer on a derived feature binds its inputs too (see [Resolution errors](troubleshooting/feature-group-resolution-errors.md#pointers-on-derived-features)). A pinned group that matches nothing without a recorded reason is reported under its own name.

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
        # Connections come from the DataAccessCollection only
        if data_access_collection is not None:
            return data_access_collection.resolve(
                "connection",
                predicate=lambda c: isinstance(c, duckdb.DuckDBPyConnection),
                hint=options.get("connection_handle") or options.get("data_access_handle"),
            )
        return None
```

### Connection Handle
- A connection lives only in the DataAccessCollection. A connection object under a feature option, including the group's class name, raises at planning and asks you to move it to the collection and select it with `connection_handle`. The option under a group's class name is only a format pointer.
- When `match_data_access` returns a connection of the collection, `global_scope_data_access` writes its handle to the `connection_handle` option, never the object under the class name. The handle is picklable, so MatchData features stay safe in mixed SYNC/MULTIPROCESSING runs.
- The handle forwards to input features, but an input feature whose own matcher selects another connection overrides the inherited one. A handle the feature sets itself still conflicts with its matcher.

## Combined Usage Example
```py
class DuckDBAnalyticsFeature(FeatureGroup, MatchData):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"analytics_input"})

    @classmethod
    def match_data_access(cls, feature_name: str, options: Options,
                         data_access_collection: DataAccessCollection | None = None,
                         framework_connection_object: Any | None = None) -> Any:
        # MatchData for connection object matching, from the DataAccessCollection
        if data_access_collection is not None:
            return data_access_collection.resolve(
                "connection", predicate=lambda c: isinstance(c, duckdb.DuckDBPyConnection)
            )
        return None
```
## Practical Examples

### Scenario 1: Standard File Processing (Format FeatureGroup Only)

**Use Case**: Reading CSV files with Pandas
**Pattern**: Neither is needed; `CsvFG` claims the feature when a file has the column

```py
Feature("id", options={"CsvFG": "data/people.csv"})
```

### Scenario 2: DuckDB Analytics (DataCreator + MatchData)

**Use Case**: Analytics with DuckDB requiring connection objects
**Pattern**: `DataCreator` supplies the data and MatchData matches the connection

```py
class DuckDBAnalyticsFeature(FeatureGroup, MatchData):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"analytics_input"})

    @classmethod
    def match_data_access(cls, ...):
        # MatchData for connection matching
        return appropriate_duckdb_connection
```

For database connection patterns, see [Framework Connection Object](framework-connection-object.md).

### Scenario 3: In-Memory Processing (DataCreator Only)

**Use Case**: Creating synthetic data with Pandas
**Pattern**: Only `DataCreator` is needed

```py
class SyntheticDataFeature(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"synthetic_data"})
```

## Integration with Other Concepts

### Compute Frameworks
- **Format FeatureGroups and `DataCreator`**: work with all compute frameworks
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

### When to Use Format FeatureGroups or DataCreator Only
- Working with stateless compute frameworks (Pandas, PyArrow, Polars)
- Files, documents and databases (format FeatureGroups)
- Runtime data injection (`ApiInputData`) and synthetic data (`DataCreator`)
- Most standard data processing scenarios

### When to Add MatchData
- Working with stateful compute frameworks (DuckDB)
- Database connections requiring persistent state
- Connection pooling scenarios
- When framework connection objects are required

### Design Considerations
- **Format FeatureGroups**: Focus on robust source discovery, error handling, and performance
- **MatchData**: Focus on accurate connection matching and state management
- **Integration**: Use MatchData only when framework connection objects are actually needed

## Related Documentation

- **[(Feature) data](access-feature-data.md)** - Comprehensive guide to data access in mloda
- **[Framework Connection Object](framework-connection-object.md)** - Managing stateful connections (essential for understanding MatchData)
- **[Feature Groups](../chapter1/feature-groups.md)** - Introduction to feature groups
- **[Compute Frameworks](../chapter1/compute-frameworks.md)** - Overview of compute framework system
- **[Feature Group Matching](feature-group-matching.md)** - How features are matched to implementations

## Summary

- **Format FeatureGroups** are the way mloda reads files, documents, databases and custom sources
- **MatchData** is a specialized pattern for connection object matching, only used by feature groups that require framework connection objects
- They are not alternatives: they solve different problems in different contexts
