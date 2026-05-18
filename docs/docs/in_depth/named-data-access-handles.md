# Named Data Access Handles

`DataAccessCollection` (DAC) is a registry of data resources keyed by stable string handles. Every consumer that needs to bind one resource uses a single resolution rule that raises on ambiguity rather than letting iteration order decide.

This page is the user-facing guide. For background and design rationale, see the design doc that accompanied [issue #443](https://github.com/mloda-ai/mloda/issues/443) in PR #446.

## The four kinds of resources

A DAC holds resources of four kinds. Each kind has its own keyed dict:

``` py
DataAccessCollection(
    connections={"warehouse": warehouse_conn, "analytics": analytics_conn},
    files={"transactions": "/data/tx.parquet", "users": "/data/users.csv"},
    folders={"raw": "/data/raw/"},
    credentials={"pg-prod": {"host": "...", "user": "..."}, "snowflake-dev": {...}},
)
```

Handle names are arbitrary strings you choose. They are globally unique across kinds: you cannot register a connection and a file under the same name. Registration raises `ValueError` on duplicates.

Mutators mirror the keyed-dict shape:

``` py
dac.add_connection("warehouse", warehouse_conn)
dac.add_file("transactions", "/data/tx.parquet")
dac.add_folder("raw", "/data/raw/")
dac.add_credentials("pg-prod", {"host": "..."})
```

A small runnable example, backed by `sqlite3` so it has no external dependencies:

```python
import sqlite3
from mloda.user import DataAccessCollection

primary = sqlite3.connect(":memory:")
secondary = sqlite3.connect(":memory:")

dac = DataAccessCollection(
    connections={"primary": primary, "secondary": secondary},
    files={"users": "/tmp/users.csv"},
)

assert dac.handles() == {
    "primary": "connection",
    "secondary": "connection",
    "users": "file",
}
```

## Resolution rule

When a feature group asks the DAC for a resource of a given kind, the resolver applies one rule:

1. **If `data_access_handle` is set on the feature's `Options`**: look up that handle. If it does not exist, exists under a different kind, or fails the consumer's type predicate, raise `ValueError` with the actual situation and the available handles.
2. **Otherwise**: filter the kind's registry by the consumer's predicate (for example, "this file matches my suffix and columns").
    - Zero matches: return `None` (the consumer typically continues looking elsewhere).
    - One match: bind it.
    - More than one match: raise `ValueError` listing the candidate handles and telling the user to set `data_access_handle`.

The error shape is the same for every kind. This is the same contract `ComputeFramework.pick_connection_from_dac` shipped in #442, generalized to every consumer.

```python
import sqlite3
import pytest
from mloda.user import DataAccessCollection

dac = DataAccessCollection(
    connections={
        "primary": sqlite3.connect(":memory:"),
        "secondary": sqlite3.connect(":memory:"),
    },
)

assert dac.resolve("connection", hint="primary") is not None

with pytest.raises(ValueError) as excinfo:
    dac.resolve("connection")
assert "data_access_handle" in str(excinfo.value)
assert "'primary'" in str(excinfo.value) and "'secondary'" in str(excinfo.value)

with pytest.raises(ValueError) as excinfo:
    dac.resolve("connection", hint="missing")
assert "missing" in str(excinfo.value)
```

## Per-feature disambiguation: `data_access_handle`

When more than one resource of the same kind matches a feature's requirements, you disambiguate by setting `data_access_handle` on the feature's `Options`:

``` py
from mloda.user import DataAccessCollection, Options, Feature, mloda

dac = DataAccessCollection(
    connections={"warehouse": warehouse_conn, "analytics": analytics_conn},
)

features = [
    Feature("revenue",  options=Options(context={"data_access_handle": "warehouse"})),
    Feature("clicks",   options=Options(context={"data_access_handle": "analytics"})),
]

mloda.run_all(features, compute_frameworks=["DuckDB"], data_access_collection=dac)
```

The key works uniformly across all four kinds and across every CFW that consumes connections (DuckDB, SQLite, Spark, Iceberg today) plus `read_file`, `read_document`, and `read_db`.

## Introspection: `handles()`

`DataAccessCollection.handles()` returns a `{handle: kind}` map of everything registered. Use it for audits, logging, or to surface candidates in your own error messages:

``` py
dac.handles()
# {"warehouse": "connection", "analytics": "connection",
#  "transactions": "file", "users": "file",
#  "raw": "folder", "pg-prod": "credentials"}
```

## Interaction with `column_to_file`

`column_to_file` is a file-specific override that takes precedence over the resolver. Its values are **file handles** (keys of the `files` dict), not paths:

``` py
DataAccessCollection(
    files={"train": "application_train.csv", "bureau": "bureau.csv"},
    column_to_file={
        "SK_ID_CURR": "train",
        "TARGET":     "train",
        "AMT_CREDIT_SUM": "bureau",
    },
)
```

For columns listed in the map, the resolver short-circuits to the pinned file. For columns not listed, the regular resolver rule applies (single match binds; multiple raises and asks for `data_access_handle`).

See [Disambiguating columns shared across multiple files](access-feature-data.md#disambiguating-columns-shared-across-multiple-files) for the full example.

## Why this exists

Before #443, every DAC field was an unkeyed set. When two resources of the same kind were registered, Python's set iteration (`PYTHONHASHSEED`-dependent) decided which one the engine picked. The same pipeline could read the wrong table, wrong file, or wrong credential across processes, with no error and no signal at planning time.

Named handles collapse that bug class to one invariant: *if the registry has more than one entry of the requested kind and the consumer did not pass a handle, raise and list the candidates*. Same rule, same error shape, regardless of field.

## Common errors

- **"Handle 'X' is already registered under kind 'K'"**: you called `add_*` (or constructed the DAC) with a handle already used by another resource. Pick a different name.
- **"Handle 'X' not found for kind 'K'. Available handles of kind 'K': [...]"**: you set `data_access_handle='X'` but the DAC has no resource of that kind named `X`. Check the spelling or register it.
- **"Handle 'X' is registered under kind 'K1', but kind 'K2' was requested"**: you set `data_access_handle='X'`, but the registry has `X` under a different kind. A connection consumer cannot bind to a file handle even if the names match.
- **"Ambiguous resolve for kind 'K': N candidates [...]; set 'data_access_handle' in Options to disambiguate"**: more than one entry of the requested kind matched. Set `data_access_handle` on the feature's `Options` to pick one, or remove the extras from the DAC.

## Related

- [(Feature) data](access-feature-data.md): overview of data access in mloda.
- [Data Access Patterns](data-access-patterns.md): `BaseInputData` vs `MatchData`.
- [Framework Connection Object](framework-connection-object.md): stateful connection lifecycle.
