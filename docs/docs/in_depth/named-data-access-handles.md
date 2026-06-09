# Named Data Access Handles

`DataAccessCollection` (DAC) is a registry of data resources keyed by stable string handles. Every consumer that needs to bind one resource uses a single resolution rule that raises on ambiguity rather than letting iteration order decide.

For background, see [issue #443](https://github.com/mloda-ai/mloda/issues/443).

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

## Naming is optional

You only need to name resources when there are multiple sources of the same kind that the resolver cannot tell apart. In the simple single-source case, pass bare values:

``` py
DataAccessCollection(files={"/data/tx.parquet"})       # set or list
DataAccessCollection(connections={duckdb_conn})        # set
DataAccessCollection(folders={"/data/raw"})            # set
DataAccessCollection(credentials=Credential(host="h")) # typed single credential (recommended)
DataAccessCollection(credentials=[{"host": "h"}])      # list (dicts are unhashable)
```

The same shape applies to the mutators:

``` py
dac.add_file("/data/tx.parquet")                       # auto-named
dac.add_file("tx", "/data/tx.parquet")                 # named (when you need a handle)
```

Unnamed entries get internal auto-handles (`_auto_file_0`, `_auto_connection_0`, etc.) that you never need to reference. They exist only so the resolver has a unique key per entry. If you later hit ambiguity, the error message will tell you to switch to the named form and pick a `data_access_handle`.

Naming is only required when:

- Two or more resources of the same kind match the same consumer (then you must set `data_access_handle` to pick one), or
- You use `column_to_file` and prefer to reference files by name rather than path (both work, see below).

## Typed credentials: `Credential`

Credentials are the one kind where the named form and the value itself look identical, because a credential *is* a dict. These two lines differ only in brackets but mean completely different things:

``` py
credentials={"sqlite": "/data/x.db"}    # named form: handle "sqlite" -> credential "/data/x.db"
credentials=[{"sqlite": "/data/x.db"}]  # one credential whose mapping is {"sqlite": "/data/x.db"}
```

The first shape is almost never what you mean: it registers the string `"/data/x.db"` as the credential, which then fails every connector's `is_valid_credentials` check. The typed `Credential` class removes the ambiguity, so the meaning comes from the type instead of the nesting depth:

``` py
from mloda.user import Credential, DataAccessCollection

DataAccessCollection(credentials=Credential(sqlite="/data/x.db"))       # kwargs form
DataAccessCollection(credentials=Credential({"sqlite": "/data/x.db"}))  # dict form
DataAccessCollection(credentials=[Credential(sqlite="/a.db"), {"pg": {"host": "h"}}])
DataAccessCollection(credentials={"pg-prod": Credential(host="h")})     # named form
dac.add_credentials(Credential(host="h"))                               # mutators too
```

`Credential` is unwrapped to a plain dict at registration time, so feature groups and `is_valid_credentials` implementations keep receiving plain dicts. Nothing changes downstream.

Two safety behaviors come with it:

- **Early mis-wrap error.** Passing a bare dict whose values are not mappings (the mis-wrap shape above) now raises `ValueError` at construction time, naming the offending handle and showing the three correct alternatives, instead of failing silently later during matcher selection.
- **Redacted repr.** `repr(Credential(password="hunter2"))` prints `Credential(password='***')`. Keys stay visible, values never reach logs or tracebacks.

### Why a typed class: four kinds of users

The design serves four user groups that hit credentials differently:

1. **Notebook users** (one data source) write the obvious shape from an example. For them the bare dict either has to work or fail loudly at construction; `Credential(sqlite="/data/x.db")` gives them a form with no nesting decision to get wrong, and the early error catches the legacy shape.
2. **Production users** (multiple sources) live in the named-handle registry and disambiguate per feature via `data_access_handle`. For them `Credential` keeps the registry values homogeneous, so ambiguity errors and `handles()` introspection stay trustworthy.
3. **Plugin authors** implement `is_valid_credentials` and previously had to isinstance-juggle whatever leaked through (including bare strings from mis-wrapped input). The framework now normalizes at the boundary: whatever the end user typed, the plugin receives a plain dict.
4. **Ops and data stewards** care about what is in the credential, not its shape. The resolver's ambiguity errors print candidate values; the redacting `repr` keeps passwords out of those messages, out of logs, and out of stack traces.

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
    files={"transactions": "/data/tx.csv", "users": "/data/users.csv"},
)

features = [
    Feature("amount",   options=Options(context={"data_access_handle": "transactions"})),
    Feature("email",    options=Options(context={"data_access_handle": "users"})),
]

mloda.run_all(features, compute_frameworks=["PyArrowTable"], data_access_collection=dac)
```

The key works across `read_file`, `read_document`, and `read_db` on a per-feature basis, and across every CFW that consumes connections (DuckDB, SQLite, Spark, Iceberg today) on an engine-wide basis.

!!! note "Connections are resolved once per engine, not per feature"
    `ComputeFramework.pick_connection_from_dac` runs at engine setup, not on the per-request path. A single CFW therefore binds a single connection per session. If you need two features in the same session to bind different connections of the same CFW, that is currently out of scope (tracked separately); use a single `data_access_handle` on the DAC's matching connections, or split the run.

## Introspection: `handles()`

`DataAccessCollection.handles()` returns a `{handle: kind}` map of everything registered. Use it for audits, logging, or to surface candidates in your own error messages:

``` py
dac.handles()
# {"warehouse": "connection", "analytics": "connection",
#  "transactions": "file", "users": "file",
#  "raw": "folder", "pg-prod": "credentials"}
```

## Interaction with `column_to_file`

`column_to_file` is a file-specific override that takes precedence over the resolver. Its values may be either a **file handle** (key of the `files` dict) or a **file path** (value of the `files` dict); both are accepted and normalized to handles internally:

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
- **"credentials value for handle 'X' is not a mapping"**: you passed a bare `{connector_id: slot}` dict as `credentials`, which the named form reads as `{handle: credential}`. Use `credentials=Credential(...)`, the list form `credentials=[{...}]`, or the named form `{handle: {connector_id: slot}}`. See [Typed credentials](#typed-credentials-credential).

## Related

- [(Feature) data](access-feature-data.md): overview of data access in mloda.
- [Data Access Patterns](data-access-patterns.md): `BaseInputData` vs `MatchData`.
- [Framework Connection Object](framework-connection-object.md): stateful connection lifecycle.
