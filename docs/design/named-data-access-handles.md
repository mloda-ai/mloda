# Named Data Access Handles

Status: proposal
Tracking issue: [#443](https://github.com/mloda-ai/mloda/issues/443)
Related: [#440](https://github.com/mloda-ai/mloda/issues/440), [#442](https://github.com/mloda-ai/mloda/pull/442)

## Problem

`DataAccessCollection` holds resources in unkeyed `set`s (or a single overwriteable dict). Every consumer that needs to pick one resource from a multi-resource DAC has reinvented disambiguation independently, or has not implemented any at all. When two resources of the same kind are present, the chosen one is non-deterministic across processes (set iteration depends on `PYTHONHASHSEED`), or the second silently overwrites the first.

Issue #443 fixed the most acute case (TFS connection binding) by raising on multi-match. That change is the right shape but covers only one DAC field. The same bug class is latent in `files`, `folders`, `credential_dicts`, and any future consumer that scans a `DataAccessCollection` set.

This document proposes a single registry pattern that closes the bug class for every field, generalizes the fail-fast contract introduced in #442, and gives data providers and stewards a stable identifier for each resource.

## Inventory on `origin/main` (post-#442)

| DAC field | Type today | Consumer | Disambiguation today |
|---|---|---|---|
| `files` | `set[str]` | `read_file.py:122`, `read_document.py:77` | `column_to_file` (file-only, column-keyed) |
| `folders` | `set[str]` | `read_file.py:122` + `os.listdir` at `:151`; `read_document.py:77` + `os.listdir` at `:104` | none |
| `initialized_connection_objects` | `set[Any]` | `ComputeFramework.pick_connection_from_dac` (4 SQL CFWs) | **fail-fast `ValueError`** (#442) |
| `credential_dicts` | `HashableDict` (single) | `read_db.py:92` | none; `add_credential_dict` silently overwrites on second call (`data_access_collection.py:46`) |
| `uninitialized_connection_objects` | `list[Any]` | none in production | dead field |

The DAC docstring itself names the problem: "Use `column_to_file` to pin column names to specific files when multiple files share the same column name, avoiding non-deterministic first-match-wins resolution." That is an admission that the bug class exists for one field, with a one-off remediation that does not generalize.

## Goal

One resolution rule that every DAC consumer can apply, with one universal hint key feature authors can set when the registry is ambiguous, and one introspection method stewards can call to enumerate what is registered.

Non-goal: changing what a resource *is* (a `duckdb.Connection`, a `Path`, a credential dict). Only how it is named and looked up.

## Proposal

### 1. Keyed registry per field

Each DAC field gains a dict-keyed variant alongside the existing set/dict field:

```python
DataAccessCollection(
    connections={"warehouse": duckdb_warehouse, "analytics": duckdb_analytics},
    files={"transactions": "/data/tx.parquet", "users": "/data/users.csv"},
    folders={"raw": "/data/raw/"},
    credentials={"pg-prod": {...}, "snowflake-dev": {...}},
)
```

Keys are arbitrary strings chosen by the data provider. They are the stable identifier used in logs, errors, and policy.

The existing positional/set parameters (`files: set[str]`, `folders: set[str]`, `initialized_connection_objects: set[Any]`, `credential_dicts: dict`) stay accepted for back compatibility. Internally they are normalized into the keyed registry using synthetic auto-keys (see Migration below).

### 2. Universal hint key: `data_access_handle`

A single options key, namespaced under the consumer it applies to, tells the resolver which handle to bind:

```python
Feature(
    "revenue",
    options=Options(context={"data_access_handle": "warehouse"}),
)
```

The hint is framework-agnostic and field-agnostic. The resolver decides which DAC field to look in based on the consumer's required type (a CFW asks for a connection, `ReadFile` asks for a file, etc.).

The existing per-FG-class options pattern at `compute_framework.py:203` can keep working as a back-compatibility alias. New code should target the universal key so handles are framework-agnostic.

### 3. Resolution rule (shared by every consumer)

For any consumer that needs to bind one resource of type `T` from the DAC:

1. **Explicit handle**: if the feature options set `data_access_handle = "X"`, look up handle `X` across all DAC fields. Bind iff the entry exists and matches type `T`. Otherwise raise, naming the handle and the type expected.
2. **Implicit single match**: else, filter the registry for entries matching type `T`. If exactly one matches, bind it.
3. **Implicit no match**: zero matches → raise with `"no resource of type T in DataAccessCollection"`.
4. **Implicit multi match**: more than one match → raise with `"ambiguous: candidates are [warehouse, analytics]; set 'data_access_handle' to disambiguate"`.

Rule 4 is exactly the contract shipped in #442 for connections, generalized to every field.

### 4. Stewardship

One introspection method:

```python
DataAccessCollection.handles() -> dict[str, str]
# {"warehouse": "connection", "analytics": "connection",
#  "transactions": "file", "users": "file",
#  "raw": "folder", "pg-prod": "credential", "snowflake-dev": "credential"}
```

Enumerates the registry without committing to a policy framework. Errors raised by the resolver can quote the relevant subset of this map so the user sees both available handles and their kinds.

## Why named handles collapse the bug class

The scenarios where today's pattern silently fails have one shape on different fields:

- Two DuckDB conns in `initialized_connection_objects` → wrong table at runtime (#443, fixed by #442's raise).
- Two CSVs with column `x` in `files` → wrong rows at runtime.
- A folder of mixed CSVs in `folders` → wrong file at runtime, OS-dependent.
- Multiple credential dicts → second one silently overwrites the first.
- A future second SparkSession or Iceberg catalog → repeat of #443 in a different framework.

Named handles reduce all five to one invariant: *if the registry has more than one entry of the requested type and the consumer did not pass a handle, raise and list the candidates*. Same rule, same error shape, regardless of field.

## Migration plan

Three landing waves so no consumer breaks in a single PR.

### Wave 1: introduce the keyed fields, normalize sets to synthetic keys

`DataAccessCollection.__init__` accepts both the legacy set/dict params and the new keyed params. Internally everything is stored as `dict[str, T]`:

- Each entry passed via a legacy `set` is assigned a synthetic key (e.g. `"_auto_<n>"` or a hash-stable derivative). Synthetic keys are reserved (prefix-locked) so they do not collide with user keys.
- `add_initialized_connection_object` and friends keep working; they append with the next synthetic key.
- When a synthetic-keyed entry is involved in an ambiguity, the resolver still raises, but the error message includes a `DeprecationWarning` pointer: "anonymous entries cannot be referenced by `data_access_handle`; pass `connections={...}` instead."
- `add_credential_dict` keeps its old single-dict shape but emits a `DeprecationWarning` on the second call (today: silent overwrite). The dict it sets is normalized into the keyed registry under a synthetic key.

This wave changes no public type signatures. Existing tests pass. The bug class is still latent for set-built DACs; users who opt into keyed registries get the fail-fast contract for free.

### Wave 2: roll the resolution rule into every consumer

- `ComputeFramework.pick_connection_from_dac` already implements the rule (#442). Generalize the same loop into a shared helper that operates on any field of the registry, parametrized by the type predicate.
- `read_file.py:122` and `read_document.py:77` switch from `set | set` first-match-wins to the shared resolver. Behavior change: ambiguity raises (matching #442's connection contract) unless `data_access_handle` or `column_to_file` resolves it. `column_to_file` keeps working as a file-specific hint that takes precedence over `data_access_handle` for the column it pins.
- `read_db.py:92` switches from "just take the one credential dict" to the shared resolver over `credentials`. If only one credential is registered, behavior is identical; if more than one, the resolver disambiguates by handle or raises.

This wave is a behavior change: code that today silently picks one of N resources now raises unless the user disambiguates. The user pain (an error at planning time, with a list of available handles) is strictly smaller than the pain it replaces (wrong data at runtime, no signal). Release notes call this out.

### Wave 3: deprecate the set/dict params

Once consumers are on the shared resolver and at least one minor release has shipped, the legacy `set`/single-dict params emit `DeprecationWarning` at construction time, with a suggested keyed-dict rewrite. Removal is a later major.

## Test matrix

One row per DAC field, two scenarios per row.

| Field | Scenario A (single resource) | Scenario B (multi-resource, no hint) | Scenario C (multi-resource, hint) |
|---|---|---|---|
| `connections` (DuckDB) | binds the conn | raises with handle list | binds the hinted conn |
| `connections` (SQLite) | binds the conn | raises with handle list | binds the hinted conn |
| `connections` (Spark) | binds the session | raises with handle list | binds the hinted session |
| `connections` (Iceberg) | binds the catalog | raises with handle list | binds the hinted catalog |
| `files` | resolves the file | raises with handle list | binds the hinted file |
| `folders` | resolves the folder | raises with handle list | binds the hinted folder |
| `credentials` | binds the dict | raises with handle list | binds the hinted dict |

Connection rows (A, B for all four frameworks) already exist as `TfsConnectionInitMixin.test_raises_on_multiple_matches` (#442). Wave 2 adds the other rows.

## Out of scope (flagged here, deferred to follow-up PRs)

- **`uninitialized_connection_objects`**: declared, settable, read by no production code. Either wire up the missing consumer or remove the field. Independent of the registry change.
- **`multi_execute_step` wiring**: `compute_framework_executor.py` calls `init_connection_from_data_access` for `sync_execute_step` and `thread_execute_step` but not `multi_execute_step`. Latent because no current SQL CFW allows MULTIPROCESSING. Flag with an inline comment so the next CFW that loosens this does not rediscover it.
- **Iceberg catalog-vs-table ambiguity**: `iceberg_framework.py` `_connection_matches` accepts both a catalog and a `Table`. With named handles, two entries of compatible-but-different shape can be registered under different handles, so users can pin one. Worth a regression test row when Wave 2 lands.
- **Per-FG-class hint (Issue #443 Option B)**: deliberately deferred. The universal `data_access_handle` key is the new convention; the FG-class-keyed convention at `compute_framework.py:203` stays as a back-compat alias. If a per-FG-class hint is needed later, it layers on top of the registry without changing the resolver rule.
- **Tagged resources** (`{"role": "warehouse", "env": "prod"}`): more expressive than names but adds policy questions (which tag axes are queryable?) that named handles dodge. Reconsider only if handles become limiting.
- **FG-owned connection declarations** (`class FraudFG: data_access_handle = "warehouse"`): can layer on as a supplement to the registry. Out of scope for the registry PR; revisit once handle adoption is high enough to warrant it.

## Personas, one line each

- **Data user** (writes features, calls `run_all`): in the ambiguous case, sees an error at planning time that names the exact handle to set, instead of a `CatalogException` at runtime.
- **Data provider** (curates the DAC): can add a second resource of any kind without breaking existing pipelines, as long as the handle names referenced by features still exist.
- **Data steward** (governance, audit): every resource has a stable identifier that shows up in logs and is queryable via `handles()`.

## Open questions

1. Reserved synthetic-key prefix: `"_auto_"` collides cheaply with hand-written keys. A less-likely prefix (`"__mloda_auto_"` or a per-field UUID) is safer; the error path is the only place users ever see them.
2. Whether `column_to_file` stays as a file-specific hint (precedence over `data_access_handle` for the pinned column) or is folded into the universal hint key with a `kind=file` qualifier. Smaller blast radius to leave it alone in Wave 2 and revisit during Wave 3.
3. Whether `credentials` becomes `dict[str, dict]` from the outset or stays as a single `HashableDict` field with a deprecation path. The asymmetry is real (today's failure mode is second-write-wins, not first-match-wins), but the resolver rule treats both uniformly, so collapsing the field is the cleaner end state.
