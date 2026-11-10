# Plan Lock

## What it records and why

Strict mode and `PluginPolicy` decide which classes may take part in a request. The plan lock records which class wins per step (feature group, compute framework, specialization, joins, framework transforms) plus input wiring and join keys, and fails when that changes. See [Plugin Registry](plugin_registry.md) for the policy side.

## Usage

```py
from mloda.steward import check_plan_lock

session = mloda.prepare(features, compute_frameworks={PandasDataFrame})
check_plan_lock(session.resolved_plan(), "plan.lock")
session.run()
```

Call `write_plan_lock(plan, path)` to create or update the file, and review the change in the pull request. `check_plan_lock` never writes and raises `PlanLockMismatchError` with a unified diff, or with the expected content when the file is missing.

CI can check `mloda.explain(...)` without running anything. Pass the same `parallelization_modes` as the real run, since `prepare` and `explain` default to `None` and `run_all` to `SYNC`.

```py
from mloda.steward import write_plan_lock

write_plan_lock(mloda.explain(features, parallelization_modes={ParallelizationMode.SYNC}), "plan.lock")
```

## What the file holds

Sorted JSON with a `format` number (not in the content hash), the `requested_features` names (in the content hash), and one record per compute, join and transform step. Class paths are `module:QualName`. Duplicate steps keep one record each. The last column says whether a key enters `plan_content_hash`.

**Compute records**

| Key | Meaning | When present | In content hash |
| --- | --- | --- | --- |
| `feature_names` | The step's feature names, sorted. | always | yes |
| `feature_group` | The feature group class path. | always | yes |
| `compute_framework` | The compute framework class path. | always | yes |
| `compute_framework_reason` | Why the step runs on its framework, for example `pinned`. | always, may be null | no |
| `specialized_from` | Parent groups this group replaced (subclass preference). | always, may be empty | yes |
| `result_framework` | The `output_framework` option if set, else the framework of the returned frame type. | always, may be null | yes |
| `input_feature_edges` | Each output feature's input names. | always, may be empty | yes |
| `leaves_connection` | The handle of the connection a root read or a requested result converted into another framework (including `output_framework`) leaves, `<unnamed>` for an auto-named connection. | always, may be null | no |
| `implementation` | The serving `Implementation` (`module:qualname`), `neutral` for a source group loaded without one. | always, may be null | yes |
| `implementation_hash` | The hash of the first-party code the implementation reaches, its group included, never dependency versions; `unavailable` when the source cannot be read (refused by `write_plan_lock`). | always, null without an implementation | yes |
| `implementation_specialized_from` | The winner's concrete `Implementation` ancestors that serve the step's pair. | always, may be empty | yes |
| `feature_group_hash` | The group's code hash; `unavailable` (not refused) for a group without readable source, whose edits are then not tracked. | only when an inherited implementation serves a subclass group or a source group loads neutrally | yes |
| `credential_handles` | Declared and bound handle names, sorted, `<unnamed>` for an auto handle, `<pointer>` for a credential not in the collection, both reserved so no handle can use them. | only when the step has any | no |
| `runtime_parameters` | The runtime parameter keys the step declares, sorted. | only when the step declares any | no |
| `contract_version` | The group's `CONTRACT_VERSION`. | only when set | yes |
| `divergences` | The stamped, scrubbed records of the group and its serving implementation that apply, each with `source`, `framework`, `kind`, `cause`, `subtypes`, `outputs`, `reason`, `variant` and `docs`, sorted. | only when non-empty | no |
| `allowed_variants` | The rule variants the step's features allow, sorted. | only when non-empty | no |

**Join records**

| Key | Meaning | When present | In content hash |
| --- | --- | --- | --- |
| `left_feature_group` | The link's declared left feature group. | always | yes |
| `right_feature_group` | The link's declared right feature group. | always | yes |
| `join_type` | The link's join type. | always | yes |
| `compute_framework` | The merge destination framework. | always | yes |
| `source_compute_framework` | The framework merged in. | always | yes |
| `destination_side` | The declared side holding the destination, `left` or `right`. | always, may be null | yes |
| `join_keys` | The link's `left=right` column pairs. | always, null for append/union | yes |

**Transform records**

| Key | Meaning | When present | In content hash |
| --- | --- | --- | --- |
| `feature_group` | The destination feature group. | always | yes |
| `from_compute_framework` | The framework the data comes from. | always | yes |
| `to_compute_framework` | The framework the data is transformed into. | always | yes |
| `leaves_connection` | The handle of the connection the transform leaves, `<unnamed>` for an auto-named connection. | always, may be null | no |

For `leaves_connection`, a step bound implicitly to the same handle as its parent counts as on that connection.

An older lock of a plan whose records carry `feature_group_hash`, `credential_handles` or `divergences` fails once until rewritten; for `credential_handles` that includes a plan whose source binds a handle sorting before a declared one. `allowed_variants` came without a format bump, so existing locks stay valid. A `format` bump (now 9) makes a lock file written by an older mloda fail `check_plan_lock` until rewritten with `write_plan_lock`.

When the only difference from the lock is a group's `contract_version`, `check_plan_lock` does not raise: it warns with `PlanLockContractVersionWarning` (from `mloda.steward`), naming each group with its locked and resolved version. A steward makes it fail with `warnings.filterwarnings("error", category=PlanLockContractVersionWarning)`.

`plan_structure_hash` includes the format number, so a bump changes it for every plan; `plan_content_hash` leaves the format out and changes only when the records do.

It never holds option values (except the handle names under `credential_handles`), `data_access_identity`, feature group or implementation versions, per-run ids or the mloda version. An overridden `Implementation.version()` does not reach the lock, which reads the hash.

## What it does not cover

- Code changes inside a feature group class. An implementation's code is covered by `implementation_hash`. For groups use `FeatureGroup.version()` on `HookContext`, see [Feature Group Version](feature-group-version.md).
- The neutral path: a group's own `load_neutral` is not hashed. Use `HookContext.feature_group_version`.
- Data.
- Extenders.

## Declared credential handles

The content hash includes the name of a declared credential handle, never the credential; the lock's `credential_handles` key is left out of it. `PlanStep.bound_credential_handle` is not written to the lock. The lock text and `plan_structure_hash` of a plan with a database source change once (rewrite with `write_plan_lock`); the content hash does not. A credential handle names a role, never an environment, so one lock serves every environment. `explain()` validates
handles like `prepare()`, so a CI lock check registers a placeholder credential (8 characters or longer, with the
required fields) under each declared handle. The lock and the content hash carry handle names, never credential values;
a credential value in an option is refused at planning. See
[Declaring a credential handle](named-data-access-handles.md#declaring-a-credential-handle).

## Structure hash

`plan_structure_hash(plan)` (from `mloda.steward`) is the sha256 hex of the lock text, so an equal hash means an equal lock file. A session with extenders carries it as `PlanContext.structure_hash`. Like the lock it excludes option values (except the handle names under `credential_handles`) and data access values (a user-named connection handle appears in `leaves_connection`, so renaming one changes the lock; auto handles never appear), so it is a plan-shape fingerprint, not a reproducibility or audit fingerprint. It includes the reason text and the lock format number, so it can change between mloda releases. The churn sources below apply.

`plan_content_hash(plan)` (from `mloda.steward`) / `PlanContext.content_hash` is the wider audit fingerprint: it includes each compute step's `implementation_hash` and adds each compute step's group option values (a mapping under a source group's name, a pointer, contributes its key names only, because the plan step snapshot redacts its values; every other value is hashed in full, mappings included, where values under credential-shaped keys are masked at any depth; a secret under another key name or inside a URL therefore enters the hash input, which is a sha256 over the whole plan), and drops the keys marked "no" in the tables above and the lock format number. A `connection_handle` option holding an auto handle hashes as `<unnamed>`, so the order of an unnamed collection does not change it. It excludes context options, the data access value, a link's asof config and discriminators, ids, per-run tokens and `FeatureGroup.version()` (it embeds the mloda version, so an unchanged plan would change on every upgrade; use `HookContext.feature_group_version` for that). Option values render through their repr, so an option object without a stable repr makes it unstable.

## Churn sources

- A reason can change without a framework change, for example `saves 1 conversion` to `saves 2 conversions`.
- Moving a class to another module changes its path.
- Classes created at runtime (for example via `DynamicFeatureGroupCreator`) get a module path that does not say where they came from (`abc:<ClassName>`), so two with the same class name share one path.
- Classes defined in `__main__` are refused by `write_plan_lock`, as is an implementation whose `implementation_hash` is `unavailable` (its source cannot be read).
- An edit to first-party code an implementation reaches (`mloda_plugins` or user code, its group included) changes `implementation_hash`. A dependency or mloda upgrade alone does not.
- A mapping option value other than a plain dict (for example `OrderedDict` or `MappingProxyType`) hashes as a plain dict in `plan_content_hash`, so its content hash changed once when masking started to walk it.

A corrupt lock file raises `json.JSONDecodeError`.
