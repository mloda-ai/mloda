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

Sorted JSON with a `format` number (there is no reader entry), the requested feature names, and one record per compute, join and transform step, as class paths (`module:QualName`). Compute records include the framework choice reason (for example `pinned`) and the `result_framework` the requested features come back in, which follows the `output_framework` option when set. Compute records also hold the serving `implementation` (`module:qualname`, `neutral` for a format group loaded without one, or null), its `implementation_hash` (the hash of the first-party code the implementation reaches, its group included, never dependency versions; null without an implementation) and `implementation_specialized_from` (the winner's concrete `Implementation` ancestors that serve the step's pair), and each output feature's input names (`input_feature_edges`), join records the link's `join_keys` (`left=right` column pairs, null for append/union). Duplicate steps keep one record each. A `format` bump makes a lock file written by an older mloda fail `check_plan_lock` until rewritten with `write_plan_lock`.

It never holds option values, `data_access_identity`, feature group or implementation versions, per-run ids or the mloda version. An overridden `Implementation.version()` does not reach the lock, which reads the hash.

## What it does not cover

- Code changes inside a feature group class. An implementation's code is covered by `implementation_hash`. For groups use `FeatureGroup.version()` on `HookContext`, see [Feature Group Version](feature-group-version.md).
- The neutral path: a group's own `load_neutral` is not hashed. Use `HookContext.feature_group_version`.
- Data.
- Extenders.

## Structure hash

`plan_structure_hash(plan)` (from `mloda.steward`) is the sha256 hex of the lock text, so an equal hash means an equal lock file. A session with extenders carries it as `PlanContext.structure_hash`. Like the lock it excludes option values and data access, so it is a plan-shape fingerprint, not a reproducibility or audit fingerprint. It includes the reason text and the lock format number, so it can change between mloda releases. The churn sources below apply.

`plan_content_hash(plan)` (from `mloda.steward`) / `PlanContext.content_hash` is the wider audit fingerprint: it includes each compute step's `implementation_hash` and adds each compute step's group option values (a group option whose value is itself a mapping contributes its key names only, because the plan step snapshot redacts its values; every other value is hashed in full, including a mapping nested inside a list or tuple, where values under credential-shaped keys are masked at any depth, and a `RegisteredCredential` hashes through its redacted repr, so a rotated secret does not change it; a secret under another key name or inside a URL therefore enters the hash input, which is a sha256 over the whole plan), and drops the reason text and lock format number. It excludes context options, the data access value, a link's asof config and discriminators, ids, per-run tokens and `FeatureGroup.version()` (it embeds the mloda version, so an unchanged plan would change on every upgrade; use `HookContext.feature_group_version` for that). Option values render through their repr, so an option object without a stable repr makes it unstable.

## Churn sources

- A reason can change without a framework change, for example `saves 1 conversion` to `saves 2 conversions`.
- Moving a class to another module changes its path.
- Classes created at runtime (for example via `DynamicFeatureGroupCreator`) get a module path that does not say where they came from (`abc:<ClassName>`), so two with the same class name share one path.
- Classes defined in `__main__` are refused by `write_plan_lock`, as is an implementation whose `implementation_hash` is `unavailable` (its source cannot be read).
- An edit to first-party code an implementation reaches (`mloda_plugins` or user code, its group included) changes `implementation_hash`. A dependency or mloda upgrade alone does not.
- A mapping option value other than a plain dict (for example `OrderedDict` or `MappingProxyType`) hashes as a plain dict in `plan_content_hash`, so its content hash changed once when masking started to walk it.

A corrupt lock file raises `json.JSONDecodeError`.
