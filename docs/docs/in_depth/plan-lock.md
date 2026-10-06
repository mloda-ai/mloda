# Plan Lock

## What it records and why

Strict mode and `PluginPolicy` decide which classes may take part in a request. The plan lock records which class wins per step (feature group, compute framework, reader, specialization, joins, framework transforms) and fails when that changes. See [Plugin Registry](plugin_registry.md) for the policy side.

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

Sorted JSON with a `format` number, the requested feature names, and one record per compute, join and transform step, as class paths (`module:QualName`). Compute records include the framework choice reason (for example `pinned`) and the `result_framework` the requested features come back in, which follows the `output_framework` option when set. Duplicate steps keep one record each.

It never holds option values, `data_access_identity`, versions, per-run ids or the mloda version.

## What it does not cover

- Code changes inside the same class. Use `FeatureGroup.version()` on `HookContext`, see [Feature Group Version](feature-group-version.md).
- Data.
- Extenders.

## Structure hash

`plan_structure_hash(plan)` (from `mloda.steward`) is the sha256 hex of the lock text, so an equal hash means an equal lock file. A session with extenders carries it as `PlanContext.structure_hash`. Like the lock it excludes option values, data access, the input wiring between steps and join keys (the lock records classes and frameworks, not which inputs a feature reads or which columns a link joins on), so it is a plan-shape fingerprint, not a reproducibility or audit fingerprint. It includes the reason text and the lock format number, so it can change between mloda releases. The churn sources below apply.

`plan_content_hash(plan)` (from `mloda.steward`) / `PlanContext.content_hash` is the wider audit/lineage fingerprint: it adds each compute step's group option values (credential-shaped values scrubbed first, so a rotated secret does not change it) and input wiring, and each join's keys, and drops the reason text and lock format number. It excludes context options, data access, a link's asof config and discriminators, ids and per-run tokens, and `FeatureGroup.version()` (that embeds the mloda version, so an unchanged plan would change on every upgrade; combine it with `HookContext.feature_group_version` if needed). Option values render through their repr, so an option object without a stable repr makes it unstable.

## Churn sources

- A reason can change without a framework change, for example `saves 1 conversion` to `saves 2 conversions`.
- Moving a class to another module changes its path.
- Classes created at runtime (for example via `DynamicFeatureGroupCreator`) get a module path that does not say where they came from (`abc:<ClassName>`), so two with the same class name share one path.
- Classes defined in `__main__` are refused by `write_plan_lock`.

A corrupt lock file raises `json.JSONDecodeError`.
