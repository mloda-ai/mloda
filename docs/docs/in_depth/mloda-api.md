## mlodaAPI

The **mlodaAPI** class serves as the primary interface for interacting with the core system, streamlining the setup and execution of computational workflows

#### Major components (and steps)

1. **Configuration**

    Initialize the mloda with features, compute frameworks, and other settings to configure your environment.

2.  **Engine Setup**

    Create an execution plan by setting up the engine, which orchestrates the computation based on the defined features and configurations.

3.  **Runner Setup**

    Prepare the runner that will execute the plan, ensuring that all components are ready for computation.

4.  **Run Engine Computation**

    Apply the runner to execute the computations based on the prepared execution plan, managing the lifecycle of the computational process.

This means, depending on your needs, you can run them all at once (**batch run**) or split them up, e.g. for realtime needs (**inference**).  For incremental result delivery see [Streaming](streaming.md).

#### Configuration for mlodaAPI

-   **requested_features**: Specify the features to process (as names, Feature objects, or a Features container).
-   **compute_frameworks** (optional): Limit the compute frameworks using framework types or names.
-   **links** (optional): Define dataset merging links with Link objects.
-   **data_access_collection** (optional): Provide data sources for feature identification.

#### Runner & Execution Configuration

-   **function_extender** (optional): Add function extenders to customize computations.
-   parallelization_modes (optional): Choose between sync, threading, or multiprocessing modes. (Default: sync)
-   flight_server (optional): Specify a flight server for multiprocessing only.
-   **column_ordering** (optional): Control the ordering of result columns.
    Accepts `"alphabetical"` (sort columns A-Z) or `"request_order"`
    (preserve the order features were requested). Default: `None` (no guaranteed order).

``` python
from mloda.user import mloda

# Alphabetical ordering
result = mloda.run_all(
    ["FeatureC", "FeatureA", "FeatureB"],
    column_ordering="alphabetical"  # Result columns: FeatureA, FeatureB, FeatureC
)

# Preserve request order
result = mloda.run_all(
    ["FeatureC", "FeatureA", "FeatureB"],
    column_ordering="request_order"  # Result columns: FeatureC, FeatureA, FeatureB
)
```

#### Two-Phase Execution: prepare() + run()

For realtime or inference scenarios, split configuration from execution.
`prepare()` builds the execution plan once; `run()` executes it with fresh data each time.

``` python
from mloda.user import mloda

# 1. Prepare once
session = mloda.prepare(
    ["MyFeature"],
    compute_frameworks=["PandasDataFrame"],
    api_data=initial_api_data,
)

# 2. Run multiple times with different data
result_1 = session.run(api_data={"MyKey": {"col": [1, 2]}})
result_2 = session.run(api_data={"MyKey": {"col": [3, 4]}})
```

`run()` also accepts an `artifacts` parameter for switching between artifact
save and load modes across calls. See [Artifacts](artifacts.md#run-time-artifact-switching-with-preparerun) for details.

`run_all()` is equivalent to `prepare()` followed by a single `run()`.
`stream_all()` is equivalent to `prepare()` followed by a single `stream_run()`.

For per-group streaming with plan reuse, call `session.stream_run()` instead of `session.run()` -- it yields each feature group's result as it completes. See [Streaming](streaming.md) for details.

#### Plugin Discovery

mloda provides functions to discover and inspect available plugins. Import them from `mloda.steward`:

```python
from mloda.steward import (
    get_feature_group_docs,
    get_compute_framework_docs,
    get_extender_docs,
    resolve_feature,
)
```

##### resolve_feature

Resolve a feature name to its matching FeatureGroup class. This is useful for debugging feature resolution or understanding which FeatureGroup handles a specific feature.

``` python
from mloda.steward import resolve_feature

# Successful resolution
result = resolve_feature("my_feature_name")
if result.feature_group:
    print(f"Resolved to: {result.feature_group.__name__}")
else:
    print(f"Error: {result.error}")

# Access all matching candidates (before subclass filtering)
print(f"Candidates: {[fg.__name__ for fg in result.candidates]}")
```

**Parameters:**

- **feature_name** (`str`): The name of the feature to resolve.
- **options** (`Options | None`, keyword-only): Options used for matching and for the compute framework capability split. Defaults to empty `Options`. Required to resolve FeatureGroups that gate matching on an option.
- **plugin_collector** (`PluginCollector | None`, keyword-only): Restricts the FeatureGroups considered, and threads its `allow_redefinition` flag into deduplication.

**Returns:** `ResolvedFeature` dataclass with fields:

- **feature_name** (`str`): The input feature name.
- **feature_group** (`Type[FeatureGroup] | None`): The resolved FeatureGroup class, or None if resolution failed.
- **candidates** (`List[Type[FeatureGroup]]`): All FeatureGroups that matched before subclass filtering.
- **error** (`str | None`): Error message if resolution failed (no match or multiple conflicts).

##### explain and resolved_plan

The runtime counterpart to `resolve_feature`: `mlodaAPI.explain(...)` builds the execution plan for a request without running it, and `session.resolved_plan()` returns the same records for a prepared session (before or after `run()`). Both return a `list[PlanStep]` in execution-plan order. Every `explain` parameter after `features` is keyword-only.

`explain` re-resolves the plan from scratch. It answers "what would this request resolve to", it is not a record of a prior `run_all` execution. For the plan of a run that actually happened, use the return value directly: `run_all` returns a `RunResult` (a `list` with a read-only `plan` property) and `stream_all` returns a `ResultStream` (generator-compatible, `plan` available before consuming). One planning pass serves both the results and the plan, unlike `explain`, which re-resolves.

``` python
from mloda.user import mloda

results = mloda.run_all(["sales__mean_aggr"], compute_frameworks=["PandasDataFrame"])
for step in results.plan:
    print(step.step_kind, step.feature_names)
``` To match a `run_all` resolution, pass the same `parallelization_modes`: `run_all` defaults to `{ParallelizationMode.SYNC}`, `prepare`/`explain` default to `None`, and compute frameworks are filtered by mode.

``` python
from mloda.user import mloda

# "sales" here stands for a root feature one of your own FeatureGroups provides.
for step in mloda.explain(["sales__mean_aggr"], compute_frameworks=["PandasDataFrame"]):
    print(step.step_kind, step.feature_names, step.feature_group_name, step.compute_framework_name)
```

**Returns:** `PlanStep` dataclass (frozen) with fields:

- **step_kind** (`Literal["compute", "join", "transform"]`).
- **feature_names** (`tuple[str, ...]`): Features computed by a compute step, empty otherwise. This includes engine-injected features (link index features, global-filter features), so it is not a clean "requested feature name -> FeatureGroup" map: the same index name can appear under two different FeatureGroups in one plan.
- **feature_group** (`type[FeatureGroup] | None`): Resolved FeatureGroup; the destination for a transform step; the link's declared left side for a join.
- **compute_framework** (`type[ComputeFramework] | None`): Selected ComputeFramework; the destination for a transform step; the merge destination for a join.
- **source_feature_group** / **source_compute_framework**: Origin of a transform step. For a join: the link's declared right side, and the framework merged in.
- **join_type** (`str | None`): The link's join type (`"inner"`, `"left"`, ...) for a join step, None otherwise.
- **feature_group_name** / **compute_framework_name** / **source_feature_group_name** / **source_compute_framework_name** (`str | None`): Class names of the above, None when unset.

Join semantics, honestly: for a join step the `*_feature_group` fields are the link's declared left/right sides, while `compute_framework`/`source_compute_framework` are the merge destination and the framework merged in. The planner swaps those two frameworks for `RIGHT` joins, so they are not guaranteed to be the frameworks of the declared left/right sides.

##### get_feature_group_docs

Get documentation for feature groups with optional filtering.

``` python
from mloda.steward import get_feature_group_docs

# Get all feature groups
all_fgs = get_feature_group_docs()

# Filter by name
fgs = get_feature_group_docs(name="timestamp")

# Filter by compute framework
fgs = get_feature_group_docs(compute_framework="PandasDataframe")
```

**Parameters:**

- **name** (`str`, optional): Filter by name (case-insensitive partial match).
- **search** (`str`, optional): Search in description (case-insensitive partial match).
- **compute_framework** (`str | Type[ComputeFramework]`, optional): Filter by compute framework.
- **version_contains** (`str`, optional): Filter by version substring.

**Returns:** `List[FeatureGroupInfo]` sorted by name.

##### get_compute_framework_docs

Get documentation for compute frameworks with optional filtering.

``` python
from mloda.steward import get_compute_framework_docs

# List every framework (is_available flags whether each is importable)
frameworks = get_compute_framework_docs()

# Only frameworks whose backing library is importable
available_frameworks = get_compute_framework_docs(available_only=True)
```

**Parameters:**

- **name** (`str`, optional): Filter by name (case-insensitive partial match).
- **search** (`str`, optional): Search in description (case-insensitive partial match).
- **available_only** (`bool`, default `False`): By default all frameworks are listed (with `is_available` as the flag); set `available_only=True` to filter to importable frameworks only.

**Returns:** `List[ComputeFrameworkInfo]` sorted by name.

##### get_extender_docs

Get documentation for extenders with optional filtering.

``` python
from mloda.steward import get_extender_docs

# Get all extenders
extenders = get_extender_docs()

# Filter by wrapped function type
extenders = get_extender_docs(wraps="formula")
```

**Parameters:**

- **name** (`str`, optional): Filter by name (case-insensitive partial match).
- **search** (`str`, optional): Search in description (case-insensitive partial match).
- **wraps** (`str`, optional): Filter by wrapped function type (case-insensitive exact match).

**Returns:** `List[ExtenderInfo]` sorted by name.

