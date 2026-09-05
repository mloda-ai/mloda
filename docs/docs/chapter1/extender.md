## Overview

The **Extender** class is an abstract base class (ABC) that provides an extensible framework for enhancing and wrapping functions with additional capabilities. It is especially useful for automating and monitoring various operations such as metadata harvesting, messaging integration, and event logging. This class offers a standardized approach to augmenting functions with critical features like performance monitoring, audit trails, and impact analysis.

In the following example, we will reuse the previous feature group example and demonstrate how to monitor the execution time of the **calculate_feature** function using a custom extender.

**Monitoring Execution Time**

We will create a DokuExtender class to monitor and log the time taken for the calculate_feature function of the feature group to execute.
#### 1. Define the Extender
```python
from typing import Set, Any
import time
from mloda.steward import Extender, ExtenderHook
import logging

logger = logging.getLogger(__name__)
```

A simple DokuExtender class:

```python
class DokuExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}
    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.error(f"Time taken: {time.time() - start}")
        return result
```
#### 2. Run the Example with the Extender
We will now run the **mlodaAPI** call, including our custom **DokuExtender** to monitor the execution time of the **calculate_feature** function.
```python
from mloda.user import mloda
from mloda.user import DataAccessCollection

file_path = "tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"
data_access_collection = DataAccessCollection(files={file_path})

feature_list = ["id","V1","V2","V3"]

example_feature_list = [f"ExampleB_{f}" for f in feature_list]


mloda.run_all(
    feature_list,
    compute_frameworks={"PyArrowTable"},
    data_access_collection=data_access_collection,
    function_extender={DokuExtender()}
)
```
Expected Output (Logged Execution Times)
```text
ERROR    test_getting_started:test_getting_started.py:29 Time taken: 0.00454258918762207
ERROR    test_getting_started:test_getting_started.py:29 Time taken: 0.001033782958984375
```

#### 3. Summary

With this simple extender, you can easily log and monitor the execution time of any functionality within feature groups. By extending the Extender class, you can wrap additional behavior such as performance monitoring, logging, or auditing around critical functions to enhance observability and traceability in your data processing workflows.

When multiple extenders are provided, they are automatically chained and executed in priority order (lower values first).

#### 4. Error handling

By default an exception raised inside an extender is **breaking**: it propagates and fails the feature calculation, just like a bug in any other code. This holds whether one extender or several are registered for a hook, so adding a second extender never changes the error semantics.

An extender that is non-critical (for example observability or telemetry) can opt out by setting `raise_on_error = False`. When such an extender fails, the error is logged as a warning and the wrapped function still runs, so a failing extender cannot break the calculation:

```python
class MetricsExtender(Extender):
    def __init__(self) -> None:
        self.raise_on_error = False  # failures log a warning instead of breaking

    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        record_metric(...)  # if this raises, the calculation still succeeds
        return result
```

Only the extender's own failure is caught: an exception raised by the wrapped function (or by a downstream breaking extender) always propagates, and the wrapped function is never run twice. Concrete extenders can also expose the flag as a constructor argument (for example `MetricsExtender(raise_on_error=True)`) to let callers opt back into breaking behavior.

#### 5. Reading call facts via HookContext

`HookContext.current()` returns the `HookContext` for the hook being dispatched (`FEATURE_GROUP_CALCULATE_FEATURE`, `VALIDATE_INPUT_FEATURE`, `VALIDATE_OUTPUT_FEATURE`, `FEATURE_GROUP_MATCHED`, `INPUT_DATA_LOAD`, `JOIN`), or `None` outside a hook call. It is set only on the thread dispatching the hook: capture it before handing work to another thread, or run that work under `contextvars.copy_context().run`. It carries which hook fired, the feature group's `module.qualname` and `version()`, the owning plugin's installed distribution version, the requested feature names, the input feature names the engine resolved for the step at planning time (`None` for a root step; a `FeatureSet` built outside an execution plan falls back to a best-effort `input_features()` re-read), the compute framework's class name, and `rows_in` (`None` when the framework can't report a count without materializing, e.g. a lazy or SQL-backed frame). `rows_in` is the framework's current data, so on `VALIDATE_OUTPUT_FEATURE` it is the freshly calculated output, not the pre-calculation input.

Read it *after* calling `func(*args, **kwargs)` in your own `__call__` to also see `rows_out`, `output_schema`, `duration_seconds`, and `status`. `status` reflects only the wrapped call: `"success"` once it returns, `"error"` if it raised; a warning-only extender's own failure does not change it. `output_schema` is `tuple[tuple[str, str | None], ...] | None`: `(column, dtype)` pairs sorted by name, dtype a framework-specific human-readable string or `None` if that column's dtype could not be read (the name is still reported). Populated best-effort from the raw `calculate_feature` result on `FEATURE_GROUP_CALCULATE_FEATURE` only, without materializing a lazy or SQL-backed result; `None` on every other hook, when no column names can be read (the result isn't yet in the framework's native shape, or has no columns), or when reading them fails. On `VALIDATE_INPUT_FEATURE`/`VALIDATE_OUTPUT_FEATURE`, `rows_out` stays `None`, since those hooks return no data. `func` is always an instrumentation wrapper around the feature group's method, not the bound classmethod: use `Extender.feature_group_name(func)` or `inspect.unwrap(func)` to reach the original. `tenant_id`, `project_id`, and `principal` carry the server-verified values set via `mloda.steward.verified_context()` for the scope of a run, `None` when nothing set them, and are never influenced by a feature's `Options`. Like `HookContext.current()`, that scope is thread/task-local: a caller handing the actual `prepare()`/`run()`/`run_all()`/`stream_run()`/`stream_all()` call to another thread must copy context (`contextvars.copy_context().run(...)`) for it to still apply there. `carrier` is an opaque `dict[str, str] | None` forwarded from the run call; core never interprets it. `worker_index` is `int | None`, set only inside a spawned MULTIPROCESSING worker.

`FEATURE_GROUP_MATCHED`, `INPUT_DATA_LOAD`, and `JOIN` populate their own extra fields and leave the rest at their defaults. On `INPUT_DATA_LOAD`, `data_access_identity` and `data_access_format` carry a string identity and format for the data-access handle/value; `data_access_dataset_version` stays `None`, since no dataset versioning exists yet. On `JOIN`, `join_type` and `join_keys` come from the `Link` being merged. On `FEATURE_GROUP_MATCHED`, `plan_feature_count`, `plan_node_count`, and `plan_depth` are running counts and recursion depth at match time, not final totals for the whole plan.

```python
from mloda.steward import Extender, ExtenderHook, HookContext

class FactsExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None:
            logger.info(f"{context.feature_group_class}: {context.duration_seconds}s, {context.rows_out} rows out")
        return result
```

#### 6. Discovering Extenders

To list all available extenders and their documentation, use the `get_extender_docs()` function from `mloda.steward`.
