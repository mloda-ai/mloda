## Overview

The **Extender** class is an abstract base class (ABC) that provides an extensible framework for enhancing and wrapping functions with additional capabilities. It is especially useful for automating and monitoring various operations such as metadata harvesting, messaging integration, and event logging. This class offers a standardized approach to augmenting functions with critical features like performance monitoring, audit trails, and impact analysis.

In the following example, we will reuse the previous feature group example and demonstrate how to monitor the execution time of the **calculate_feature** function using a custom extender.

**Monitoring Execution Time**

We will create a DokuExtender class to monitor and log the time taken for the calculate_feature function of the feature group to execute.
#### 1. Define the Extender
```python
from typing import Any
import time
from mloda.steward import Extender, ExtenderHook
import logging

logger = logging.getLogger(__name__)
```

A simple DokuExtender class:

```python
class DokuExtender(Extender):
    def wraps(self) -> set[ExtenderHook]:
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

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        record_metric(...)  # if this raises, the calculation still succeeds
        return result
```

Only the extender's own failure is caught: an exception raised by the wrapped function (or by a downstream breaking extender) always propagates, and the wrapped function is never run twice. Concrete extenders can also expose the flag as a constructor argument (for example `MetricsExtender(raise_on_error=True)`) to let callers opt back into breaking behavior.

Once the wrapped call has run, an extender's return value is discarded: it can observe or fail the calculation but not substitute a different result. An extender that never calls the wrapped function has no wrapped result to prefer, so its own return value is used instead (this is what keeps a gate's raise-to-refuse or fallback pattern below working).

A gate (an authorization or identity check that refuses a call by raising) sets `never_fall_back = True` in its class body or `__init__`: its failure always propagates, whatever `raise_on_error` says, and the wrapped function never runs as a fallback. Give a gate a `priority` value strictly lower than every other extender's on its hook, so it sorts outermost (equal values sort in no fixed order); an outer extender that catches errors from `func` could otherwise swallow the refusal. Registry strict mode (`"strict"`, see [Plugin Registry](../in_depth/plugin_registry.md#strict-mode)) drops unregistered extenders, gates included.

#### 5. Reading call facts via HookContext

`HookContext.current()` returns the `HookContext` for the hook being dispatched (`FEATURE_GROUP_CALCULATE_FEATURE`, `VALIDATE_INPUT_FEATURE`, `VALIDATE_OUTPUT_FEATURE`, `FEATURE_GROUP_MATCHED`, `INPUT_DATA_LOAD`, `JOIN`), or `None` outside a hook call. It is set only on the thread dispatching the hook: capture it before handing work to another thread, or run that work under `contextvars.copy_context().run`. It carries which hook fired, the feature group's `module.qualname` and `version()`, the owning plugin's installed distribution version, the requested feature names, the input feature names the engine resolved for the step at planning time (`None` for a root step; an injected filter or index feature does not break this, since the step inherits its host's resolved names; a best-effort `input_features()` re-read only for a `FeatureSet` assembled outside an execution plan), `input_feature_edges` (a `dict[str, tuple[str, ...]]` mapping each output feature name to its declared input names; `None` for a root step; a feature declaring no inputs is absent; same-named features union-merge into one entry keyed by name; injected features are absent when the engine resolved the step, but a runtime re-read, i.e. a `FeatureSet` assembled outside a plan or after option defaults materialize, may include them; the mapping is copied on construction like `carrier`), the compute framework's class name, and `rows_in` (`None` when the framework can't report a count without materializing, e.g. a lazy or SQL-backed frame). `rows_in` is the framework's current data, so on `VALIDATE_OUTPUT_FEATURE` it is the freshly calculated output, not the pre-calculation input.

Read it *after* calling `func(*args, **kwargs)` in your own `__call__` to also see `rows_out`, `output_schema`, `duration_seconds`, and `status`. `status` reflects only the wrapped call: `"success"` once it returns, `"error"` if it raised; a warning-only extender's own failure does not change it. `output_schema` is `tuple[tuple[str, str | None], ...] | None`: `(column, dtype)` pairs sorted by name, dtype a framework-specific human-readable string or `None` if that column's dtype could not be read (the name is still reported). Populated best-effort from the raw `calculate_feature` result on `FEATURE_GROUP_CALCULATE_FEATURE`, without materializing a lazy or SQL-backed result. A dict result (column to values, the interchange shape accepted before transform) is read the same way on every framework, with the Python type name of the first non-`None` value as dtype. On `VALIDATE_OUTPUT_FEATURE`, since the hook's return value carries no schema, it's read from the framework's finalized `self.data` (the native, post-transform/post-filter shape) the same way `rows_in` is, but only once `validate_output_features` returns without raising; `rows_in` is captured before that call, so it stays populated even when the call raises. `None` on `VALIDATE_INPUT_FEATURE` and every other hook, and also when the read data is in neither the dict nor the framework's native shape or has no columns, or when reading it fails. On `VALIDATE_INPUT_FEATURE`/`VALIDATE_OUTPUT_FEATURE`, `rows_out` stays `None`, since those hooks return no data. `func` is always an instrumentation wrapper around the feature group's method, not the bound classmethod: use `Extender.feature_group_name(func)` or `inspect.unwrap(func)` to reach the original. `tenant_id`, `project_id`, and `principal` carry the server-verified values set via `mloda.steward.verified_context()` for the scope of a run, `None` when nothing set them, and are never influenced by a feature's `Options`. Like `HookContext.current()`, that scope is thread/task-local: a caller handing the actual `prepare()`/`run()`/`run_all()`/`stream_run()`/`stream_all()` call to another thread must copy context (`contextvars.copy_context().run(...)`) for it to still apply there. `carrier` is an opaque `dict[str, str] | None` forwarded from the run call; core never interprets it. Each `HookContext` copies `carrier` on construction, so a hook mutating it never affects another hook or the run context, except when multiple extenders are composed on the same hook via `CompositeExtender`, which share one `HookContext` instance (and thus its carrier) by design. `worker_index` is `int | None`, set only inside a spawned MULTIPROCESSING worker.

`FEATURE_GROUP_MATCHED`, `INPUT_DATA_LOAD`, and `JOIN` populate their own extra fields and leave the rest at their defaults. On `INPUT_DATA_LOAD`, `data_access_identity` and `data_access_format` carry a string identity and format for the data-access handle/value (a URI identity keeps scheme, host and path, dropping user info, query and fragment, except an `abfs`/`abfss`/`wasb`/`wasbs` container, scheme matched case-insensitively; percent-encode `/`, `?` and `#` in user info; the path is also cut at an encoded `?`/`#` whose decoded tail holds a recognized or secret key; a scheme-less keyword or ODBC connection string is identified by its recognized connection and secret key names (so strings with the same keys share an identity); `user:pw@host` unconditionally drops query and fragment along with user info; without userinfo, a scheme-less string's own query or fragment is dropped only when it holds a recognized or secret key, otherwise left as is; an encoded `?`/`#` anchor and the query/fragment are percent-decoded once before that key scan; not detected: double encoding, an encoded key/value pair with no `?`/`#` anchor, and a secret carried as a value under an unrecognized key; a mapping is identified by its sorted key names, a path by its filesystem path, any other non-string value by its type name only); `data_access_dataset_version` stays `None`, since no dataset versioning exists yet. On `JOIN`, `join_type` and `join_keys` come from the `Link` being merged. On `FEATURE_GROUP_MATCHED`, `plan_feature_count`, `plan_node_count`, and `plan_depth` are running counts and recursion depth at match time, not final totals for the whole plan.

`FEATURE_GROUP_MATCHED` reads the `mloda.steward.verified_context()` scope active while a session is planned (for example `prepare()`, `explain()`, `diagnose()`, and the planning half of `run_all()`/`stream_all()`). Every other hook reads the scope active at run-call time (`run()`, `stream_run()`, and the run half of `run_all()`/`stream_all()`; for the stream calls that is creation, not first iteration).

A session prepared under one scope and run under another (or none) therefore reports each hook its own phase's identity, so a match-time gate authorizes the preparer, not the runner. Wrap `prepare()` and `run()` in one scope or use `run_all()`/`stream_all()`; a server that prepares once and runs per tenant should gate on a run-time hook such as `FEATURE_GROUP_CALCULATE_FEATURE`. Identity is not snapshotted at prepare, so a run outside any scope never inherits the preparer's identity.

`explain()` and `diagnose()` let an exception a breaking extender (`raise_on_error = True`, or `never_fall_back = True`) raises propagate, for example a refusal at `FEATURE_GROUP_MATCHED`; `diagnose()` still projects it when it is one of the error types it projects. An extender with `raise_on_error = False` and no `never_fall_back` is logged and swallowed instead, so resolution falls back.

```python
from mloda.steward import Extender, ExtenderHook, HookContext

class FactsExtender(Extender):
    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        if context is not None:
            logger.info(f"{context.feature_group_class}: {context.duration_seconds}s, {context.rows_out} rows out")
        return result
```

To unit-test an extender's `__call__` without running the engine, build a `HookContext` directly and `activate()` it. Only `hook`, `feature_group_class`, `feature_group_version` and `compute_framework_name` are required. `plugin_version`, `input_features` and `input_feature_edges` default to `None` (a mapping passed in is copied, like `carrier`), `feature_names` defaults to an empty tuple, and every other field defaults to `None`.

```python
from mloda.steward import ExtenderHook, HookContext

context = HookContext(
    hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
    feature_group_class="my_plugin.MyFeatureGroup",
    feature_group_version="1",
    compute_framework_name="PyArrowTable",
)

with context.activate():
    FactsExtender()(lambda: "result")
```

#### 6. Discovering Extenders

To list all available extenders and their documentation, use the `get_extender_docs()` function from `mloda.steward`.

#### 7. Extender identity under `ParallelizationMode.MULTIPROCESSING`

A framework that stays resident in the parent process uses the caller's own extender objects, not copies: identity and mutable state on an extender (a client, tracer, connection) hold across every framework built there, including one whose own resolved `parallelization_mode` is not `MULTIPROCESSING` inside an overall MULTIPROCESSING-enabled run (e.g. a compute framework that only supports `SYNC`). For that SYNC-only framework to run at all, the requested `parallelization_modes` set must also contain the framework's own supported mode, e.g. `{ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING}`; requesting `{ParallelizationMode.MULTIPROCESSING}` alone filters the framework out at setup and the run raises.

A parent-resident extender is one shared object, invoked by every framework built there, potentially concurrently under `ParallelizationMode.THREADING`. It must be thread-safe.

For a framework whose class declares `MULTIPROCESSING` support, a framework dispatched to a spawned worker gets an independent copy, materialized only when the framework is unpickled inside the worker: the parent only ever holds the pickled snapshot taken at run start. The extender must be picklable at that point: a live handle that pickle can't round-trip (a client, tracer, connection, lock) must be stripped in `__getstate__` and rebuilt in `__setstate__` or on first use in `__call__`. Constructing the handle eagerly in `__init__` is fine as long as `__getstate__` strips it before pickling. State a worker copy accumulates stays in that worker unless the extender ships it out itself (a file, a socket, an exporter).

Two copies in the same process that resolve to the same underlying handle (e.g. a pooled client) share its lifecycle, not two independently rebuilt handles: an extender exposing `close()` must make the closed state visible to every copy resolving to that handle, or closing one leaves the others writing to a dead handle.

An extender that accepts an injected sink (client, provider, connection) should trial-pickle it in `__getstate__` and drop it if it can't survive: the run then completes with the sink dropped, instead of being rejected outright by mloda's plan-time picklability check. Log the drop once per instance. `mloda.steward` ships two small primitives for this: `pickle_failure_reason(value)` (`None` if `value` pickles cleanly, else the caught exception's type name) and `WarnOncePerInstance`, a thread-safe double-checked-locking guard whose `warn_once(emit)` fires `emit` at most once per instance and always resets to unwarned on any copy, so a pickled worker copy decides independently whether to warn instead of inheriting the parent's already-fired state:

```py
import logging
from typing import Any

from mloda.steward import Extender, pickle_failure_reason, WarnOncePerInstance

logger = logging.getLogger(__name__)

class MyExtender(Extender):
    def __init__(self, sink: Any = None) -> None:
        self._sink = sink
        self._drop_guard = WarnOncePerInstance()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        reason = pickle_failure_reason(self._sink) if self._sink is not None else None
        if reason is not None:
            self._drop_guard.warn_once(lambda: logger.warning(f"dropping unpicklable sink ({reason})"))
            state["_sink"] = None
        return state
```

No custom `__setstate__` is needed for `_drop_guard` itself: pickle reconstructs it as a fresh, unwarned guard on its own. Plan-time picklability validation and worker-dispatch pickling both pickle the same live extender instance, so this drop-and-warn fires once, in the parent, during plan-time validation; the later dispatch pickle is then silent, since the guard has already fired for that instance.

#### 8. Knowing when a run is finished

`on_run_complete(run_id)` is a no-op hook called once per `run()`, `run_all()`, `stream_run()` or `stream_all()` call that got as far as setting up execution (a stream does on its first iteration; a stream closed early fires after its workers are joined). It fires when setup (e.g. the MULTIPROCESSING picklability preflight) or execution raised, so it is not a success signal. It does not fire for prepare, explain, a never-iterated stream, a failure while planning before setup, or when finalizing raised (collecting artifacts, joining or terminating the workers). It runs in the parent, on the caller's own extender objects that the run used (registry strict mode drops unregistered ones, see section 4), after all workers were joined, in every parallelization mode; `close()`, by contrast, runs only on a MULTIPROCESSING worker's copy. A worker that does not exit within `graceful_shutdown_timeout` (one deadline for all workers) is terminated, possibly mid-`close()`, and the hook still fires: a sink whose records must not be lost should flush synchronously, not only in `close()`.

`run_id` is the session's id, the same value as `HookContext.run_id` in the per-calculation hooks, so records can be correlated; re-running a prepared session fires again with the same id. `HookContext.current()` is `None` inside this hook. Extenders are notified in ascending `priority` order (ties in no fixed order), even if `wraps()` returns nothing. The hook is synchronous with no time budget: a blocking hook blocks the caller. An `Exception` raised in it is logged and never propagated, `raise_on_error` and `never_fall_back` do not apply, and the remaining extenders are still notified.

```python
from typing import Any

from mloda.steward import Extender, ExtenderHook

class SealingExtender(Extender):
    def __init__(self) -> None:
        self.completed_run_ids: list[str | None] = []

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def on_run_complete(self, run_id: str | None) -> None:
        self.completed_run_ids.append(run_id)
```
