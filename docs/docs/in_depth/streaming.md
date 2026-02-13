## Streaming with `stream_all`

`stream_all` lets you consume results incrementally instead of waiting for every feature group to finish.  Each time a feature group completes, its result is yielded immediately so you can begin processing it while the remaining groups are still computing.  It accepts the same parameters as `run_all` (see [mloda API](mloda-api.md)).

``` python3
from mloda.user import stream_all

for result in stream_all(["FeatureA", "FeatureB", "FeatureC"]):
    print(result)
```

## Comparison with `run_all`

`run_all` returns all results at once after every feature group has finished:

``` python3
from mloda.user import mloda

results = mloda.run_all(["FeatureA", "FeatureB", "FeatureC"])
# results is a list — all groups are already complete
```

`stream_all` yields each result as soon as its group is done:

``` python3
from mloda.user import stream_all

for result in stream_all(["FeatureA", "FeatureB", "FeatureC"]):
    # each result arrives as its feature group finishes
    process(result)
```

Both produce the same data — `list(stream_all(...))` is equivalent to `run_all(...)`.

## What "streaming" means here

Streaming operates at **feature-group granularity**.  Each yielded value is a **complete** result table for one feature group (e.g. a `pa.Table`).  There is no partial or incremental data within a single group; you always receive the full result.

## What is not supported

- **Row-by-row streaming** — individual rows are not yielded as they are computed.
- **Partial results** — you cannot observe a feature group's output before it has fully completed.
- **Chunked input** — a single feature group's computation is not split into smaller streaming chunks.

## When to use `stream_all`

`stream_all` is most useful when you request **many independent feature groups** and want to start processing early results while the rest are still running.  Typical scenarios:

- Displaying partial dashboards or reports as data becomes available.
- Feeding completed features into a downstream pipeline without waiting for all groups.
- Reducing peak memory when results can be consumed and released one at a time.

If you only request a single feature group, `stream_all` behaves like `run_all` since there is only one result to yield.

## Continuous data processing

mloda processes data in **blocks** — complete datasets that are computed and returned at once. Continuous processing means wrapping an outer loop around mloda's APIs, feeding micro-batches from an external source (Kafka consumer, WebSocket, file watcher, generator) into mloda one block at a time.

All patterns below build on the [two-phase execution API (`prepare()` / `run()`)](mloda-api.md#two-phase-execution-prepare-run).

## Pattern 1 — Micro-batch loop with `prepare()` / `run()`

The recommended pattern for continuous processing. Call `prepare()` once to build the execution plan, then loop `run()` for each micro-batch. The plan is reused across calls, so only the first call pays the planning cost.

``` python3
from mloda.user import mloda

def data_source():
    """Yields micro-batches from any source (Kafka, WebSocket, file watcher, generator)."""
    yield {"SensorData": {"timestamp": [1, 2], "value": [10.5, 11.2]}}
    yield {"SensorData": {"timestamp": [3, 4], "value": [9.8, 10.1]}}

# Prepare once with a representative schema
session = mloda.prepare(
    ["ProcessedSensor"],
    api_data={"SensorData": {"timestamp": [0], "value": [0.0]}},
)

# Process micro-batches
for batch in data_source():
    results = session.run(api_data=batch)
    for result in results:
        consume(result)
```

This works with any iterable source — swap `data_source()` for a Kafka consumer, an async WebSocket buffer, or a directory watcher.

## Pattern 2 — Micro-batch loop with `prepare()` / `stream_run()`

Combines plan reuse with per-group streaming. Call `prepare()` once, then `stream_run()` for each micro-batch. Each feature group's result is yielded as soon as it completes, while the execution plan is reused across calls.

``` python3
from mloda.user import mloda

def sensor_source():
    yield {
        "Sensors": {
            "timestamp": [1, 2, 3],
            "temperature": [22.1, 22.5, 23.0],
            "pressure": [1013, 1012, 1014],
            "vibration": [0.01, 0.02, 0.015],
        }
    }

features = ["TemperatureStats", "PressureAnomaly", "VibrationFFT"]

session = mloda.prepare(
    features,
    api_data={"Sensors": {"timestamp": [0], "temperature": [0.0], "pressure": [0], "vibration": [0.0]}},
)

for batch in sensor_source():
    for result in session.stream_run(api_data=batch):
        # Each panel updates as its feature group completes
        dashboard.update_panel(result)
```

`stream_run()` has the same parameters as `run()`. `list(session.stream_run(...))` produces the same results as `session.run(...)`.

## Pattern 3 — One-shot streaming with `stream_all()`

When you don't need plan reuse and want a single-call streaming API, use `stream_all()`. It internally calls `prepare()` + `stream_run()`.

``` python3
from mloda.user import stream_all

for result in stream_all(["FeatureA", "FeatureB", "FeatureC"], api_data=batch):
    process(result)
```

`stream_all()` rebuilds the execution plan on every call. For repeated calls, prefer Pattern 2.

## Choosing between patterns

| | Pattern 1: `prepare()` / `run()` | Pattern 2: `prepare()` / `stream_run()` | Pattern 3: `stream_all()` |
|---|---|---|---|
| **Plan reuse** | Yes | Yes | No — rebuilt every call |
| **Per-batch overhead** | Low | Low | Higher (planning repeated) |
| **Result delivery** | All groups finish, then list returned | Each group yielded as it completes | Each group yielded as it completes |
| **Best for** | High-throughput pipelines, single feature group | Multi-feature dashboards with plan reuse | Quick one-shot streaming |
| **Memory** | All results in memory at once per batch | Results can be released incrementally | Results can be released incrementally |

## Capabilities and limitations

The table below compares what mloda's block-based streaming achieves versus true row-by-row streaming frameworks (e.g. Flink, Kafka Streams).

| Aspect | mloda (block-based) | True row-by-row streaming |
|---|---|---|
| **Granularity** | Micro-batch (N rows per block) | Single row / event |
| **Latency** | Batch interval + compute time | Sub-millisecond per event |
| **State management** | Stateless between batches | Built-in windowing, sessionization |
| **Back-pressure** | Manual (control batch size / loop rate) | Framework-managed |
| **Fault tolerance** | Retry at batch level (re-run failed batch) | Per-event acknowledgment, exactly-once semantics |
| **Plan reuse** | Yes, via `prepare()` / `run()` | N/A (continuous operators) |
| **Feature composition** | Full mloda feature graph per batch | Limited to streaming operator algebra |

mloda is not a replacement for dedicated streaming frameworks. It is designed for scenarios where you want mloda's feature composition and dependency resolution applied to data that arrives continuously in small batches — for example, training a model on batch data and then reusing the same feature definitions for realtime inference, or powering live dashboards that update as new micro-batches arrive.
