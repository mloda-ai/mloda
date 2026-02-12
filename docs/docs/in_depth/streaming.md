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
