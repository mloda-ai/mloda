---
name: mloda
description: >
  Give an AI agent declarative, deterministic data access via mloda (https://github.com/mloda-ai/mloda) - a
  Python plugin framework where the caller describes WHAT features or data it needs and mloda resolves HOW to
  compute or fetch them across a plugin graph, with built-in lineage back to source. Use when an agent needs to
  request structured data or feature aggregations as a tool call instead of writing ad hoc fetch/transform code
  (the "LLM Tool Function" pattern), when assembling a multi-source context window declaratively, when chaining
  retrieval/validation/redaction steps for a RAG pipeline by feature name, or when writing/reviewing a mloda
  FeatureGroup, ComputeFramework, or Extender plugin.
license: Apache-2.0
---

# mloda: Declarative Data Access for AI Agents

## Mental model

mloda separates **WHAT** a caller needs from **HOW** it gets computed. A caller (human or LLM) requests
features by name; mloda resolves dependencies across a plugin graph and executes them on a compute framework
(`PandasDataFrame`, `PyArrowTable`, `PythonDictFramework`, ...), returning the result with lineage back to
source. You never hand-wire the pipeline - you name the end result and mloda traces back through whatever
chain of plugins produces it.

Three plugin types, most work happens in the first:

| Type | Role |
|------|------|
| **FeatureGroup** | Defines a data transformation - the unit you'll write most often |
| **ComputeFramework** | Execution backend the transformation runs on |
| **Extender** | Hooks for logging, validation, monitoring |

Roles map to modules: `mloda.provider` (define plugins), `mloda.user` (request data), `mloda.steward`
(govern execution).

## When to reach for mloda

- The agent should declare a data/feature request as a tool call instead of writing ad hoc fetch/transform
  code (see "LLM Tool Function" below).
- Multiple context sources (system prompt, user profile, retrieved docs, history) need declarative assembly
  before being handed to a model.
- A RAG pipeline needs chained steps (retrieve -> validate -> redact) expressed as a feature name or config
  instead of hand-wired code.
- You are writing or reviewing a FeatureGroup / ComputeFramework / Extender plugin, or planning one for a new
  data source.

## LLM Tool Function pattern

The agent emits a feature request as JSON, no Python required from the agent itself:

```python
from mloda.user import PluginLoader, load_features_from_config, mloda

PluginLoader.all()  # every mloda example calls this once before the first run_all()

llm_output = '''
[
    "customer_id",
    {"name": "income__sum_aggr"},
    {"name": "total_spend", "options": {"aggregation_type": "sum", "in_features": "income"}}
]
'''

features = load_features_from_config(llm_output, format="json")
result = mloda.run_all(
    features=features,
    compute_frameworks=["PandasDataFrame"],
    api_data={"SampleData": {"customer_id": ["C001", "C002"], "income": [50000, 75000]}},
)
```

`api_data` inlines data under a label (e.g. `"SampleData"`); features are matched to columns by name. For
data on disk, pass a `DataAccessCollection` instead (see mloda's
[API Request docs](https://mloda-ai.github.io/mloda/chapter1/api-request/)).

## Feature-name chaining

A chain like `documents__retrieved__pii_redacted` expresses a pipeline without wiring it by hand - if you ask
for `pii_redacted`, mloda traces that it depends on `retrieved`, which depends on `documents`, and resolves
the whole chain from the single requested name:

```python
Feature(name="user_query__injection_checked__retrieved__pii_redacted")
```

This only works when a FeatureGroup plugin exists for every step in the chain - the chaining syntax alone
does not fetch data mloda has no plugin for.

## Latency-sensitive / repeated requests

`mloda.run_all()` rebuilds the full execution plan every call, fine for batch jobs but wasteful when the same
features are recomputed per request (model serving, streaming, dashboards). Use the two-phase API to plan
once and execute cheaply per request:

```python
session = mloda.prepare(features, compute_frameworks=["PyArrowTable"], data_access_collection=data_access_collection)
result = session.run(api_data={"MyKey": {"col": [1, 2]}})
```

## Reference

- Docs: <https://mloda-ai.github.io/mloda/>
- Install: `pip install mloda`
- Source: <https://github.com/mloda-ai/mloda>
