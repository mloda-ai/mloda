# [mloda.ai](https://mloda.ai): Open Data Access for ML & AI

[![Website](https://img.shields.io/badge/website-mloda.ai-blue.svg)](https://mloda.ai)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue.svg)](https://mloda-ai.github.io/mloda/)
[![PyPI version](https://badge.fury.io/py/mloda.svg)](https://badge.fury.io/py/mloda)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mloda-ai/mloda/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-1500%2B-green.svg)](https://github.com/mloda-ai/mloda)

> **Declarative data access for AI agents. Describe what you need - mloda delivers it.**

```bash
pip install mloda
```

## 30-Second Example

Your AI describes what it needs. mloda figures out how to get it:

```python
from mloda.user import PluginLoader, mloda
PluginLoader.all()

result = mloda.run_all(
    features=["customer_id", "income", "income__sum_aggr", "age__avg_aggr"],
    compute_frameworks=["PandasDataFrame"],
    api_data={"SampleData": {
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "age": [25, 35, 45, 30, 50],
        "income": [50000, 75000, 90000, 60000, 85000]
    }}
)
```

Copy, paste, run. mloda resolves dependencies, chains plugins, delivers data.

---

## What mloda Does

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA USERS                                 │
│  AI Agents  •  ML Pipelines  •  Data Science  •  Analytics      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ describe what they need
                            ▼
                    ┌───────────────┐
                    │     mloda     │  ← resolves HOW from WHAT
                    │   [Plugins]   │
                    └───────────────┘
                            │ delivers trusted data
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                │
│  Databases  •  APIs  •  Files  •  Any source via plugins        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why mloda?

| You want to... | mloda gives you... |
|----------------|-------------------|
| Give AI agents data access | Declarative API - agents describe WHAT, not HOW |
| Trace every result | Built-in lineage back to source |
| Reuse across projects | Plugins work anywhere - notebook to production |
| Mix data sources | One interface for DBs, APIs, files, anything |

---

## AI Use Case: LLM Tool Function

Let LLMs request data without writing code:

```python
# LLM generates this JSON
llm_request = '["customer_id", {"name": "income__sum_aggr"}]'

# mloda executes it
from mloda.user import load_features_from_config
features = load_features_from_config(llm_request, format="json")
result = mloda.run_all(
    features=features,
    compute_frameworks=["PandasDataFrame"],
    api_data={"SampleData": {"customer_id": ["C001", "C002"], "income": [50000, 75000]}}
)
```

More patterns: [Context Window Assembly](#2-context-window-assembly) • [RAG Pipelines](#3-rag-with-feature-chaining)

---

## How mloda is Different

mloda separates **WHAT** you need from **HOW** to get it - through plugins. Existing tools solve parts of this, but none bridge the full gap:

| Category | Products | What it does | Why it's not enough |
|----------|----------|--------------|---------------------|
| Feature Stores | Feast, Tecton, Featureform | Store + serve features | Infrastructure-tied, storage-only |
| Semantic Layers | dbt Semantic Layer, Cube | Declarative metrics | SQL-only, centralized |
| DAG Frameworks | Hamilton, Kedro | Dataflows as code | Function-first, no plugin abstraction |
| Data Catalogs | DataHub, Atlan | Metadata & discovery | No execution, no contracts |
| ORMs | SQLAlchemy, Django ORM | Database abstraction | Single database, no ML lifecycle |

**mloda is the connection layer** - separating WHAT you compute from HOW you compute it. Plugins define transformations. Users describe requirements. mloda resolves the pipeline.

---

## Plugins: The Building Blocks

mloda's architecture follows three roles: **providers** (define plugins), **users** (access data), and **stewards** (govern execution). The module structure reflects this: `mloda.provider`, `mloda.user`, `mloda.steward`.

mloda uses three types of plugins:

| Type | What it does |
|------|--------------|
| **FeatureGroup** | Defines data transformations |
| **ComputeFramework** | Execution backend (Pandas, Spark, etc.) |
| **Extender** | Hooks for logging, validation, monitoring |

Most of the time, you'll work with **FeatureGroups** - Python classes that define how to access and transform data (see Quick Example above).

**Why plugins?**
- **Steps, not pipelines** - Build transformations. mloda wires them together.
- **Small and testable** - Each plugin is a focused unit. Easy to test, easy to debug.
- **AI-friendly** - Small, template-like structures. Let AI generate plugins for you.
- **Share what isn't secret** - Your pipeline runs steps a,b,c,d. Steps b,c,d have no proprietary logic? Share them across projects, teams, even organizations.
- **Experiment to production** - Same plugins in your notebook and your cluster. No rewrite.
- **Stand on shoulders** - Combine community plugins with your own. Build on what exists.

---

## AI Use Case Patterns

### 1. LLM Tool Function

Give LLMs deterministic data access - they declare what, mloda handles how:

```python
from mloda.user import PluginLoader, load_features_from_config, mloda
PluginLoader.all()

# LLM generates this JSON (no Python code needed)
llm_output = '''
[
    "customer_id",
    {"name": "income__sum_aggr"},
    {"name": "age__avg_aggr"},
    {"name": "total_spend", "options": {"aggregation_type": "sum", "in_features": "income"}}
]
'''

# mloda parses JSON into Feature objects
features = load_features_from_config(llm_output, format="json")

result = mloda.run_all(
    features=features,
    compute_frameworks=["PandasDataFrame"],
    api_data={"SampleData": {
        "customer_id": ["C001", "C002", "C003"],
        "income": [50000, 75000, 90000],
        "age": [25, 35, 45]
    }}
)
```

**LLM-friendly:** The agent only declares what it needs - mloda handles the rest.

### 2. Context Window Assembly

Gather context from multiple sources declaratively - mloda validates and delivers. Why not let an AI agent do it?

*Example: This shows the API pattern. Requires custom FeatureGroup implementations for your data sources.*

```python
from mloda.user import Feature, mloda

# Build complete context from multiple sources
features = [
    Feature(name="system_instructions", options={"template": "support_agent"}),
    Feature(name="user_profile", options={"user_id": user_id, "include_preferences": True}),
    Feature(name="knowledge_base", options={"query": user_query, "top_k": 5}),
    Feature(name="conversation_history", options={"limit": 20, "summarize_old": True}),
    Feature(name="available_tools", options={"category": "customer_service"}),
    Feature(name="output_format", options={"format": "markdown", "max_length": 500}),
]

result = mloda.run_all(
    features=features,
    compute_frameworks=["PythonDictFramework"],
    api_data={"UserQuery": {"query": [user_query]}}
)

# Each feature resolved via its plugin, validated
```

### 3. RAG with Feature Chaining

Build RAG pipelines declaratively - mloda chains the steps for you.

*Example: This shows the chaining syntax. Requires custom FeatureGroup implementations for retrieval and processing.*

```python
# String-based chaining: query -> validate -> retrieve -> redact
Feature(name="user_query__injection_checked__retrieved__pii_redacted")

# Configuration-based chaining: explicit pipeline
Feature(
    name="safe_context",
    options=Options(context={
        "in_features": "documents__retrieved__pii_redacted",
        "redact_types": ["email", "phone", "ssn"]
    })
)
```

mloda resolves the full chain - you declare the end result, not the steps.

**Automatic dependency resolution:** You only declare what you need. If `pii_redacted` depends on `retrieved` which depends on `documents`, just ask for `pii_redacted` - mloda traces back and resolves the full chain.

Beyond string-based chaining, you can declare dependencies directly via `input_features()`. Each plugin states what it needs, mloda resolves the rest. Because resolution depends on which plugins are registered, the same request can have different chain lengths per environment: realtime might resolve `income` straight from a live source, while a RAG pipeline routes it through ETL, validation, and enrichment first. The calling code stays the same:

```python
class RiskAssessment(FeatureGroup):
    def input_features(self, options, feature_name):
        return {Feature("debt_to_income"), Feature("age"), Feature("employment_years")}

class DebtToIncome(FeatureGroup):
    def input_features(self, options, feature_name):
        return {Feature("debt"), Feature("income")}

# Request only risk_assessment. mloda auto-resolves:
#   risk_assessment -> debt_to_income -> {debt, income}
#                   -> age, employment_years
result = mloda.run_all(features=[Feature(name="risk_assessment")], ...)
```

---

## Compute Frameworks

Mix multiple backends in a single pipeline - mloda routes each feature to the right framework:

```python
result = mloda.run_all(
    features=[...],
    compute_frameworks=["PandasDataFrame", "PolarsDataFrame", "SparkFramework"]
)

# Results may come from different frameworks based on plugin compatibility
```

Add your own frameworks - mloda is extensible.

---

## Extenders

Wrap plugin execution for logging, validation, or lineage tracking:

```python
import time
from mloda.steward import Extender, ExtenderHook

class LogExecutionTime(Extender):
    def wraps(self):
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Took {time.time() - start:.2f}s")
        return result

# Use it
result = mloda.run_all(features, function_extender={LogExecutionTime()})
```

Built-in and custom extenders give you full lineage - trace any result back to its source.

---

## When to Use mloda

**Use mloda when:**
- Your agents need data from multiple sources
- You want consistent, validated data access
- You need traceability (audit, debugging)
- Multiple agents share the same data patterns

**Don't use mloda for:**
- Single database, simple queries → use an ORM
- One-off scripts → just write the code
- Real-time streaming (<5ms) → use Kafka/Flink

---

## Documentation

- **[Getting Started](https://mloda-ai.github.io/mloda/chapter1/installation/)** - Installation and first steps
- **[Plugin Development](https://mloda-ai.github.io/mloda/chapter1/feature-groups/)** - Build your own plugins
- **[API Reference](https://mloda-ai.github.io/mloda/in_depth/mloda-api/)** - Complete API docs

---

## Ecosystem

Most plugins currently live in `mloda_plugins/` within this repository. The goal is to gradually migrate them to standalone packages in the registry.

| Repository | Description |
|------------|-------------|
| [mloda-registry](https://github.com/mloda-ai/mloda-registry) | Official plugin packages and 40+ development guides |
| [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template) | Cookiecutter template for creating standalone plugins |

---

## Contributing

We welcome contributions.

**Plugin development:** see the [mloda-registry guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides) (40+ step-by-step), or scaffold a new package with the [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template).

**Core framework:** start with [`good first issue`](https://github.com/mloda-ai/mloda/labels/good%20first%20issue) or [`help wanted`](https://github.com/mloda-ai/mloda/labels/help%20wanted), then:

```bash
git clone https://github.com/mloda-ai/mloda.git
cd mloda
uv sync --all-extras
source .venv/bin/activate
tox  # full test, lint, type-check, and security suite
```

If `tox` passes, you're set. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full PR workflow and code style, and the [Code of Conduct](CODE_OF_CONDUCT.md).

- [GitHub Issues](https://github.com/mloda-ai/mloda/issues/) - report bugs or request features
- [Development Guide](https://mloda-ai.github.io/mloda/development/) - in-depth contributor docs
