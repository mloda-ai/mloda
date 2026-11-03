# mloda

**Declarative data access for AI agents and ML. Describe what you need, mloda resolves how.**

[![PyPI version](https://img.shields.io/pypi/v/mloda.svg)](https://pypi.org/project/mloda/)
[![Python versions](https://img.shields.io/pypi/pyversions/mloda.svg)](https://pypi.org/project/mloda/)
[![CI](https://github.com/mloda-ai/mloda/actions/workflows/ci.yaml/badge.svg)](https://github.com/mloda-ai/mloda/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mloda-ai/mloda/blob/main/LICENSE)

[Documentation](https://mloda-ai.github.io/mloda/) · [Plugin guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides) · [Website](https://mloda.ai)

You name the features you want. mloda finds the plugins that produce them, orders them, runs them, and shows you the plan it used. The same request runs on plain Python, pandas, Polars, PyArrow and more.

## See it

Needs [uv](https://docs.astral.sh/uv/). Nothing is added to your project: `uvx` fetches mloda into uv's own cache and runs it from there.

```bash
uvx mloda@latest demo
```

It runs the example below and prints the result, then the two steps mloda chose: read the data, then aggregate. You named the features; mloda found the steps.

## Run it

Install mloda, paste the example into a file and run it. Then change the feature names or the data and run it again.

```bash
pip install mloda
```

```python
from mloda.user import PluginLoader, mloda
PluginLoader.all()

result = mloda.run_all(
    features=["customer_id", "income", "income__sum_aggr", "age__avg_aggr"],
    compute_frameworks=["PythonDictFramework"],
    api_data={"SampleData": {
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "age": [25, None, 45, 30, 50],
        "income": [50000, 75000, None, 60000, 85000],
    }},
)
print(result.summary())
```

```text
Result
customer_id  income
C001         50000
C002         75000
C003         None
C004         60000
C005         85000

age__avg_aggr  income__sum_aggr
37.5           270000
37.5           270000
37.5           270000
37.5           270000
37.5           270000

Plan
step 1  ApiDataFG               age, customer_id, income  (source SampleData)
step 2  AggregatedFeatureGroup  age__avg_aggr, income__sum_aggr  (from age, income)
```

What happened: you asked for four features by name. `income__sum_aggr` and `age__avg_aggr` are not in your data, so mloda matched them to the plugin that computes aggregations (`AggregatedFeatureGroup`), saw that it needs `income` and `age`, and read those from your data first (`ApiDataFG`). Nothing in the request said how.

Derived features follow one naming pattern, `column__operation_group`, for example `income__sum_aggr`. The plugin, called a FeatureGroup, declares the pattern. You only use it.

To run the same request on a DataFrame, install a backend and change one line:

```bash
pip install "mloda[pandas]"   # or mloda[polars], mloda[pyarrow]
```

Then set `compute_frameworks=["PandasDataFrame"]` (or `PolarsDataFrame`, `PyArrowTable`). The features and the plan stay the same.

## Using an agent?

Claude Code, Claude Desktop, Cursor, Codex, or an agent you build: give it these rules before it writes mloda code. Paste them into `AGENTS.md`, `CLAUDE.md` or your agent's system prompt.

```markdown
mloda rules
- Import from `mloda.user`, not `import mloda`.
- Call `PluginLoader.all()` once before the first request.
- A bare install runs `compute_frameworks=["PythonDictFramework"]`. pandas needs `pip install "mloda[pandas]"`.
- Derived features are named `column__operation_group`, for example `income__sum_aggr`.
- Check a name before running it: `resolve_feature("income__sum_aggr")` from `mloda.user`.
```

Writing a plugin? Point the agent at the [plugin guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides) in mloda-registry.

## Why mloda

| You want to | mloda gives you |
|---|---|
| Give agents and pipelines data without hand-written glue | One request format: name the features, mloda resolves the plugins |
| Trace every result | The plan and the lineage back to the source, for every run |
| Reuse work across notebooks, projects and teams | Plugins that run unchanged on any supported backend |

Don't use mloda for a single database with simple queries (use an ORM), for a one-off script, or for sub-millisecond streaming.

## Documentation

- [Getting started](https://mloda-ai.github.io/mloda/chapter1/installation/)
- [Build your own plugins](https://mloda-ai.github.io/mloda/chapter1/feature-groups/)
- [API reference](https://mloda-ai.github.io/mloda/in_depth/mloda-api/)
- [Plugin guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides): pattern guides in mloda-registry, also written as context for coding agents

## Ecosystem

| Repository | What it holds |
|---|---|
| [mloda-registry](https://github.com/mloda-ai/mloda-registry) | Plugin packages and the plugin guides |
| [mloda-plugin-template](https://github.com/mloda-ai/mloda-plugin-template) | GitHub template for a standalone plugin package |

Most plugins still live in `mloda_plugins/` in this repository and move to the registry over time.

## Contributing

Start with [good first issue](https://github.com/mloda-ai/mloda/labels/good%20first%20issue) or [help wanted](https://github.com/mloda-ai/mloda/labels/help%20wanted), then:

```bash
git clone https://github.com/mloda-ai/mloda.git
cd mloda
uv sync --all-extras
source .venv/bin/activate
tox
```

If `tox` passes, you are set. [CONTRIBUTING.md](https://github.com/mloda-ai/mloda/blob/main/CONTRIBUTING.md) has the PR workflow and code style, and the [Code of Conduct](https://github.com/mloda-ai/mloda/blob/main/CODE_OF_CONDUCT.md) applies.

If mloda is useful to you, star the repository so others find it.
