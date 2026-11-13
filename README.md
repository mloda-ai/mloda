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

![mloda demo: the result tables and the two-step plan mloda resolved](https://raw.githubusercontent.com/mloda-ai/mloda/main/docs/assets/mloda-demo.gif)

It prints the result, then the plan: the two steps mloda chose to make the features you named.

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

Two of the four features do not exist in the data. The plan shows how mloda made them:

- `income__sum_aggr` and `age__avg_aggr` name an aggregation, so step 2 is the aggregation plugin, `AggregatedFeatureGroup`.
- That plugin needs `income` and `age`, so step 1 reads them from your data with `ApiDataFG`.

You described what. mloda resolved how.

Change a name and run again: `income__max_aggr` gives the maximum, `age__count_aggr` the count. Names follow one shape, `column__operation_suffix`. The suffix picks the plugin (here `_aggr`, a FeatureGroup), and the plugin declares which operations it accepts.

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

## Your own FeatureGroup

A FeatureGroup is a class that names the feature it provides, declares what it needs, and computes it. Define one, then request it by name like any other feature.

```python
from mloda.provider import FeatureGroup
from mloda.user import Feature, PluginLoader, mloda
PluginLoader.all()

class IncomeAfterTax(FeatureGroup):
    @classmethod
    def feature_names_supported(cls):
        return {"income_after_tax"}

    def input_features(self, options, feature_name):
        return {Feature("income")}

    @classmethod
    def calculate_feature(cls, data, features):
        return {"income_after_tax": [None if v is None else round(v * 0.7) for v in data["income"]]}

result = mloda.run_all(
    features=["income", "income_after_tax"],
    compute_frameworks=["PythonDictFramework"],
    api_data={"SampleData": {"income": [50000, 75000, None, 60000, 85000]}},
)
print(result.summary())
```

```text
Result
income
50000
75000
None
60000
85000

income_after_tax
35000
52500
None
42000
59500

Plan
step 1  ApiDataFG       income  (source SampleData)
step 2  IncomeAfterTax  income_after_tax  (from income)
```

Three methods: the name it answers to, the input it needs, the computation. mloda read `income` from your data and ran your class after it. Any FeatureGroup your code imports is available to `run_all`; the [plugin guides](https://github.com/mloda-ai/mloda-registry/tree/main/docs/guides) cover naming patterns, options, joins and the rest.

## Same request, other backend

The request does not change when the backend does. Install one and name it:

```bash
pip install "mloda[pandas]"
```

```python
from mloda.user import PluginLoader, mloda
PluginLoader.all()

result = mloda.run_all(
    features=["customer_id", "income", "income__sum_aggr", "age__avg_aggr"],
    compute_frameworks=["PandasDataFrame"],
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
  customer_id   income
0        C001  50000.0
1        C002  75000.0
2        C003      NaN
3        C004  60000.0
4        C005  85000.0

   age__avg_aggr  income__sum_aggr
0           37.5          270000.0
1           37.5          270000.0
2           37.5          270000.0
3           37.5          270000.0
4           37.5          270000.0

Plan
step 1  ApiDataFG               age, customer_id, income  (source SampleData)
step 2  AggregatedFeatureGroup  age__avg_aggr, income__sum_aggr  (from age, income)
```

Same features, same plan, pandas DataFrames in the result. PyArrow (`mloda[pyarrow]`, `PyArrowTable`) works the same way.

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
