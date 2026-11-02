## 1. Install mloda via pip
You can install mloda using either pip or by cloning the repository from Git.
```bash
pip install mloda
```

The core package has no mandatory third-party dependencies. Choose extras for the backends or integrations you need,
for example `pip install "mloda[pyarrow]"`.

| Extra | Enables |
| --- | --- |
| `pyarrow` | PyArrow tables and file readers |
| `pandas` | Pandas DataFrames |
| `polars` | Polars DataFrames |
| `numpy` | NumPy for PyArrow multi-column aggregations and time windows |
| `yaml` | YAML file input |
| `duckdb` | DuckDB SQL backend |
| `sqlite` | SQLite SQL backend |
| `iceberg` | Apache Iceberg tables |
| `spark` | PySpark DataFrames |
| `sklearn` | scikit-learn transformations and pipelines |
| `text_cleaning` | NLTK text cleaning |
| `mcp` | Model Context Protocol (MCP) server |

Extras that install pyarrow require `pyarrow>=14.0.1` (CVE-2023-47248); use that version or newer if you supply your own.

## 2. Install mloda via Git
Alternatively, you can clone the repository and install from source:
```bash
git clone https://github.com/mloda-ai/mloda.git
cd mloda
pip install .
```

## 3. Verify Installation
To verify the installation, run:
```python
from importlib import metadata
print(metadata.version("mloda"))
```

## 4. Quick Start

A plain `pip install mloda` is enough to run this with the built-in Python dict backend:

```python
from mloda.user import PluginLoader, mloda
PluginLoader.all()

result = mloda.run_all(
    features=["customer_id", "income", "income__mean_imputed", "age__median_imputed"],
    compute_frameworks=["PythonDictFramework"],
    api_data={"SampleData": {
        "customer_id": ["C001", "C002", "C003", "C004", "C005"],
        "age": [25, None, 45, 30, 50],
        "income": [50000, 75000, None, 60000, 85000],
    }},
)
print(result)
```

```text
[{'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'], 'income': [50000, 75000, None, 60000, 85000]}, {'age__median_imputed': [25, 37.5, 45, 30, 50], 'income__mean_imputed': [50000, 75000, 67500, 60000, 85000]}]
```

Prefer DataFrames? Install a backend extra:

```bash
pip install "mloda[pandas]"
# or mloda[polars], mloda[pyarrow], mloda[duckdb], mloda[spark]
```

```python
from mloda.user import mloda, PluginLoader
PluginLoader.all()

result = mloda.run_all(
    features=["customer_id", "income", "income__sum_aggr"],
    compute_frameworks=["PandasDataFrame"],
    api_data={"SampleData": {
        "customer_id": ["C001", "C002", "C003"],
        "income": [50000, 75000, 90000]
    }}
)
print(result)
```

`api_data` passes inline data to mloda. Each top-level key (e.g. `"SampleData"`)
is a label grouping related columns. Features are matched to columns by name.

Next: [API Request](api-request.md) | [Feature Groups](feature-groups.md)
