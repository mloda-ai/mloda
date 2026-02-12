### API Request Example

This example demonstrates a simple request to the mloda API. You describe WHAT data you need - mloda resolves HOW to get it.

> **Tip:** For AI agents or quick prototyping, you can also use inline data with `api_data` - see the [30-second example](https://mloda-ai.github.io/mloda/#30-second-example).

In this example, mloda automatically determines that a **CsvReader** feature group will fulfill the request and respond with the resulting DataFrame.

#### 1. Import the Required Modules
We first need to import the necessary components to set up our request:
```python
from mloda.user import mloda
from mloda.user import DataAccessCollection
```

#### 2. Define Data Sources
Next, we define the data source by specifying the file path and creating a **DataAccessCollection** object:
```python
file_path = "tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"
data_access_collection = DataAccessCollection(files={file_path})
``` 

#### 3. Define the features
We now specify the list of features we want to retrieve. The features are provided as a comma-separated string, which we split into a list:
```python
feature_list = "id,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class"
feature_list = feature_list.split(",")
```

#### 4. Execute the Request and Retrieve Results
Finally, we send the request to the mloda API, specifying the compute framework and the data source. The result is returned as a DataFrame:
```python
result = mloda.run_all(
    feature_list,
    compute_frameworks=["PyArrowTable"],
    data_access_collection=data_access_collection)
result[0]
```
Expected output:
``` python
pyarrow.Table
V8: double
id: int64
...
V8: [[-0.13000604758867731,-0.13311827417649086,-0....]]
id: [[0,1,2,...]]
....
```
#### 5. Summary

This example shows how easy it is to interact with the mloda API for data retrieval. By defining the data source and specifying the features, mloda handles the rest.

#### For Realtime / Inference Use Cases

With `run_all()`, mloda rebuilds the full execution plan on every call. That is fine for
batch jobs, but in latency-sensitive scenarios the repeated planning overhead matters.

Typical examples where this applies:

- **ML model serving** — a prediction endpoint computes the same derived features for every incoming request; only the raw input row changes.
- **Streaming / event-driven pipelines** — each incoming event needs the same feature transformations applied in real time.
- **Interactive applications** — a dashboard or API recalculates features on every user action with fresh parameters.

The two-phase API lets you pay the planning cost once at startup and then execute
cheaply per request:

``` python
# 1. Prepare once (e.g. at server startup)
session = mloda.prepare(feature_list, compute_frameworks=["PyArrowTable"], data_access_collection=data_access_collection)

# 2. Execute per request with fresh data
result = session.run(api_data={"MyKey": {"col": [1, 2]}})
```

See [mloda API — Two-Phase Execution](../in_depth/mloda-api.md#two-phase-execution-prepare-run) for details.

#### For AI Agents: JSON-Based Requests

LLMs can generate JSON feature requests without writing Python code:

``` python
from mloda.user import load_features_from_config

llm_request = '["id", "V1", "V2", "Amount"]'
features = load_features_from_config(llm_request, format="json")
result = mloda.run_all(features=features, compute_frameworks=["PandasDataFrame"], ...)
```

See [Feature Configuration](https://mloda-ai.github.io/mloda/in_depth/feature-config/) for more details on JSON-based configuration.
