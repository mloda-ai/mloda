## Compute Framework

In the previous examples, you may have noticed that we used the following parameter in the mlodaAPI call: **compute_frameworks=["PyArrowTable"]**. Let’s take a moment to dive into this concept.

The Compute Framework is the second critical plugin in mloda, after the feature group. It is responsible for holding the state of the data and defining the technology used to execute operations.

#### 1. Key Use Cases

The compute framework concept provides significant flexibility and enables a variety of use cases, such as:

-   Online and Offline Computation: Seamlessly switch between real-time and batch computations.
-   Testing: Easily compare different compute technologies or frameworks.
-   Migrations: Move from one environment to another (e.g., local to cloud or db to other db) without changing the underlying feature definitions.

This flexibility is one of mloda’s key advantages, allowing users to decouple feature definitions from specific computation technologies—something that traditional feature stores don’t easily offer.

#### 2. Balancing Flexibility and Complexity

However, this flexibility introduces a bit of complexity. Let's look at an example where we remove the compute_frameworks=["PyArrowTable"] parameter from the mlodaAPI call.

#### 3. Example Without a Specified Compute Framework
```python
import mloda
from mloda.user import DataAccessCollection

file_path = "tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"
data_access_collection = DataAccessCollection(files={file_path})

feature_list = ["id","V1","V2","V3"]
```

Expected Error when running.

``` python
mloda.run_all(
    feature_list,
    data_access_collection=data_access_collection
)

ValueError: Multiple feature groups
{<class 'CsvReader'>: {<class 'PyArrowTable'>, <class 'PandasDataFrame'>}
.... found for feature name: id.
```

In this case, the framework finds multiple compute frameworks (like **PyArrowTable** and **PandasDataFrame**) that can handle the same file through the **CsvReader** feature group. Without explicitly specifying a compute framework, mloda doesn't know which one to use, leading to ambiguity.

This might seem counterintuitive, but it’s actually a **feature**, allowing you to compare different technologies and computation methods, particularly useful in scenarios such as:

-   **Migrations**: Moving from one environment to another.
-   **Scaling Projects**: Upleveling a project from MVP to production.
-   **ML Lifecycle**: Using the same KPIs (feature groups) across training, real-time inference, and model evaluation.

#### 4. Design options
There are several ways to resolve this ambiguity by explicitly defining the compute framework:

-   **Using specific feature configuration** to define compute frameworks for individual features.
-   **Within the feature group definition**, by enforcing a specific compute framework rule.
-   **As part of the API request** (as shown in previous examples).
##### Part of API request

##### Specific feature configuration
You can configure individual features to use a specific compute framework. Here’s how to specify that a feature should use the PyArrowTable framework:
```python
import mloda
from mloda.user import Feature

feature = Feature("id", options={"compute_framework": "PyArrowTable"})

result = mloda.run_all(
    [feature],
    data_access_collection=data_access_collection
)
result[0]
```
Expected output:
``` python
pyarrow.Table
id: int64
id: [[0,1,2,3,...]]
```
##### Defining the Compute Framework in a Feature Group
In this example, we define a compute framework rule inside the feature group. This ensures that the feature group can only run on a **PyArrowTable**. We also specify that the input feature should use **PandasDataFrame**, allowing automatic conversion from **PandasDataFrame** to **PyArrowTable** behind the scenes.


```python
import pyarrow.compute as pc
import pyarrow as pa

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.provider import FeatureGroup


class ExampleB(FeatureGroup):

    @classmethod
    def compute_framework_rule(cls):
            return {PyArrowTable}
    def input_features(self, option, feature_name):
        return {Feature(name=feature_name.name.split("_")[1], 
                compute_framework="PandasDataFrame")}

    @classmethod
    def calculate_feature(cls, data, _):
        multiplied_columns = [pc.multiply(data[column], 2) for column in data.column_names]
        col_names = [f"{cls.get_class_name()}_{col_names}" for col_names in data.column_names]
        multiplied_table = pa.table(multiplied_columns, names=col_names)
        return multiplied_table
```
Running the ExampleB Feature Group
```python
example_feature_list = [f"ExampleB_{f}" for f in feature_list]

result = mloda.run_all(
    example_feature_list,
    compute_frameworks={PyArrowTable, PandasDataFrame},
    data_access_collection=data_access_collection,
)
result[0]
```
In this case, the feature group ExampleB will only run on the PyArrowTable framework, while the input feature group uses PandasDataFrame, ensuring that the framework correctly handles the conversion between these technologies.


#### 5. Available Compute Frameworks

| Framework | Technology | Strengths | Best For | Dependencies |
|-----------|------------|-----------|----------|--------------|
| **PandasDataFrame** | pandas DataFrame | Rich data transformation, familiar API | Development, data exploration, smaller datasets | pandas, numpy |
| **PyArrowTable** | Apache Arrow Tables | Memory-efficient, high performance, columnar format | Production, big data, interoperability | pyarrow |
| **PolarsDataFrame** | Polars DataFrame | Fast, memory-efficient, eager evaluation | Development, immediate results | polars |
| **PolarsLazyDataFrame** | Polars LazyFrame | Query optimization, lazy evaluation | Large datasets, performance optimization | polars |
| **DuckDBFramework** | DuckDB Relations | SQL interface, fast analytics, OLAP queries | Analytical workloads, SQL-based transformations, data warehousing | duckdb |
| **IcebergFramework** | Apache Iceberg Tables | Schema evolution, time travel, data lake management | Data lake scenarios, versioned datasets, large-scale analytics | pyiceberg, pyarrow |
| **SparkFramework** | Apache Spark DataFrames | Distributed processing, scalability, fault tolerance | Big data, distributed computing, production clusters | pyspark, Java 8+ |
| **PythonDictFramework** | List[Dict[str, Any]] | Zero dependencies, simple, lightweight | Minimal environments, education, prototyping | None (Python stdlib only) |

##### Automatic Dependency Detection

mloda automatically detects which compute frameworks are available based on installed dependencies. If a required dependency (like `pandas` or `pyarrow`) is not installed, the corresponding compute framework will be automatically excluded from discovery, preventing runtime errors.

This means you can:
- Install only the dependencies you need for your specific use case
- Deploy mloda in minimal environments without all compute framework dependencies
- Avoid import errors when optional dependencies are missing

For example, if `polars` is not installed, `PolarsDataFrame` will not be available as a compute framework option, and mloda will automatically work with the remaining available frameworks.

Example using PythonDictFramework:
``` python
import mloda
from mloda.user import Feature

feature = Feature("id", options={"compute_framework": "PythonDictFramework"})

result = mloda.run_all(
    [feature],
    data_access_collection=data_access_collection
)
result[0]  # Returns List[Dict[str, Any]]
```

Example using Polars frameworks:
``` python
import mloda
from mloda.user import Feature

# Using Polars eager evaluation
feature_eager = Feature("id", options={"compute_framework": "PolarsDataFrame"})

# Using Polars lazy evaluation
feature_lazy = Feature("id", options={"compute_framework": "PolarsLazyDataFrame"})

result = mloda.run_all(
    [feature_eager],
    data_access_collection=data_access_collection
)
result[0]  # Returns polars.DataFrame
```

Example using DuckDB framework:
``` python
import mloda
from mloda.user import Feature, DataAccessCollection
import duckdb

# Create DuckDB connection
connection = duckdb.connect()

# Set up data access with connection
data_access_collection = DataAccessCollection(
    initialized_connection_object={connection}
)

feature = Feature("id", options={"compute_framework": "DuckDBFramework"})

result = mloda.run_all(
    [feature],
    data_access_collection=data_access_collection
)
result[0]  # Returns duckdb.DuckDBPyRelation
```

**Note**: DuckDB framework requires a connection object and does not support mloda framework inherent multiprocessing. Multiprocessing from DuckDB still works. It's optimized for analytical workloads and provides SQL-like operations on data.

Example using Iceberg framework:
``` python
import mloda
from mloda.user import Feature, DataAccessCollection
from pyiceberg.catalog import load_catalog

# Create Iceberg catalog (example with REST catalog)
catalog = load_catalog("default", **{
    "uri": "http://localhost:8181",
    "credential": "client-credentials",
    "client.id": "admin",
    "client.secret": "password"
})

# Set up data access with catalog
data_access_collection = DataAccessCollection(
    initialized_connection_objects={catalog}
)

feature = Feature("id", options={"compute_framework": "IcebergFramework"})

result = mloda.run_all(
    [feature],
    data_access_collection=data_access_collection
)
result[0]  # Returns pyiceberg.table.Table or pyarrow.Table
```

**Note**: Iceberg framework requires a catalog connection object for table operations. It's optimized for data lake scenarios with schema evolution, time travel capabilities, and large-scale analytics. The framework uses PyArrow as an interchange format for compatibility with other mloda frameworks.

Example using Spark framework:
``` python
import mloda
from mloda.user import Feature, DataAccessCollection
from pyspark.sql import SparkSession

# Create SparkSession (optional - framework will auto-create if not provided)
spark = SparkSession.builder \
    .appName("MLoda-Spark-Example") \
    .master("local[*]") \
    .getOrCreate()

# Set up data access with SparkSession
data_access_collection = DataAccessCollection(
    initialized_connection_objects={spark}
)

feature = Feature("id", options={"compute_framework": "SparkFramework"})

result = mloda.run_all(
    [feature],
    data_access_collection=data_access_collection
)
result[0]  # Returns pyspark.sql.DataFrame
```

**Note**: Spark framework requires PySpark installation and Java 8+ environment with JAVA_HOME configured. It's optimized for big data processing, distributed computing, and production clusters. The framework can auto-create a local SparkSession if none is provided, but for production use, you should provide a configured SparkSession. Does not support mloda framework inherent multiprocessing (uses Spark's own distributed processing).

#### 6. Summary

mloda's compute framework adds flexibility, allowing you to select the best tool for different stages of data and feature engineering. While it introduces some complexity, it's invaluable for comparing technologies and managing environments.

That said, **you can configure mloda to use just one compute framework** for a more familiar workflow, similar to traditional feature stores, data pipelines, or ETL systems. Whether you prefer flexibility or simplicity, mloda adapts to your needs while ensuring consistent feature processing.

### 6. Advanced Compute Framework Topics

For more in-depth information about compute frameworks, check out these advanced topics:

- [Framework Transformers](../in_depth/framework-transformers.md) - How data is transformed between different compute frameworks
- [Compute Framework Integration](../in_depth/compute-framework-integration.md) - How feature groups integrate with different compute frameworks
- [Framework Connection Object](../in_depth/framework-connection-object.md) - How stateful frameworks manage persistent connections and state

#### 7. Discovering Compute Frameworks

To list all available compute frameworks and their documentation, use the `get_compute_framework_docs()` function from `mloda.steward`.
