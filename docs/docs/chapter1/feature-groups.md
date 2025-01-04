### Creating a Custom Feature Group

In this example, we’ll create a custom feature group that multiplies the results of each feature by 2. We'll implement a new feature group and then use it within mloda.

#### 1. Import the Required Modules and Set File References

Start by importing the necessary modules to define the custom feature group and perform calculations:
```python
import pyarrow.compute as pc
import pyarrow as pa

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection

file_path = "tests/test_plugins/feature_group/src/dataset/creditcard_2023_short.csv"
data_access_collection = DataAccessCollection(files={file_path})

feature_list = ["id","V1","V2","V3"]
example_feature_list = [f"Example_{f}" for f in feature_list]

```

#### 2. Define the Feature Group
The custom feature group, Example, operates on a set of input features. It depends on the root features (e.g., "id", "V1", etc.) and renames them with the prefix "Example_". 

The calculation logic for multiplying each feature by 2 is implemented in the calculate_feature function.

```python
class Example(AbstractFeatureGroup):
    def input_features(self, _, feature_name):
        return {feature_name.name.split("_")[1]}

    @classmethod
    def calculate_feature(cls, data, _):
        multiplied_columns = [pc.multiply(data[column], 2) for column in data.column_names]
        col_names = [f"{cls.get_class_name()}_{col_names}" for col_names in data.column_names]
        multiplied_table = pa.table(multiplied_columns, names=col_names)
        return multiplied_table
```

#### 3. Execute the Request Using the New Feature Group
To use the newly defined feature group, simply add the **"Example_"** prefix to each feature name. mloda will automatically resolve the dependency between the **ReadCsvFeatureGroup** and the **ExampleFeatureGroup**.

```python
from mloda_core.api.request import mlodaAPI

result = mlodaAPI.run_all(
            example_feature_list, 
            compute_frameworks=["PyarrowTable"], 
            data_access_collection=data_access_collection
        )
result[0]
```
Expected output:
``` python
pyarrow.Table
Example_V28: double
Example_id: int64
...
Example_V28: [[-0.26000604758867731,-0.26311827417649086,-0....]]
Example_id: [[0,2,4,...]]
....
```

#### 4. Summary

In this example, we implemented a custom feature group, Example, that multiplies each feature value by 2. By defining a straightforward input_features method and a calculate_feature method, we were able to extend mloda's feature engineering capabilities with custom transformations. We then executed the request by simply modifying the feature names with a prefix ("Example_"), allowing mloda to handle dependencies and computations automatically.
