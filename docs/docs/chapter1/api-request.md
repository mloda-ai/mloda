
### mloda Request Example
This example demonstrates a simple request to the mloda API. In this case, mloda automatically determines that a **CsvReader** feature group will fulfill the request and respond with the resulting DataFrame.

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
This example shows how easy it is to interact with the mloda API for data retrieval. By defining the data source and specifying the features, mloda handles the rest, ensuring you get the required data in an efficient and seamless manner.
