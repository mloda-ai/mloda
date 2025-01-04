## Access (feature) data

In this framework, features can access and manage data through several structured methods, each designed for specific use cases. Whether a feature requires access to shared data, isolated data, or needs to generate data independently, the system offers flexible solutions. This guide covers the key mechanisms for data access, including the 

-   **DataAccessCollection** for centralized management, 
-   **Feature Scope Data Access** for feature-specific control, 
-   **Input Features** for inter-feature data sharing, 
-   and **Data Creator** for generating independent data. 

These methods ensure efficient data handling across the framework while promoting flexibility and scalability.

#### DataAccessCollection

The DataAccessCollection is designed to control the access to data of any kind. The main purpose of this class is to organize and simplify interactions with these different data elements, making it easier to work to ingest data of various form into the framework. It provides as an interface for accessing and storing of data:

The DataAccessCollection can only be added via **mlodaAPI**.

List options:

1.  **Files:** Specifies the exact location of files: path/folder/text.txt

2.  **Folders:** Points to directories where files are located: path/folder/

3.  **Credential dicts:** Contains the necessary credentials to access data:  
 {host: example.com, password: example}

4.  **Initialized connection object:** Stores connection objects that are already initialized: (DBConnectionObject)

5.  **Unitialized connection object:** Stores not initialized connection objects: (UninitializedDBConnection)

Example:

``` python
data_access = DataAccessCollection()

# Add file paths, folder paths, credentials, and connection objects
data_access.add_file('path/to/folder/text.txt')
data_access.add_folder('path/to/folder/')
data_access.add_credential_dict({'host': 'example.com', 'password': 'example'})
data_access.add_initialized_connection_object('InitializedDBConnection')
data_access.add_uninitialized_connection_object('UninitializedDBConnection')

mlodaAPI.run_all(
    feature_list,
    data_access_collection=data_access)
```


#### Feature Scope Data Access

If data needs to be added specifically for a single feature (or features from the same feature group), you can use the feature_scope_data_access_name functionality.

Example feature group definition:
``` python
class ReadCsv(ReadFile):  # ReadFile is inheriting from AbstractFeatureGroup
    _end: str = "csv"

    @classmethod
    def suffix(cls) -> str:
        return f".{cls._end}"

    @classmethod
    def feature_scope_data_access_name(cls) -> Optional[str]:
        return f"read_{cls._end}"
```

Example usage
``` python
feature = Feature(
    name="example_feature", 
    options={ReadCsv.feature_scope_data_access_name(): "example.csv"}
)

mlodaAPI.run_all([feature])
```

#### Input features 

The input_features method allows a feature to access data from other features. This enables data sharing and collaboration between different components of your system.

This is one of the key aspects in how we achieve to split data from processes.

``` python

class ExampleFeatureClass(AbstractFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        _input_features = {}
        _input_features.add(Feature.int32_of("SUM_V1"))
        _input_features.add(Feature.str_of("Name"))

        # You can add diverse logics based on given options and/or feature names.
        complicated_feature = Feature(name=feature_name.name.split("_")[1], compute_framework="PandasDataframe")
        _input_features.add(complicated_feature)
            
        return _input_features

    # Use the features then in the calculate part.
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {
            data["SUM_V1"], 
            data["Name"],
            data["ExampleFeatureClass_SPLIT_VALUE"],
        }

```

As the input features can be fulfilled by **multiple other features**, we can have the same processes running in different environments, migrations and processes.

#### Data Creator

The data creator can create data independent of any other dependency. It is essentially a base feature that does not needs a **DataAccessCollection** or **Feature Scope Data Access**.

Usage:

- test data,
- sample data,
- dummy data,
- parameter data

One could imagine that for experimenting one wants to see data. Then one could use this feature as input feature to another feature instead of e.g. the true data.

``` python

class ExampleFeatureClass(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"ExampleFeatureClass", "idx"})
    
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"ExampleFeatureClass": [1, 2, 3], 
                "idx": ["a", "b", "c"]}
```
