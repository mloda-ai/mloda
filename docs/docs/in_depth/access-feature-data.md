## Access (feature) data

This framework provides several structured mechanisms for features to access and manage data, catering to diverse needs. Features can retrieve data through:

-   **DataAccessCollection** for global data loading management, 
-   **Feature Scope Data Access** for feature-specific data loading management, 
-   **Api Data** for using data directly with the mlodaAPI request.
-   **Data Creator** for generating data instead of loading data,
-   **Input Features** facilitates data sharing between features.

These methods ensure efficient data management while maintaining flexibility and scalability. A typical scenario involves a complex feature relying on three input features. These input features, in turn, may depend on other input features or load data using ApiData.

> **Advanced**: For a detailed explanation of the underlying data access patterns (BaseInputData vs MatchData), see [Data Access Patterns](data-access-patterns.md).

#### DataAccessCollection - global data access

The DataAccessCollection is designed to control the access to data of any kind. The main purpose of this class is to organize and simplify interactions with these different data elements, making it easier to work to ingest data of various form into the framework. It provides as an interface for accessing and storing of data on a global level.

The DataAccessCollection can only be added via **mlodaAPI**.

List options:

1.  **Files:** Specifies the exact location of files: path/folder/text.txt

2.  **Folders:** Points to directories where files are located: path/folder/

3.  **Credential dicts:** Contains the necessary credentials to access data:  
 {host: example.com, password: example}

4.  **Initialized connection object:** Stores connection objects that are already initialized: (DBConnectionObject)

5.  **Unitialized connection object:** Stores not initialized connection objects: (UninitializedDBConnection)

You can apply these options like so:

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

#### Global Scope Data Access

A concrete, simplified global scope data access is shown in this example:

```python
import os

from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.api.request import mlodaAPI


file_path = os.getcwd()
file_path += "/docs/docs/in_depth"

data_access_collection = DataAccessCollection(folders={str(file_path)})

result = mlodaAPI.run_all(
            ["AExample", "BExample"], 
            compute_frameworks=["PandasDataframe"], 
            # Define data access on a global level
            data_access_collection=data_access_collection
        )
print(result)
```

Output

``` python
[  AExample  BExample
0   Value1         2
1   Value2         3]
```

#### Feature Scope Data Access

The Feature Scope Data access is instead designed to control the access to data of any kind 
on a local level. 

If data needs to be added specifically for a single feature (or features from the same feature group), you can use the feature_scope_data_access_name functionality.

We show the ReadFileFeature as example. It uses the input_data ReadFile. 
In this case, we need to provide the specific reader class: CsvReader.

``` python

# This feature is already implemented as plug-in, so do not run it again. This will raise intentional errors.
class ReadFileFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return ReadFile()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        reader = cls.input_data()
        if reader is not None:
            data = reader.load(features)
            return data
        raise ValueError(f"Reading file failed for feature {features.get_name_of_one_feature()}.")
```

As a side note, the ReadFileFeature was also used for the global scope automatism.

To use it, we can simply:

```python


from typing import Optional, Any, List
from pathlib import Path

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


file_path = os.getcwd()
file_path += "/docs/docs/in_depth"

feature_list: List[Feature | str] = []
feature_list.append(
    Feature(
        name="AExample",
        # Define data access on a feature level
        options={CsvReader.get_class_name(): file_path}),
)


result = mlodaAPI.run_all(feature_list, compute_frameworks=["PandasDataframe"])
print(result)
```

Output

``` python
[  AExample
0   Value1
1   Value2]
```

Of course, we do not always want to load data during run. 
We might want to give data to the framework. 
For this purpose, we have the ApiData.

#### ApiData

The ApiData can read data given to mlodaAPI at request time. The built-in `ApiInputDataFeature` handles this - it receives data from the `ApiInputDataCollection` and makes it available as features.

Use cases:

- web requests
- real-time prediction
- features as parameters

The following example shows a simple ApiData setup.

```python

from typing import List

from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import ApiInputDataCollection

# Setup the ApiInputDataCollection and api data.
# These 2 objects are needed to relate given data to the correct features.
api_input_data_collection = ApiInputDataCollection()
api_data = api_input_data_collection.setup_key_api_data(
    key_name="ExampleApiData", api_input_data={"FeatureInputAPITest": ["TestValue3", "TestValue4"]}
)

result = mlodaAPI.run_all(
        ["FeatureInputAPITest"],
        compute_frameworks={PandasDataframe},
        api_input_data_collection=api_input_data_collection,
        api_data=api_data,
)
for res in result:
    print(res)
``` 

Output:

``` python
  FeatureInputAPITest
0          TestValue3
1          TestValue4
```

Further, we do not want to always load data from outside, be it before or during the framework run, but we want to be able to create Data. For this purpose, we have the Data Creator.

#### Data Creator

The data creator can create data independent of any other dependency. It is essentially a base feature that does not need a **DataAccessCollection** or **Feature Scope Data Access**.

Usage:

- test data,
- sample data,
- dummy data,
- parameter data

One could imagine that for experimenting one wants to see data. Then one could use this feature as input feature to another feature instead of e.g. the true data.

```python


from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator

# Create a Creator FeatureGroup class, which delivers the data needed
class AFeatureInputCreator(AbstractFeatureGroup):

    # Define input_data with using DataCreator
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"AFeatureInputCreator"})

    # Define the data this feature creates
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"AFeatureInputCreator": ["TestValue5", "TestValue6"]}

result = mlodaAPI.run_all(
        ["AFeatureInputCreator"],
        compute_frameworks={PandasDataframe},
)
for res in result:
     print(res)
```

Output

``` python
  AFeatureInputCreator
0          TestValue5
1          TestValue6
```

Finally, as also the most important way to get data, is actually to depent on data inside the framework already. For this purpose, a feature can load data depending on other features.

#### Input features 

The input_features method allows a feature to access data from other features. This enables data sharing and collaboration between different components of your system.

This is one of the key aspects in how we achieve to split data from processes. 

In the following example, we will use data from another feature.
```python

from typing import Set

from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector

# Set this variable as convention (internal key name)
_in_features = "in_features"


# First, we create a class, which uses input features from another class
class AInputFeatureGroup(AbstractFeatureGroup):

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:

        # We use the source to make this feature flexible.
        # One could give here different feature names via the configuration.
        mloda_source = options.get(_in_features)
        if mloda_source is None:
            raise ValueError(f"Option '{_in_features}' is required.")

        features = set()
        for source in mloda_source:
            features.add(Feature(name=source,                # source in this example is <AFeatureInputCreator>
                                 initial_requested_data=True # To see this feature also in the output, we can set this var to true.
                                 )
            )
        return features

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["AInputFeatureGroup"] = len(data)
        return data


feature_list = []
feature_list.append(
    Feature(name="AInputFeatureGroup", options={_in_features: frozenset(["AFeatureInputCreator"])})
)

result = mlodaAPI.run_all(
    feature_list,
    compute_frameworks={PandasDataframe},
    plugin_collector=PlugInCollector.enabled_feature_groups({AInputFeatureGroup, AFeatureInputCreator})

)
print(result)
```

Output:

``` python
[AFeatureInputCreator
0           TestValue5
1           TestValue6,    
AInputFeatureGroup
0                   2
1                   2]
.......
```

As the input features can be fulfilled by **multiple other features**, we can have the same processes running in different environments, migrations and processes.

#### Combining Data Sources with Links

When input features come from different sources (e.g., ApiData and DataCreator), you can join them using Links. Create a `Link` in `input_features()` and attach it to a `Feature`:

```python
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.feature_group.input_data.api_data.api_data import ApiInputDataFeature

class JoinedFeature(AbstractFeatureGroup):

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        # Create a LEFT join between API data and Creator data
        link = Link.left(
            (ApiInputDataFeature, Index(("api_id",))),
            (CreatorDataFeature, Index(("creator_id",)))
        )

        # Attach link to one feature, set index on both
        return {
            Feature(name="api_value", link=link, index=Index(("api_id",))),
            Feature(name="creator_value", index=Index(("creator_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Data from both sources is now joined
        data["JoinedFeature"] = data["api_value"] + "_" + data["creator_value"]
        return data
```

Available join methods: `Link.left()`, `Link.inner()`, `Link.outer()`, `Link.append()`, `Link.union()`.

> See `tests/test_plugins/integration_plugins/test_api_link_join.py` for a complete working example.
