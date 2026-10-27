## Access (feature) data

This framework provides several structured mechanisms for features to access and manage data, catering to diverse needs. Features can retrieve data through:

-   **DataAccessCollection** for global data loading management, 
-   **Feature Scope Data Access** for feature-specific data loading management, 
-   **Api Data** for using data directly with the mlodaAPI request.
-   **DataCreatorFeatureGroup** for generating data instead of loading data,
-   **Input Features** facilitates data sharing between features.

These methods ensure efficient data management while maintaining flexibility and scalability. A typical scenario involves a complex feature relying on three input features. These input features, in turn, may depend on other input features or load request data through `ApiDataFG`.

> **Advanced**: For a detailed explanation of the underlying data access patterns (source FeatureGroups), see [Data Access Patterns](data-access-patterns.md).

#### DataAccessCollection - global data access

The DataAccessCollection is designed to control the access to data of any kind. The main purpose of this class is to organize and simplify interactions with these different data elements, making it easier to work to ingest data of various forms into the framework. It serves as an interface for accessing and storing of data on a global level.

The DataAccessCollection can only be added via **mlodaAPI**.

A DAC holds resources of four kinds:

1.  **Files:** the exact location of files, e.g. `path/folder/text.txt`.
2.  **Folders:** directories where files are located, e.g. `path/folder/`.
3.  **Credentials:** the information needed to reach a data source, expressed as a typed `Credential` (recommended) or a plain dict, e.g. `Credential(host="example.com", password="example")`.
4.  **Connections:** already-initialized connection objects (database connections, Spark sessions, Iceberg catalogs, etc.).

When the collection holds more than one resource of the same kind, you can name them with stable handles and disambiguate per feature via `Options(context={"data_access_handle": "..."})`. See [Named Data Access Handles](named-data-access-handles.md) for the full naming model, the resolution rule, and the available error shapes; for the simple single-source cases below, naming is optional.

Any feature group can also call `cls.read_credential(handle)` in `calculate_feature`, `validate_input_features`, or `validate_output_features`. Register a named `Credential` in the collection and carry only its handle in an option; the value stays out of feature options, hashes and plan output. This complements source groups' match-time credential binding. See [Runtime credential access](named-data-access-handles.md#runtime-credential-access) for an example and the scope of error redaction.

You can apply these options like so:

```py
from mloda.user import Credential, DataAccessCollection

data_access = DataAccessCollection()

# Add file paths, folder paths, credentials, and connection objects
data_access.add_file('path/to/folder/text.txt')
data_access.add_folder('path/to/folder/')
data_access.add_credentials(Credential(host='example.com', password='example'))
data_access.add_connection('InitializedDBConnection')

mloda.run_all(
    feature_list,
    data_access_collection=data_access)
```

#### Global Scope Data Access

A concrete, simplified global scope data access is shown in this example:

```python
import os

from mloda.user import mloda, PluginCollector
from mloda.user import DataAccessCollection
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG


file_path = os.getcwd()
file_path += "/docs/docs/in_depth"

data_access_collection = DataAccessCollection(folders={str(file_path)})

result = mloda.run_all(
            ["AExample", "BExample"],
            compute_frameworks=["PandasDataFrame"],
            # Define data access on a global level
            data_access_collection=data_access_collection,
            plugin_collector=PluginCollector.enabled_feature_groups({CsvFG}),
        )
print(result)
```

Output

```text
[  AExample  BExample
0   Value1         2
1   Value2         3]
```

#### Document source groups

For unstructured files, mloda ships one group per format: `TextFG` (`.text`/`.txt`/`.TXT`), `PyFG` (`.py`),
`MarkdownFG` (`.md`), `YamlFG` (`.yaml`/`.yml`) and `JsonDocumentFG` (`.json`). Each answers three names:
`TextFG` (the content), `TextFG~source` (the path) and `TextFG~file_type` (the file's actual suffix, lowercased, without the dot).
A request reads one file; with several matching files, pick one with a `data_access_handle`, a
`column_to_file` entry or `options={"TextFG": path}`.

To read a `.json` file as a document, list it in `document_suffixes`; `JsonFG` then steps aside and
`JsonDocumentFG` claims it:

```py
Feature("JsonDocumentFG", options={"document_suffixes": frozenset({".json"})})
```

#### Disambiguating columns shared across multiple files

When a `DataAccessCollection` holds multiple files that share the same column name (e.g. `id` appears in both `application_train.csv` and `bureau.csv`), the resolver refuses to guess: it raises `ValueError` listing the candidate files and asks for `data_access_handle`. `column_to_file` is a per-column shortcut that binds columns to files at construction so you don't have to set `data_access_handle` on every feature.

Use `column_to_file` to pin each column to its canonical file:

```py
from mloda.user import DataAccessCollection, mloda

data_access = DataAccessCollection(
    files={"train": "application_train.csv", "bureau": "bureau.csv"},
    column_to_file={
        "SK_ID_CURR": "train",
        "TARGET":     "train",
        "AMT_CREDIT_SUM": "bureau",
    },
)

result = mloda.run_all(
    ["SK_ID_CURR", "TARGET", "AMT_CREDIT_SUM"],
    compute_frameworks=["PyArrowTable"],
    data_access_collection=data_access,
)
```

Rules:
- A `column_to_file` pin is authoritative once it applies to a requested feature: it short-circuits the resolver and takes precedence over `data_access_handle`, regardless of whether the `DataAccessCollection` is global- or per-feature-scoped.
- A file or folder pointer such as `options={"CsvFG": path}` that disagrees with the pin raises; drop one of them.
- Values may be either a file handle (a key of the `files` dict) or a file path (a value of the `files` dict); they are normalized to handles internally. Construction raises `ValueError` if a value matches neither.
- For columns not listed in the map, the consumer falls back to the shared resolver: a single matching file binds, multiple matching files raise `ValueError` listing the candidates. Set `data_access_handle` on the feature's `Options` to disambiguate without `column_to_file`. See [Named Data Access Handles](named-data-access-handles.md).

The existing per-feature alternative still works for one-off pinning:

```py
Feature("SK_ID_CURR", options=Options({"CsvFG": "application_train.csv"}))
```

#### Feature Scope Data Access

The Feature Scope Data access is instead designed to control the access to data of any kind 
on a local level. 

If data needs to be added specifically for a single feature (or features from the same feature group), you can point the feature at its own source through its options.

We show `CsvFG` as example, a `ReadFileFG` subclass that claims CSV files.
In this case, we point it at a path with its class name as option key: `CsvFG`.

```py
# CsvFG is already implemented as plugin, so do not define it again.
class CsvFG(ReadFileFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".csv", ".CSV")
```

As a side note, `CsvFG` is also used for the global scope automatism.

To use it, we can simply:

```python
import os
from typing import Any
from pathlib import Path

from mloda.user import mloda
from mloda.user import Feature
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG


file_path = os.getcwd()
file_path += "/docs/docs/in_depth"

feature_list: list[Feature | str] = []
feature_list.append(
    Feature(
        name="AExample",
        # Define data access on a feature level
        options={CsvFG.get_class_name(): file_path}),
)


result = mloda.run_all(feature_list, compute_frameworks=["PandasDataFrame"])
print(result)
```

Output

```text
[  AExample
0   Value1
1   Value2]
```

Of course, we do not always want to load data during run. 
We might want to give data to the framework. 
For this purpose, mloda accepts request data through the `api_data` parameter.

#### ApiDataFG

The built-in `ApiDataFG` (a `SourceFeatureGroup`) reads data given to mloda at request time. It claims a feature whose name is a column of a set in the `api_data` parameter, and loads it through the `INPUT_DATA_LOAD` hook with `data_access_identity` `api_data:<set key>`.

`Options({"ApiDataFG": "<set key>"})` narrows a feature to one set; an unknown key aborts. `prepare()` fixes each set's columns, and `run(api_data=...)` must give the same set keys and the same column names per set (order may differ), otherwise a `ValueError` is raised before the run starts. A FeatureGroup computing a name that is also an `api_data` column is an ambiguity error. For a custom transport, subclass `ApiDataFG` and override `load_neutral`, reading the run's values with `cls.api_values(match)`. A subclass overriding `load_neutral` or `api_values` declares `neutral_type` itself; one building its own `ApiDataRef` passes `value_type`.

Without a `compute_frameworks` list, a set given as a dict of lists runs on `PythonDictFramework` (reason `loads natively`) when the dict framework can serve the group, and consumers that can run there follow, so the result is a dict unless `output_framework` converts it. Sets with other column types (tuples, arrays) keep today's choice. A pandas-style group without `compute_framework_rule` then also runs on dicts: declare the rule or pass `compute_frameworks`. On a rerun, a set whose step loads natively as a dict must stay a dict of lists. Plan locks of default api_data runs written before this release no longer match; rewrite them with `write_plan_lock`.

Use cases:

- web requests
- real-time prediction
- features as parameters

The following example shows a simple `api_data` setup.

```python
from mloda.user import mloda
from mloda.user.pandas import PandasDataFrame

# Pass api_data directly as a dict - the framework handles registration internally
result = mloda.run_all(
        ["FeatureInputAPITest"],
        compute_frameworks=[PandasDataFrame],
        api_data={"ExampleApiData": {"FeatureInputAPITest": ["TestValue3", "TestValue4"]}},
)
for res in result:
    print(res)
```

Output:

```text
  FeatureInputAPITest
0          TestValue3
1          TestValue4
```

Further, we do not want to always load data from outside, be it before or during the framework run, but we want to be able to create Data. For this purpose, we have the `DataCreatorFeatureGroup`.

#### DataCreatorFeatureGroup

Subclass `DataCreatorFeatureGroup` and return the data from `calculate_feature`; a dict result becomes its own frame. The class name always matches; add more names via `feature_names_supported()`. It needs no **DataAccessCollection** or **Feature Scope Data Access**.

Usage:

- test data,
- sample data,
- dummy data,
- parameter data

One could imagine that for experimenting one wants to see data. Then one could use this feature as input feature to another feature instead of e.g. the true data.

```python
from mloda.user import mloda
from mloda.provider import FeatureGroup, FeatureSet, DataCreatorFeatureGroup
from mloda.user.pandas import PandasDataFrame

# Create a Creator FeatureGroup class, which delivers the data needed
class AFeatureInputCreator(DataCreatorFeatureGroup):

    # Define the data this feature creates
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"AFeatureInputCreator": ["TestValue5", "TestValue6"]}

result = mloda.run_all(
        ["AFeatureInputCreator"],
        compute_frameworks=[PandasDataFrame],
)
for res in result:
     print(res)
```

Output

```text
  AFeatureInputCreator
0          TestValue5
1          TestValue6
```

Finally, as also the most important way to get data, is actually to depend on data inside the framework already. For this purpose, a feature can load data depending on other features.

#### Input features 

The input_features method allows a feature to access data from other features. This enables data sharing and collaboration between different components of your system.

This is one of the key aspects in how we achieve to split data from processes. 

In the following example, we will use data from another feature.
```python
from mloda.user import mloda
from mloda.provider import FeatureGroup, FeatureSet
from mloda.user import Options, FeatureName, Feature, PluginCollector

# Set this variable as convention (internal key name)
_in_features = "in_features"


# First, we create a class, which uses input features from another class
class AInputFeatureGroup(FeatureGroup):

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:

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

result = mloda.run_all(
    feature_list,
    compute_frameworks=[PandasDataFrame],
    plugin_collector=PluginCollector.enabled_feature_groups({AInputFeatureGroup, AFeatureInputCreator})

)
print(result)
```

Output:

```text
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

When input features come from different sources (e.g., ApiDataFG and a DataCreatorFeatureGroup), you can join them using Links. Create a `Link` in `input_features()` and attach it to a `Feature`:

```python
from mloda.user import Link, Index, Feature, JoinSpec
from mloda.provider import FeatureGroup, FeatureSet
from mloda.provider import ApiDataFG

class JoinedFeature(FeatureGroup):

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        # Create a LEFT join between mloda data and Creator data
        link = Link.left(
            JoinSpec(ApiDataFG, Index(("api_id",))),
            JoinSpec(CreatorDataFeature, Index(("creator_id",)))
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
