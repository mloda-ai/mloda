# Artifacts

Artifacts are a crucial part of the mloda, enabling the storage and retrieval of intermediate results feature engineering processes.
Example use cases are embeddings, Feature Matrices, Model Checkpoints and more.

## Overview

Artifacts are managed through a set of abstract and concrete classes that define how artifacts are created, saved, and loaded. The primary classes involved in artifact management include:

- `BaseArtifact`: The base class for all artifacts.
- `AbstractFeatureGroup`: An abstract class that defines the structure for feature groups, including methods for artifact management. This class may contain a `BaseArtfact`.

## Key Components

#### BaseArtifact

The `BaseArtifact` class serves as the foundation for all artifacts. It provides the basic interface and functionality required for artifact management.


#### AbstractFeatureGroup

The `AbstractFeatureGroup` class defines the structure for feature groups, including methods for creating data, calculating features, and managing artifacts. It includes methods such as `artifact` and `load_artifact` to handle artifact operations.

## Example

#### Feature Group with Artifact implementation

The following example demonstrates how to implement and test an artifact.

Here, we create a `FeatureGroup` with a configured `BaseArtifact`.

```python
from typing import Type, Any, Optional
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData


class BaseExampleArtifactFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        return BaseArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = "BasicArtifact"

        if features.artifact_to_load:
            result = cls.load_artifact(features)
            print(f"{result} is the loaded artifact.")

        return {cls.get_class_name(): [1, 2, 3]}
```

Now, we run the query to the feature group to save the artifact. This example is very basic, but could be a much more complex artifact.

```python
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable

api = mlodaAPI(["BaseExampleArtifactFeature"], {PyarrowTable})
api._batch_run()
artifacts = api.get_artifacts()
print(artifacts)
```

Result:

``` python
{'BaseExampleArtifactFeature': 'BasicArtifact'}
```

Now, let us use this artifact.

```python
from mloda_core.abstract_plugins.components.feature import Feature

feat = Feature(name="BaseExampleArtifactFeature", options=artifacts)
api = mlodaAPI([feat], {PyarrowTable})
api._batch_run()
```

Result:

``` python
"BasicArtifact is the loaded artifact."
```

#### Testing Artifacts

Testing artifact features involves creating test cases that ensure artifacts are correctly saved and loaded. The following example shows how to test the `BaseTestArtifactFeature` class.

[View the test file on GitHub](https://github.com/TomKaltofen/mloda/tree/feature/main/tests/test_core/test_artifacts/test_artifacts.py)


#### Conclusion

Artifacts are a powerful feature in mloda, enabling efficient management of intermediate results in the machine learning pipeline. By understanding and utilizing the provided classes and methods, you can effectively manage artifacts in your projects.
