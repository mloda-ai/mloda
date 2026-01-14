# Artifacts

Artifacts are a crucial part of the mloda, enabling the storage and retrieval of intermediate results feature engineering processes.
Example use cases are embeddings, Feature Matrices, Model Checkpoints and more.

## Overview

Artifacts are managed through a set of abstract and concrete classes that define how artifacts are created, saved, and loaded. The primary classes involved in artifact management include:

- `BaseArtifact`: The base class for all artifacts.
- `FeatureGroup`: The base class for feature groups, including methods for artifact management. This class may contain a `BaseArtfact`.

## Key Components

#### BaseArtifact

The `BaseArtifact` class serves as the foundation for all artifacts. It provides the basic interface and functionality required for artifact management.


#### FeatureGroup

The `FeatureGroup` class defines the structure for feature groups, including methods for creating data, calculating features, and managing artifacts. It includes methods such as `artifact` and `load_artifact` to handle artifact operations.

## Example

#### Feature Group with Artifact implementation

The following example demonstrates how to implement and test an artifact.

Here, we create a `FeatureGroup` with a configured `BaseArtifact`.

```python
from typing import Type, Any, Optional
from mloda.provider import FeatureGroup, FeatureSet, BaseArtifact, DataCreator, BaseInputData


class BaseExampleArtifactFeature(FeatureGroup):
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
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

api = mloda(["BaseExampleArtifactFeature"], {PyArrowTable})
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
from mloda.user import Feature, mloda

feat = Feature(name="BaseExampleArtifactFeature", options=artifacts)
api = mloda([feat], {PyArrowTable})
api._batch_run()
```

Result:

``` python
"BasicArtifact is the loaded artifact."
```

#### Testing Artifacts

Testing artifact features involves creating test cases that ensure artifacts are correctly saved and loaded. The following example shows how to test the `BaseTestArtifactFeature` class.

[View the test file on GitHub](https://github.com/mloda-ai/mloda/blob/main/tests/test_core/test_artifacts/test_artifacts.py)

#### Complex Artifact Example: SklearnArtifact

For more advanced use cases, artifacts can handle complex data structures and multiple objects. The `SklearnArtifact` demonstrates this with fitted scikit-learn transformers:

```python
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact

class MySklearnFeatureGroup(FeatureGroup):
    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        return SklearnArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Save multiple artifacts with unique keys
        if features.artifact_to_save:
            SklearnArtifact.save_sklearn_artifact(
                features, 
                "my_transformer", 
                {"fitted_transformer": fitted_model, "feature_names": ["col1", "col2"]}
            )
        
        # Load specific artifact by key
        if features.artifact_to_load:
            artifact_data = SklearnArtifact.load_sklearn_artifact(features, "my_transformer")
            fitted_model = artifact_data["fitted_transformer"]
```

This pattern supports file-based storage, multiple artifact management, and complex serialization.

#### Conclusion

Artifacts are a powerful feature in mloda, enabling efficient management of intermediate results in the machine learning pipeline. By understanding and utilizing the provided classes and methods, you can effectively manage artifacts in your projects.
