# Artifacts

Artifacts store and retrieve intermediate results (embeddings, feature matrices, model checkpoints). A `FeatureGroup` declares its artifact class (a `BaseArtifact` subclass) through `artifact()` and uses `load_artifact` to read it back.

## Example

#### Feature Group with Artifact implementation

The following example demonstrates how to implement and test an artifact.

Here, we create a `FeatureGroup` with a configured `BaseArtifact`.

```python
from typing import Any
from mloda.provider import FeatureGroup, FeatureSet, BaseArtifact, DataCreatorFeatureGroup


class BaseExampleArtifactFeature(DataCreatorFeatureGroup):

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return BaseArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save is not None:
            features.save_artifact = "BasicArtifact"

        if features.artifact_to_load is not None:
            result = cls.load_artifact(features)
            print(f"{result} is the loaded artifact.")

        return {cls.get_class_name(): [1, 2, 3]}
```

Now, we run the query to the feature group to save the artifact. This example is very basic, but could be a much more complex artifact. The saver runs right after `calculate_feature` returns, before transform, filtering and output validation, so a step that fails after `calculate_feature` has already saved; a saver with external side effects should tolerate that.

```python
from mloda.user import mloda
from mloda.user.pyarrow import PyArrowTable

api = mloda(["BaseExampleArtifactFeature"], [PyArrowTable])
artifacts = api.run().artifacts
print(artifacts)
```

Result:

```text
{'BaseExampleArtifactFeature': 'BasicArtifact'}
```

Now, let us use this artifact.

```python
from mloda.user import Feature, mloda

feat = Feature(name="BaseExampleArtifactFeature", options=artifacts)
mloda.run_all([feat], [PyArrowTable])
```

Result:

```text
"BasicArtifact is the loaded artifact."
```

## Run-time Artifact Switching with prepare/run

The two-phase API (`prepare()` + `run()`) supports switching between save and
load mode across `run()` calls. Pass a previously saved artifact dict to
`run(artifacts=...)` to switch to load mode for that run without re-preparing.

This is useful for train-then-predict workflows and cross-validation, where a
single session alternates between training (save) and inference (load).

#### Train-then-predict in a single session

```python
from mloda.user import mloda
from mloda.user.pyarrow import PyArrowTable

# 1. Prepare once
session = mloda.prepare(["BaseExampleArtifactFeature"], [PyArrowTable])

# 2. First run: save mode (train)
saved_artifacts = session.run().artifacts
# {'BaseExampleArtifactFeature': 'BasicArtifact'}

# 3. Second run: load mode (predict), using artifacts from step 2
result = session.run(artifacts=saved_artifacts)
```

When `artifacts` is not provided (the default), the artifact mode from
`prepare()` is used. Baking an artifact into `Options` at prepare time and also
supplying it via `run(artifacts=...)` for the same feature are mutually
exclusive: resolving a runtime artifact when the key is already present in
`Options.group` raises, rather than silently picking one over the other.

Runtime keys are validated at run start against `session.artifact_inputs`: an
unknown key and the `Options.group` conflict are refused before any step runs.
A feature set without a runtime artifact keeps its prepare-time mode. Produced
artifacts come back only in `RunResult.artifacts`; the plan is never written.
Runtime keys must be the feature names the plan lists in `session.artifact_inputs`
(a custom saver returning a dict keyed by other names cannot be passed back as
is), and a loaded artifact value is shared by concurrent runs, so treat it as
read-only in `calculate_feature`.

A custom `custom_loader` reads `features.artifact_value`
(the prepare-time Options value or this run's `artifacts` value, bound for this
FeatureSet's one artifact key, see below), not Options.

#### Several artifacts from one saver

The saver's return is split per key only when `features.save_artifact` is a dict
and the saver returns a dict. Each key then becomes its own entry in
`RunResult.artifacts`, holding that key's value (for example a path). Otherwise
the return is one artifact under the set's alphabetically first feature name.

```py
@classmethod
def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
    paths = {}
    for name, model in artifact.items():  # artifact is features.save_artifact
        paths[name] = f"/models/{name}.bin"
        write_model(model, paths[name])
    return paths

# save_artifact = {"feature_a": model_a, "feature_b": model_b} gives
# result.artifacts == {"feature_a": "/models/feature_a.bin", "feature_b": "/models/feature_b.bin"}
```

- Return only keys in `features.get_all_names()`. A key that is not a feature name
  is refused by `run(artifacts=...)` and ignored on the Options path, where the set
  saves again. Features of one group can land in separate FeatureSets, and a key
  another set also returns raises.
- One binding per FeatureSet: on load (Options or `run(artifacts=...)`), the set
  binds the alphabetically first of its feature names that has a value.
  `features.artifact_to_load` is that name and `features.artifact_value` is its
  value only. The other keys are not bound, and passing them raises nothing.
- A `custom_loader` serving several features must load the others itself, for
  example from storage it controls, as `SklearnArtifact.custom_loader` does. Do not
  read them from `features.options`: they are absent on the runtime path.

#### Testing Artifacts

[View the test file on GitHub](https://github.com/mloda-ai/mloda/blob/main/tests/test_core/test_artifacts/test_artifacts.py)

#### Complex Artifact Example: SklearnArtifact

For more advanced use cases, artifacts can handle complex data structures and multiple objects. The `SklearnArtifact` demonstrates this with fitted scikit-learn transformers:

```python
from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact

class MySklearnFeatureGroup(FeatureGroup):
    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return SklearnArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Key each artifact by a feature name, so it can be passed back to run(artifacts=...)
        key = features.get_name_of_one_feature()
        if features.artifact_to_save is not None:
            SklearnArtifact.save_sklearn_artifact(
                features, 
                key, 
                {"fitted_transformer": fitted_model, "feature_names": ["col1", "col2"]}
            )
        
        # Load specific artifact by key
        if features.artifact_to_load is not None:
            artifact_data = SklearnArtifact.load_sklearn_artifact(features, key)
            fitted_model = artifact_data["fitted_transformer"]
```
