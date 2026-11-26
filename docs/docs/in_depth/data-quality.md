# Data Quality

## Feature validation

Feature validation checks that input and output features meet expectations.

#### Validate Input Features

Before `validate_input_features` and any extender on that hook, core checks that every declared input exists in the incoming data and raises `MissingInputFeatureError` (a `ValueError`) otherwise. Root steps and schema-less empty data are skipped.

Example: Input Feature Validation Using Custom Validators

###### Simple validator

```python
from typing import Any
from mloda.user import mloda
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Options, FeatureName, Feature
from mloda.user.pyarrow import PyArrowTable


class DocBaseValidateInputFeaturesBase(DataCreatorFeatureGroup):

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}


class DocSimpleValidateInputFeatures(FeatureGroup):

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(name="DocBaseValidateInputFeaturesBase", options=options)}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        """This function is a naive implementation of a validator."""

        if len(data["DocBaseValidateInputFeaturesBase"]) != 3:
            raise ValueError("Data should have 3 elements")
```

As we run it, it will return an error.

```text
results = mloda.run_all(
            ["DocSimpleValidateInputFeatures"], [PyArrowTable]
        )
ValueError: Data should have 3 elements
```

###### Loading a validator based on BaseValidator

In the following example, we replace the validate input features function.
This function shows 2 examples:
    
- Loading a validator from the feature config.
- Instantiating a validator inplace.

```python
class DocCustomValidateInputFeatures(DocSimpleValidateInputFeatures):

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        """This function should be used to validate the input data."""

        validation_rules = {
            "DocBaseValidateInputFeaturesBase": Column(int, Check.in_range(1, 2)),
        }

        # Loading from feature config
        if features.get_options_key("DocExamplePanderaValidator") is not None:
            validator = features.get_options_key("DocExamplePanderaValidator")
            if not isinstance(validator, DocExamplePanderaValidator):
                raise ValueError("DocExamplePanderaValidator should be an instance of DocExamplePanderaValidator")
        else:
            validation_log_level = features.get_options_key("ValidationLevel")
        # Instantiating a validator inplace
            validator = DocExamplePanderaValidator(validation_rules, validation_log_level)

        validator.validate(data)
```

The DocExamplePanderaValidator is based on the BaseValidator, which provides basic functionalities around logging. 

We show here an example by using the tool [Pandera](https://github.com/unionai-oss/pandera).

```python
from mloda.provider import BaseValidator
import pyarrow as pa
from pandera import pandas
from pandera import Column, Check
from pandera.errors import SchemaError


class DocExamplePanderaValidator(BaseValidator):
    """Custom validator to validate input features based on a specific rule."""

    def validate(self, data: pa.Table) -> None:
        """This function should be used to validate the input data."""

        # Convert PyArrow Table to Pandas DataFrame if necessary
        if isinstance(data, pa.Table):  # If the data is a PyArrow Table
            data = data.to_pandas()

        schema = pandas.DataFrameSchema(self.validation_rules)

        try:
            schema.validate(data)
        except SchemaError as e:
            self.handle_log_level("SchemaError:", e)
```

The validator should raise an error again.

```text
results = mloda.run_all(
            ["DocCustomValidateInputFeatures"], [PyArrowTable]
        )
SchemaError: Column 'DocBaseValidateInputFeaturesBase' failed element-wise validator: in_range(1, 2)
```

###### Log only validator and Extender use

-   We throw a warning instead of raising an error.
-   We use the extender functionality to print out the runtime as an example.

```python
from mloda.user import mloda
from mloda.steward import Extender, ExtenderHook
from mloda.user import Feature
import time


class DokuValidateInputFeatureExtender(Extender):
    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time taken: {time.time() - start}")
        return result

example_feature = Feature("DocCustomValidateInputFeatures", {"ValidationLevel": "warning"})

results = mloda.run_all(
            [example_feature], [PyArrowTable], function_extender={DokuValidateInputFeatureExtender()}
        )
```
This time it does not raise an error, we should see the following output:
```text
"Time taken: 0.19909930229187012"
```

#### Validate Output Features

###### Simple validator

```python
from mloda.user import mloda
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet


class DocBaseValidateOutputFeaturesBase(DataCreatorFeatureGroup):

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        """This function should be used to validate the output data."""

        if len(data[cls.get_class_name()]) != 3:
            raise ValueError("Data should have 3 elements")

results = mloda.run_all(
            ["DocBaseValidateOutputFeaturesBase"], [PyArrowTable]
        )
results
```
As this case works, we should not see an error.
`validate_output_features(cls, data, features)` works like the input hook, with the same validators; wrap it with `ExtenderHook.VALIDATE_OUTPUT_FEATURE`.

#### Artifacts

`Artifacts` can also be used for validation as the full mloda is available. A use case could be to store statistics of a feature and then validate them later on.
For more details on artifacts, refer to the [artifact documentation](artifacts.md).

#### Empty-Result Guard

`run_validate_output_features` runs a schema-presence check **before** data-type enforcement and before the custom `validate_output_features` hook. A result with no schema (zero columns) raises `EmptyResultError` immediately, so custom output validators are not reached on it; a zero-*row* result with a valid schema flows through normally.

For details on the underlying schema-presence gate, see
[Empty Results](compute-framework-integration.md#empty-results).
