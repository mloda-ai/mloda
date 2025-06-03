# Data Quality

Ensuring data quality is crucial for the success of any data-driven project. This document outlines the various aspects of data quality and the measures taken to maintain it.

## Feature validation

Feature validation ensures that the input and output features meet the expected standards and requirements.

#### Validate Input Features

Input features are validated to ensure they conform to the expected formats, ranges, and distributions. This includes:

-   Checking for missing values and handling them appropriately.
-   Validating data types and ranges.
-   Ensuring data consistency and integrity.

Example: Input Feature Validation Using Custom Validators

###### Simple validator

```python
from typing import Any, Optional, Set
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable


class DocSimpleValidateInputFeatures(AbstractFeatureGroup):

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature(name="BaseValidateInputFeaturesBase", options=options)}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function is a naive implementation of a validator."""

        if len(data["BaseValidateInputFeaturesBase"]) == 3:
            raise ValueError("Data should have 3 elements")
        return True
```

As we run it, it will return an error.

``` python
results = mlodaAPI.run_all(
            ["DocSimpleValidateInputFeatures"], {PyarrowTable}
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
    def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function should be used to validate the input data."""

        validation_rules = {
            "BaseValidateInputFeaturesBase": Column(int, Check.in_range(1, 2)),
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

        return validator.validate(data)  # type: ignore
```

The DocExamplePanderaValidator is based on the BaseValidator, which provides basic functionalities around logging. 

We show here an example by using the tool [Pandera](https://github.com/unionai-oss/pandera).

```python
from mloda_core.abstract_plugins.components.base_validator import BaseValidator
import pyarrow as pa
from pandera import pandas
from pandera import Column, Check
from pandera.errors import SchemaError


class DocExamplePanderaValidator(BaseValidator):
    """Custom validator to validate input features based on a specific rule."""

    def validate(self, data: pa.Table) -> Optional[bool]:
        """This function should be used to validate the input data."""

        # Convert PyArrow Table to Pandas DataFrame if necessary
        if isinstance(data, pa.Table):  # If the data is a PyArrow Table
            data = data.to_pandas()

        schema = pandas.DataFrameSchema(self.validation_rules)

        try:
            schema.validate(data)
        except SchemaError as e:
            self.handle_log_level("SchemaError:", e)
        return True
```

The validator should raise an error again.

``` python
results = mlodaAPI.run_all(
            ["DocCustomValidateInputFeatures"], {PyarrowTable}
        )
```

###### Log only validator and Extender use

-   We throw a warning instead of raising an error.
-   We use the extender functionality to print out the runtime as an example.

```python
from mloda_core.abstract_plugins.function_extender import WrapperFunctionEnum, WrapperFunctionExtender
from tests.test_documentation.test_documentation import DokuValidateInputFeatureExtender

example_feature = Feature("DocCustomValidateInputFeatures", {"ValidationLevel": "warning"})

results = mlodaAPI.run_all(
            [example_feature], {PyarrowTable}, function_extender={DokuValidateInputFeatureExtender()}
        )
```
This time it does not raise an error, we should see the following output:
```python
"Time taken: 0.19909930229187012"
```

#### Validate Output Features

Output features are validated to ensure they meet the expected outcomes and performance metrics. This includes:

- Comparing output features against expected results.
- Validating the statistics of data.
- Ensuring the output data has the right types.


**The implementation and use is very similar to validating input features.**

###### Simple validator

```python
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from tests.test_plugins.integration_plugins.test_validate_features.example_validator import BaseValidateOutputFeaturesBase


class DocBaseValidateOutputFeaturesBase(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def validate_output_features(cls, data: Any, config: Options) -> Optional[bool]:
        """This function should be used to validate the output data."""

        if len(data[cls.get_class_name()]) != 3:
            raise ValueError("Data should have 3 elements")
        return True

results = mlodaAPI.run_all(
            ["DocBaseValidateOutputFeaturesBase"], {PyarrowTable}
        )
results
```
As this case works, we should not see an error. However, we see how similar the functionalities of input and output validations are.

###### Loading a validator based on BaseValidator

After this simple validation, let's reuse the pandera example from before.

```python
class DocBaseValidateOutputFeaturesBaseNegativePandera(DocBaseValidateOutputFeaturesBase):
    """Pandera example test case. This one is related to the pandera testcase for validate_input_features."""

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function should be used to validate the output data."""

        validation_rules = {
            cls.get_class_name(): Column(int, Check.in_range(1, 2)),
        }
        validator = DocExamplePanderaValidator(validation_rules, features.get_options_key("ValidationLevel"))
        return validator.validate(data)
```

This one should fail:

``` python
results = mlodaAPI.run_all(
            ["DocBaseValidateOutputFeaturesBaseNegativePandera"], {PyarrowTable}
        )
```

###### Log only validator and Extender use

We can of course also use an extender, which was defined somewhere else.

```python
from tests.test_plugins.integration_plugins.test_validate_features.test_validate_output_features import ValidateOutputFeatureExtender

results = mlodaAPI.run_all(
            ["DocBaseValidateOutputFeaturesBase"], {PyarrowTable},
            function_extender={ValidateOutputFeatureExtender()}
        )
```

Output similar to:
```python
"Time taken: 3.409385681152344e-05"
```

#### Artifacts

`Artifacts` can also be used for validation as the full API is available. A use case could be to store statistics of a feature and then validate them later on.
For more details on artifacts, refer to the [artifact documentation](https://tomkaltofen.github.io/mloda/in_depth/artifacts/).

#### Conclusion

In conclusion, feature validation is crucial for ensuring data quality in both input and output stages. By leveraging custom validators and extenders, validation can be tailored to specific needs while maintaining flexibility. This process helps detect inconsistencies early, improving the accuracy and robustness of data  and feature pipelines.

## Software Testing

Software testing is an integral part of maintaining data quality. It ensures that the software components used in data processing and analysis are functioning correctly.

#### Unit tests

Unit tests are written to test individual components of the software. These tests ensure that each function and method works as expected in isolation. Unit tests are typically run using a testing framework like `pytest`.

#### Integration Tests

Integration tests are used to test the interaction between different components of the software. These tests ensure that the components work together as expected and that data flows correctly through the system.

## Data Comparison

Data comparison involves comparing different datasets to ensure consistency and accuracy. This includes:
- Comparing new datasets with historical data to identify any discrepancies.
- Validating data transformations to ensure they produce the expected results.
- Using statistical methods to compare distributions and identify anomalies.
