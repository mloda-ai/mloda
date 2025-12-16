from typing import Any, Optional

from pandera import Column, Check

from mloda import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from pandera import pandas
import pyarrow as pa
from pandera.errors import SchemaError

from mloda.provider import BaseValidator


class ExamplePanderaValidator(BaseValidator):
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


class BaseValidateOutputFeaturesBase(FeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function should be used to validate the output data."""

        if len(data[cls.get_class_name()]) != 3:
            raise ValueError("Data should have 3 elements")
        return True


class BaseValidateOutputFeaturesBaseNegativePandera(BaseValidateOutputFeaturesBase):
    """Pandera example test case. This one is related to the pandera testcase for validate_input_features."""

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        """This function should be used to validate the output data."""

        validation_rules = {
            cls.get_class_name(): Column(int, Check.in_range(1, 2)),
        }

        validator = ExamplePanderaValidator(validation_rules, features.get_options_key("ValidationLevel"))
        return validator.validate(data)
