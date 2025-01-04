from typing import Optional

from pandera import DataFrameSchema
import pyarrow as pa
from pandera.errors import SchemaError

from mloda_core.abstract_plugins.components.base_validator import BaseValidator


class ExamplePanderaValidator(BaseValidator):
    """Custom validator to validate input features based on a specific rule."""

    def validate(self, data: pa.Table) -> Optional[bool]:
        """This function should be used to validate the input data."""

        # Convert PyArrow Table to Pandas DataFrame if necessary
        if isinstance(data, pa.Table):  # If the data is a PyArrow Table
            data = data.to_pandas()

        schema = DataFrameSchema(self.validation_rules)

        try:
            schema.validate(data)
        except SchemaError as e:
            self.handle_log_level("SchemaError:", e)
        return True
