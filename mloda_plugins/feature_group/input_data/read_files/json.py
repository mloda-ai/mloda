from typing import Any

try:
    from pyarrow import json as pyarrow_json
except ImportError:
    pyarrow_json = None

from mloda.provider import FeatureSet
from mloda.user import DataType
from mloda_plugins.feature_group.input_data.read_file import ReadFile, requires_dependency


class JsonReader(ReadFile):
    """
    Base class for JSON file reading feature groups.

    This feature group enables reading data from JSON (JavaScript Object Notation) files,
    providing efficient data loading using PyArrow. Supports both line-delimited JSON
    and standard JSON formats with automatic schema detection.

    ## Supported Operations

    - `json_file_loading`: Load data from JSON files with automatic schema inference
    - `column_selection`: Select specific columns based on requested features
    - `column_discovery`: Automatically discover available fields in JSON files
    - `schema_validation`: Validate JSON structure against inferred schema

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features reference field names from JSON files:

    Examples:
    ```python
    features = [
        "user_id",          # Top-level field from JSON
        "email",            # String field from JSON
        "purchase_count"    # Numeric field from JSON
    ]
    ```

    ### 2. Configuration-Based Creation

    ```python
    from mloda.user import Feature
    from mloda.user import Options

    feature = Feature(
        name="user_name",
        options=Options(
            context={
                "BaseInputData": (JsonReader, "/path/to/data.json")
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic JSON Feature Access

    ```python
    from mloda.user import Feature
    from mloda.user import Options

    # Simple field reference from JSON file
    feature = Feature(
        name="customer_email",
        options=Options(
            context={
                "BaseInputData": (JsonReader, "customers.json")
            }
        )
    )
    ```

    ### Multiple Features from Same JSON

    ```python
    feature1 = Feature(
        name="order_id",
        options=Options(
            context={"BaseInputData": (JsonReader, "orders.json")}
        )
    )

    feature2 = Feature(
        name="order_total",
        options=Options(
            context={"BaseInputData": (JsonReader, "orders.json")}
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    - `file_path`: Path to the JSON file (absolute or relative)
    - Field names are automatically detected from JSON structure

    ### Group Parameters
    Currently none for JsonReader.

    ## Requirements

    - JSON file must exist at the specified path
    - JSON must be properly formatted (valid JSON syntax)
    - Feature names must match field names in the JSON structure
    - All features must use the same JSON file
    - PyArrow library must be installed

    ## Additional Notes

    - Uses PyArrow's JSON reader for efficient parsing
    - Supports both .json and .JSON file extensions
    - Schema is inferred from a sample of data for efficiency
    - Unexpected fields result in errors to ensure data quality
    - Only requested fields are loaded into memory
    """

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (
            ".json",
            ".JSON",
        )

    @classmethod
    @requires_dependency("pyarrow_json", file_format="JSON")
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        result = pyarrow_json.read_json(
            data_access,
            parse_options=pyarrow_json.ParseOptions(
                explicit_schema=None,
                unexpected_field_behavior="error",
            ),
        )
        return result.select(list(features.get_all_names()))

    @classmethod
    @requires_dependency("pyarrow_json")
    def get_column_names(cls, file_name: str) -> Any:
        # block_size is a chunking granularity, not a row-count sample; this parses the whole file.
        read_options = pyarrow_json.ReadOptions(block_size=65536)
        table = pyarrow_json.read_json(file_name, read_options=read_options)
        return table.schema.names

    @classmethod
    @requires_dependency("pyarrow_json")
    def describe_columns(cls, data_access: Any) -> dict[str, DataType | None]:
        read_options = pyarrow_json.ReadOptions(block_size=65536)
        table = pyarrow_json.read_json(data_access, read_options=read_options)
        return {field.name: DataType.from_arrow_type_safe(field.type) for field in table.schema}
