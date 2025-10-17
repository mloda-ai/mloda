from typing import Any, Tuple

from pyarrow import csv as pyarrow_csv

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class CsvReader(ReadFile):
    """
    Base class for CSV file reading feature groups.

    This feature group enables reading data from CSV (Comma-Separated Values) files,
    providing efficient data loading using PyArrow for optimal performance. It
    automatically detects column names and supports various CSV formats.

    ## Supported Operations

    - `csv_file_loading`: Load data from CSV files with automatic schema detection
    - `column_selection`: Select specific columns based on requested features
    - `column_discovery`: Automatically discover available columns in CSV files
    - `case_insensitive_extensions`: Support both .csv and .CSV file extensions

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features reference column names from CSV files:

    Examples:
    ```python
    features = [
        "customer_id",      # Column from CSV file
        "email_address",    # Email column from CSV
        "total_purchases"   # Numeric column from CSV
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with file path configuration:

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="customer_name",
        options=Options(
            context={
                "BaseInputData": (CsvReader, "/path/to/data.csv")
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic CSV Feature Access

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    # Simple column reference from CSV file
    feature = Feature(
        name="user_email",
        options=Options(
            context={
                "BaseInputData": (CsvReader, "users.csv")
            }
        )
    )
    ```

    ### Multiple Features from Same CSV

    ```python
    # Load multiple columns from the same CSV file
    feature1 = Feature(
        name="customer_id",
        options=Options(
            context={
                "BaseInputData": (CsvReader, "customers.csv")
            }
        )
    )

    feature2 = Feature(
        name="customer_name",
        options=Options(
            context={
                "BaseInputData": (CsvReader, "customers.csv")
            }
        )
    )

    feature3 = Feature(
        name="registration_date",
        options=Options(
            context={
                "BaseInputData": (CsvReader, "customers.csv")
            }
        )
    )
    ```

    ### Using DataAccessCollection

    ```python
    from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection

    # Configure file access at the collection level
    data_access = DataAccessCollection(files=["/data/sales.csv"])

    # Features will automatically use the configured file
    feature = Feature(name="revenue")
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `file_path`: Path to the CSV file (absolute or relative)
    - Column names are automatically detected from the CSV header row

    ### Group Parameters
    Currently none for CsvReader. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Requirements

    - CSV file must exist at the specified path
    - CSV file must have a header row with column names
    - File must be readable and properly formatted
    - Feature names must match column names in the CSV header
    - All features must use the same CSV file
    - PyArrow library must be installed

    ## Additional Notes

    - Uses PyArrow's CSV reader for efficient parsing and memory usage
    - Automatically detects column types and schema
    - Supports both .csv and .CSV file extensions
    - Only requested columns are loaded into memory for efficiency
    - The reader automatically skips the header row after reading column names
    """

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".csv",
            ".CSV",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        result = pyarrow_csv.read_csv(data_access)
        return result.select(list(features.get_all_names()))

    @classmethod
    def get_column_names(cls, file_name: str) -> Any:
        read_options = pyarrow_csv.ReadOptions(skip_rows_after_names=1)
        table = pyarrow_csv.read_csv(file_name, read_options=read_options)
        return table.schema.names
