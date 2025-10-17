from typing import Any, Tuple

from pyarrow import parquet as pyarrow_parquet

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class ParquetReader(ReadFile):
    """
    Base class for Parquet file reading feature groups.

    This feature group enables reading data from Apache Parquet files, a columnar
    storage format optimized for analytical workloads. Provides highly efficient
    data loading with built-in compression and schema preservation.

    ## Supported Operations

    - `parquet_file_loading`: Load data from Parquet files with preserved schema
    - `columnar_reading`: Efficiently read only requested columns
    - `column_discovery`: Discover available columns from Parquet metadata
    - `compression_support`: Automatic handling of compressed Parquet files

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features reference column names from Parquet files:

    Examples:
    ```python
    features = [
        "transaction_id",   # Column from Parquet file
        "amount",           # Numeric column with preserved precision
        "timestamp"         # Timestamp column with timezone info
    ]
    ```

    ### 2. Configuration-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="customer_segment",
        options=Options(
            context={
                "BaseInputData": (ParquetReader, "/path/to/data.parquet")
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic Parquet Feature Access

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    # Simple column reference from Parquet file
    feature = Feature(
        name="revenue",
        options=Options(
            context={
                "BaseInputData": (ParquetReader, "sales.parquet")
            }
        )
    )
    ```


    ## Parameter Classification

    ### Context Parameters (Default)
    - `file_path`: Path to the Parquet file (absolute or relative)
    - Column names and types are preserved from Parquet schema

    ### Group Parameters
    Currently none for ParquetReader.

    ## Requirements

    - Parquet file must exist at the specified path
    - Feature names must match column names in the Parquet schema
    - All features in a FeatureSet must use the same Parquet file
    - PyArrow library must be installed

    ## Additional Notes

    - Uses PyArrow's Parquet reader for optimal performance
    - Supports .parquet, .PARQUET, .pqt, and .PQT extensions
    - Columnar format allows reading only needed columns efficiently
    - Preserves data types, compression, and encoding from source
    - Ideal for large datasets due to efficient compression
    """

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".parquet",
            ".PARQUET",
            ".pqt",
            ".PQT",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pyarrow_parquet.read_table(data_access, columns=list(features.get_all_names()))

    @classmethod
    def get_column_names(cls, file_name: str) -> Any:
        parquet_file = pyarrow_parquet.ParquetFile(file_name)
        return [column.name for column in parquet_file.schema]
