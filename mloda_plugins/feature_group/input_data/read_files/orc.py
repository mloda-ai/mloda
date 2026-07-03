from typing import Any

try:
    from pyarrow import orc as pyarrow_orc
except ImportError:
    pyarrow_orc = None

from mloda_plugins.feature_group.input_data.pyarrow_read_file import PyArrowReadFile


class OrcReader(PyArrowReadFile):
    """
    Base class for ORC file reading feature groups.

    This feature group enables reading data from Apache ORC (Optimized Row Columnar)
    files, a columnar storage format optimized for Hadoop workloads. Provides
    efficient compression and predicate pushdown capabilities.

    ## Supported Operations

    - `orc_file_loading`: Load data from ORC files with schema preservation
    - `columnar_reading`: Efficiently read only requested columns
    - `column_discovery`: Discover available columns from ORC metadata
    - `compression_handling`: Automatic decompression of ORC files

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features reference column names from ORC files:

    Examples:
    ```python
    features = [
        "event_id",         # Column from ORC file
        "event_type",       # Categorical column
        "event_timestamp"   # Temporal column
    ]
    ```

    ### 2. Configuration-Based Creation

    ```python
    from mloda.user import Feature
    from mloda.user import Options

    feature = Feature(
        name="log_message",
        options=Options(
            context={
                "BaseInputData": (OrcReader, "/path/to/data.orc")
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic ORC Feature Access

    ```python
    from mloda.user import Feature
    from mloda.user import Options

    # Simple column reference from ORC file
    feature = Feature(
        name="page_views",
        options=Options(
            context={
                "BaseInputData": (OrcReader, "analytics.orc")
            }
        )
    )
    ```


    ## Parameter Classification

    ### Context Parameters (Default)
    - `file_path`: Path to the ORC file (absolute or relative)
    - Column schema is preserved from ORC metadata

    ### Group Parameters
    Currently none for OrcReader.

    ## Requirements

    - ORC file must exist at the specified path
    - Feature names must match column names in the ORC schema
    - All features in a FeatureSet must use the same ORC file
    - PyArrow library must be installed

    ## Additional Notes

    - Uses PyArrow's ORC reader for efficient data access
    - Supports .orc and .ORC file extensions
    - Columnar format allows reading only needed columns
    - Commonly used in big data and Hadoop ecosystems
    - Provides excellent compression ratios for large datasets
    """

    _file_format_label = "ORC"

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (
            ".orc",
            ".ORC",
        )

    @classmethod
    def _pyarrow_module(cls) -> Any:
        return pyarrow_orc

    @classmethod
    def produce_table(cls, data_access: Any, column_names: list[str]) -> Any:
        return pyarrow_orc.read_table(source=data_access, columns=column_names)
