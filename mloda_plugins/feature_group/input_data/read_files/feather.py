from typing import Any, Tuple

from pyarrow import feather as pyarrow_feather

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class FeatherReader(ReadFile):
    """
    Base class for Feather file reading feature groups.

    This feature group enables reading data from Apache Arrow Feather files, a
    lightweight columnar format designed for fast data transfer and storage.
    Provides extremely fast read/write performance for data interchange.

    ## Supported Operations

    - `feather_file_loading`: Load data from Feather files with Arrow schema
    - `columnar_reading`: Efficiently read only requested columns
    - `fast_io`: Optimized for rapid data loading and minimal overhead
    - `schema_preservation`: Maintains Arrow data types and metadata

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features reference column names from Feather files:

    Examples:
    ```python
    features = [
        "sensor_id",        # Column from Feather file
        "reading_value",    # Numeric measurement
        "timestamp"         # Temporal data
    ]
    ```

    ### 2. Configuration-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="measurement",
        options=Options(
            context={
                "BaseInputData": (FeatherReader, "/path/to/data.feather")
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic Feather Feature Access

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    # Simple column reference from Feather file
    feature = Feature(
        name="metric_value",
        options=Options(
            context={
                "BaseInputData": (FeatherReader, "metrics.feather")
            }
        )
    )
    ```


    ## Parameter Classification

    ### Context Parameters (Default)
    - `file_path`: Path to the Feather file (absolute or relative)
    - Column schema is preserved from Feather format

    ### Group Parameters
    Currently none for FeatherReader.

    ## Requirements

    - Feather file must exist at the specified path
    - Feature names must match column names in the Feather file
    - All features in a FeatureSet must use the same Feather file
    - PyArrow library must be installed

    ## Additional Notes

    - Uses PyArrow's Feather reader for maximum speed
    - Supports .feather extension
    - Designed for fast read/write operations
    - Ideal for temporary data storage and data exchange
    - Columnar format enables selective column reading
    """

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".feather",
            ".feather",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pyarrow_feather.read_table(source=data_access, columns=list(features.get_all_names()))
