from typing import TYPE_CHECKING, Any

from mloda.core.optional_dependency import require
from mloda.provider import FeatureSet
from mloda.user import DataType
from mloda_plugins.feature_group.input_data.read_file import ReadFile

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework


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
    from mloda.user import Feature
    from mloda.user import Options

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
    from mloda.user import Feature
    from mloda.user import Options

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
    def suffix(cls) -> tuple[str, ...]:
        return (
            ".feather",
            ".feather",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        pyarrow_ipc = require("pyarrow.ipc", "reading Feather files")
        columns = list(features.get_all_names())
        # Feather V2 is the Arrow IPC file format; use ipc.open_file instead of the
        # deprecated pyarrow.feather.read_table (removed warning as of pyarrow 24).
        with pyarrow_ipc.open_file(data_access) as reader:
            return reader.read_all().select(columns)

    @classmethod
    def get_column_names(cls, file_name: str) -> list[str]:
        return list(cls.describe_columns(file_name))

    @classmethod
    def describe_columns(cls, data_access: Any) -> dict[str, DataType | None]:
        pyarrow_ipc = require("pyarrow.ipc", "reading Feather files")
        file_name = cls._file_path(data_access)
        with pyarrow_ipc.open_file(file_name) as reader:
            return {field.name: DataType.from_arrow_type_safe(field.type) for field in reader.schema}

    @classmethod
    def count_rows(cls, data_access: Any, compute_framework: "type[ComputeFramework]") -> int | None:
        if cls._is_overridden(FeatherReader, "load_data"):
            return None
        pyarrow_dataset = require("pyarrow.dataset", "counting Feather rows")
        pyarrow_fs = require("pyarrow.fs", "counting Feather rows")
        file_name = cls._file_path(data_access)
        # One local file, not a dataset: a directory or missing path raises OSError, no URI fetch.
        fragment = pyarrow_dataset.IpcFileFormat().make_fragment(file_name, filesystem=pyarrow_fs.LocalFileSystem())
        count: int = fragment.count_rows()
        return count
