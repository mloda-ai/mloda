from copy import deepcopy
from typing import Any, Tuple

from mloda_plugins.feature_group.input_data.read_file import ReadFile
from pyarrow import fs as pyarrow_fs

from mloda_core.abstract_plugins.components.feature_set import FeatureSet

try:
    import pandas as pd
except ImportError:
    pd = None


class TextFileReader(ReadFile):
    """
    Base class for text file reading feature groups.

    This feature group enables reading entire text files as single feature values,
    useful for loading unstructured text data, documents, or source code files.
    Optionally includes file path metadata in the content.

    ## Supported Operations

    - `text_file_loading`: Load entire text file contents as a single string
    - `utf8_decoding`: Automatic UTF-8 decoding of file contents
    - `path_injection`: Optional inclusion of file path in content
    - `content_feature`: Entire file content as a single feature value

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features typically use the class name as the feature identifier:

    Examples:
    ```python
    features = [
        "TextFileReader",   # Feature containing entire text file content
    ]
    ```

    ### 2. Configuration-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="TextFileReader",
        options=Options(
            context={
                "BaseInputData": (TextFileReader, "/path/to/document.text"),
                "TextFileReader": "/path/to/document.text"
            }
        )
    )
    ```

    ## Usage Examples

    ### Basic Text File Loading

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    # Load entire text file as a feature
    feature = Feature(
        name="TextFileReader",
        options=Options(
            context={
                "TextFileReader": "document.text"
            }
        )
    )
    ```

    ### With File Path Metadata

    ```python
    # Include file path in the content
    feature = Feature(
        name="TextFileReader",
        options=Options(
            context={
                "TextFileReader": "readme.text",
                "AddPathToContent": True
            }
        )
    )
    # Result: "The file path of the following file is readme.text : [content]"
    ```

    ### Loading Source Code Files

    ```python
    # Use PyFileReader for Python files
    feature = Feature(
        name="PyFileReader",
        options=Options(
            context={
                "PyFileReader": "script.py"
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    - `TextFileReader`: File path to the text file (absolute or relative)
    - `AddPathToContent`: Boolean flag to prepend file path to content (optional)

    ### Group Parameters
    Currently none for TextFileReader.

    ## Requirements

    - Text file must exist at the specified path
    - File must be UTF-8 encoded
    - Pandas library must be installed
    - PyArrow library must be installed

    ## Additional Notes

    - Returns a Pandas DataFrame with a single column containing file content
    - Column name is the class name (TextFileReader or PyFileReader)
    - Entire file is loaded into memory as a single string
    - PyFileReader is a specialized variant for .py files
    - Useful for LLM-based feature processing or document analysis
    """

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".text",)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        if pd is None:
            raise ImportError("pandas is not installed.")

        file_path = deepcopy(features.get_options_key(cls.__name__))
        local_fs = pyarrow_fs.LocalFileSystem()

        with local_fs.open_input_file(file_path) as file:
            content = file.read().decode("utf-8")

        if features.get_options_key("AddPathToContent"):
            file_path = file_path.strip("\n")
            content = f"The file path of the following file is {file_path} : {content}"

        data = {cls.get_class_name(): [content]}

        return pd.DataFrame(data)


class PyFileReader(TextFileReader):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".py",)
