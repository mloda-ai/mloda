import logging
import os
from typing import Any, Optional, Set

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.llm.llm_api.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

logger = logging.getLogger(__name__)


class LLMFileSelector(AbstractFeatureGroup):
    """
    Base class for using LLMs to intelligently select relevant files from directories.

    This feature group combines file reading, LLM analysis, and validation to identify
    files that match a natural language prompt. It reads files from a target directory,
    sends their combined content to an LLM (default: Gemini), and receives file paths
    that are relevant to the given prompt.

    ## Key Capabilities

    - Natural language file selection using LLM reasoning
    - Automatic file discovery and content aggregation
    - Built-in validation of returned file paths
    - Support for file type filtering (e.g., only .py files)
    - Disallowed files list for excluding unwanted files
    - Integrates with ConcatenatedFileContent for efficient file reading

    ## Common Use Cases

    - Finding files related to a specific feature or functionality
    - Identifying test files for a given implementation
    - Discovering documentation files for a topic
    - Locating configuration files for a service
    - Building context for code refactoring tasks

    ## Supported Operations

    The LLM analyzes file contents and returns comma-separated file paths based on:
    - Semantic understanding of file content
    - Natural language prompt matching
    - Code structure and organization patterns
    - File naming conventions and relationships

    ## Feature Creation Methods

    ### 1. String-Based Creation

    ```python
    # Note: This feature group typically requires configuration-based creation
    # due to required parameters (prompt, target_folder)
    ```

    ### 2. Configuration-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="LLMFileSelector",
        options=Options(
            context={
                "prompt": "Find all files related to data validation",
                "target_folder": frozenset(["/path/to/project/src"]),
                "file_type": "py",  # Optional: filter by file extension
                "disallowed_files": frozenset(["__init__.py"]),  # Optional
            }
        )
    )
    ```

    ## Usage Examples

    ### Finding Feature Implementation Files

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="LLMFileSelector",
        options=Options(
            context={
                "prompt": "Find files that implement authentication logic",
                "target_folder": frozenset(["/workspace/src"]),
                "file_type": "py",
            }
        )
    )
    # Returns: "/workspace/src/auth/login.py,/workspace/src/auth/session.py,..."
    ```

    ### Finding Test Files for a Component

    ```python
    feature = Feature(
        name="LLMFileSelector",
        options=Options(
            context={
                "prompt": "Find test files for the database connection module",
                "target_folder": frozenset(["/workspace/tests"]),
                "file_type": "py",
                "disallowed_files": frozenset(["__init__.py", "conftest.py"]),
            }
        )
    )
    ```

    ### Multi-Directory Search

    ```python
    feature = Feature(
        name="LLMFileSelector",
        options=Options(
            context={
                "prompt": "Find all configuration files for the web service",
                "target_folder": frozenset([
                    "/workspace/config",
                    "/workspace/settings"
                ]),
                "file_type": "json",
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)

    These parameters don't affect Feature Group resolution/splitting:

    - `prompt` (required): Natural language description of files to find
    - `target_folder` (required): Frozenset of directory paths to search
    - `file_type` (optional): File extension to filter (e.g., "py", "js", "json")
    - `disallowed_files` (optional): Frozenset of filenames to exclude
      (default: ["__init__.py"])

    ### Group Parameters

    Currently none for LLMFileSelector.

    ## Output Format

    Returns a DataFrame column with comma-separated file paths:
    ```
    "/path/to/file1.py,/path/to/file2.py,/path/to/file3.py"
    ```

    ## Requirements

    - Google Gemini API key set in environment (GEMINI_API_KEY)
    - Target directory must exist and be accessible
    - Files must be readable
    - ConcatenatedFileContent feature group available
    - GeminiRequestLoop feature group available

    ## Validation

    Both input and output validation ensure:
    - All returned file paths exist on the filesystem
    - No newline characters in file paths
    - Files are accessible and readable

    ## Related Feature Groups

    - `ConcatenatedFileContent`: Reads and combines file contents
    - `GeminiRequestLoop`: Handles LLM API communication
    - `ListDirectoryFeatureGroup`: Provides directory structure context
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        prompt = options.get("prompt")
        if not prompt:
            raise ValueError("Prompt is required for LLMFileSelector")

        target_folder = options.get("target_folder")
        if not target_folder:
            raise ValueError("Target folder is required for LLMFileSelector")

        disallowed_files = list(options.get("disallowed_files")) if options.get("disallowed_files") else ["__init__.py"]

        file_type = options.get("file_type")

        llm_feature = Feature(
            name=GeminiRequestLoop.get_class_name(),
            options={
                "model": "gemini-2.0-flash-exp",  # Choose your desired model
                "prompt": prompt,
                DefaultOptionKeys.in_features: frozenset([ConcatenatedFileContent.get_class_name()]),
                "target_folder": frozenset(target_folder),
                "disallowed_files": frozenset(disallowed_files),
                "file_type": file_type,
            },
        )
        return {llm_feature}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        if GeminiRequestLoop.get_class_name() not in data.columns:
            raise ValueError(f"Feature {GeminiRequestLoop.get_class_name()} not found in input data.")

        for str_paths in data[GeminiRequestLoop.get_class_name()].values:
            paths = str_paths.split(",")
            for path in paths:
                if "\n" in path:
                    raise ValueError(f"File path {path} contains a newline character.")

                if not os.path.exists(path):
                    raise ValueError(f"File <{path}> does not exist.")

        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data = data.rename(
            columns={GeminiRequestLoop.get_class_name(): LLMFileSelector.get_class_name()}, inplace=False
        )
        return data

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        for str_paths in data[cls.get_class_name()].values:
            paths = str_paths.split(",")
            for path in paths:
                if "\n" in path:
                    raise ValueError(f"File path {path} contains a newline character.")

                if not os.path.exists(path):
                    raise ValueError(f"File <{path}> does not exist.")
        return True
