import subprocess  # nosec
import sys
from typing import Any, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class InstalledPackagesFeatureGroup(AbstractFeatureGroup):
    """
    Base class for retrieving installed Python packages in the current environment.

    This feature group executes `pip freeze` to capture all installed Python packages
    and their versions. It's particularly useful for LLM-based workflows that need to
    understand the available dependencies, generate environment documentation, or verify
    package compatibility.

    ## Key Capabilities

    - Executes pip freeze to list all installed packages
    - Returns package list as a single string with newline-separated entries
    - Works in any Python environment (virtualenv, conda, system)
    - No configuration required - works out of the box

    ## Common Use Cases

    - Providing package context to LLMs for code generation
    - Documenting development environments
    - Verifying dependency availability before code execution
    - Generating requirements.txt-style outputs
    - Environment auditing and comparison

    ## Usage Examples

    ### Basic String-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature

    # Create the feature
    feature = Feature(name="InstalledPackagesFeatureGroup")

    # The feature will return all installed packages as a string
    # Output format: "package1==1.0.0\npackage2==2.1.3\n..."
    ```

    ### Configuration-Based Creation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="placeholder",
        options=Options(context={})
    )
    # No context parameters required
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    Currently none. This feature group requires no parameters.

    ### Group Parameters
    Currently none.

    ## Output Format

    Returns a DataFrame with a single column named `InstalledPackagesFeatureGroup`
    containing a string of all installed packages in pip freeze format:

    ```
    package1==1.0.0
    package2==2.1.3
    package3==0.5.0
    ...
    ```

    ## Requirements

    - Python environment with pip installed
    - Subprocess execution permissions
    - Pandas for DataFrame creation

    ## Implementation Details

    - Uses `subprocess.run()` to execute `pip freeze`
    - Captures stdout as text
    - Returns packages as a single string in a list (DataFrame-compatible)
    - On error, returns error message dictionary

    ## Security Considerations

    - Uses `# nosec` markers as subprocess call is controlled and safe
    - No user input is passed to subprocess
    - Command is fixed: `[sys.executable, "-m", "pip", "freeze"]`
    """

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)  # nosec
            packages = result.stdout
            return {cls.get_class_name(): [packages]}
        except subprocess.CalledProcessError as e:
            error_message = f"Command '{e.cmd}' failed with return code {e.returncode}. Error output: {e.stderr}"
            return {"error": error_message}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataFrame}
