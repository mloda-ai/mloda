import subprocess  # nosec
import sys
from typing import Any, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe


class InstalledPackagesFeatureGroup(AbstractFeatureGroup):
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
        return {PandasDataframe}
