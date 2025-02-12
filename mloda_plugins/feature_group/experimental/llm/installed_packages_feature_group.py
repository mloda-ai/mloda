import subprocess  # nosec
import sys
from typing import Any

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

from mloda_core.abstract_plugins.components.feature_set import FeatureSet


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
