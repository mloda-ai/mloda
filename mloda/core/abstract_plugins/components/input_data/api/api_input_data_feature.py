from typing import Any

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class ApiInputDataFeature(FeatureGroup):
    """
    Pass-through feature group for in-memory data supplied at runtime.

    Returns columns provided via the ``api_data`` parameter of ``mloda.run_all(...)``
    or ``session.run(...)`` unchanged as requested features.

    Example:
        ```python
        from mloda.user import PluginLoader, mloda

        PluginLoader().all()
        result = mloda.run_all(
            ["query"],
            api_data={"UserQuery": {"query": ["hello"]}},
            compute_frameworks=["PythonDictFramework"],
        )
        ```
    """

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return ApiInputData()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data
