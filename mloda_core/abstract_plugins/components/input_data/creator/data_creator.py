from typing import Optional, Set
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.options import Options


class DataCreator(BaseInputData):
    """
    This class represents base input data, which is created in place.
    This is useful if you want to create data, which is not dependent on any other dependency.

    Usage:
        - test data
        - dummy data
        - parameter data
    """

    def __init__(self, supports_features: Set[str]) -> None:
        self.feature_names = supports_features

    def matches(
        self, feature_name: str, options: Options, data_access_collection: Optional[DataAccessCollection] = None
    ) -> bool:
        """
        This function can be overwritten to support more complex matching.

        A simple example would be to use the option of a feature, in which you add the data_access_name "DataCreator".
        """
        if feature_name in self.feature_names:
            return True
        return False
