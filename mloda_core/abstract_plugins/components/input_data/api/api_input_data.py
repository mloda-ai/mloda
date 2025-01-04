from typing import Optional
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.options import Options


class ApiInputData(BaseInputData):
    """
    This class represents api input data, which was passed through the api.
    """

    def matches(
        self, feature_name: str, options: Options, data_access_collection: Optional[DataAccessCollection] = None
    ) -> bool:
        """
        We match the feature name with the column names given in ApiInputData.
        If we find a match, we return True, otherwise False.

        This function can be overwritten to support more complex matching.
        """

        _data_access_name = self.data_access_name()
        if not _data_access_name:
            raise ValueError(f"Data access name was not set for ApiInputData class {self.__class__.__name__}.")

        api_input_data_column_names = options.get(_data_access_name)
        if api_input_data_column_names is None:
            return False

        for key, value in api_input_data_column_names.items():
            if feature_name in value:
                return True
        return False
