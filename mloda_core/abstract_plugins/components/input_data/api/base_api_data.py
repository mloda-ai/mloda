from abc import ABC, abstractmethod
from typing import Any, List, Optional, final
from mloda_core.abstract_plugins.components.options import Options


class BaseApiData(ABC):
    """
    Base class for representing API input data in the mloda system.

    This abstract class defines the interface and common functionality for API data used throughout the system.
    Subclasses should implement the `column_names` method to specify the columns associated with the API data.

    get_data_by_using_api_data can be overwritten if you need to use other means of data transportation.
    This may be the case if the data cannot be pickeled and you want to use multi-processing.
    Or if you want to get data from a stream.

    The __init__ can also be overwritten if you need to pass additional information to the API input data.

    Attributes:
        api_input_name (str): The name identifier for the API input.
        feature_name (Optional[str]): The name of the feature associated with the API input.
        options (Optional[Options]): Additional options or configurations for the API input data.
    """

    def __init__(self, api_input_name: str, feature_name: Optional[str], options: Optional[Options]) -> None:
        self.api_input_name = api_input_name

    def get_data_by_using_api_data(self, api_data: Any) -> Any:
        """Functionality to connect to other means."""
        return api_data

    @classmethod
    @abstractmethod
    def column_names(cls) -> List[str]:
        """Abstract method to retrieve the list of column names associated with the API data."""
        raise NotImplementedError

    @final
    def get_api_input_name(self) -> str:
        return self.api_input_name
