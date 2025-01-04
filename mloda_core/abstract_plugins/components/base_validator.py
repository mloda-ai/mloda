from abc import ABC, abstractmethod
from typing import Any, Optional, Dict


import logging

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """
    BaseValidator is an abstract base class for creating custom validators.
    In case of log level error, the application will raise an exception if the validation fails.
    In other cases, it will log the message.

    This enables:
        -   data creation for debugging purposes
        -   hypothesis testing
        -   cache writes and just recalculating the data of one feature manually

    The default case however should be error.

    Attributes:
        validation_rules (Dict[str, Any]): A dictionary containing the rules for validation.
        log_level (str): The logging level to be used. Defaults to "error".

    Methods:
        validate(data: Any) -> Optional[bool]:
            Abstract method to be implemented by subclasses to validate the given data.

        handle_log_level(_error: str, _exception: Exception) -> None:
            Handles logging based on the specified log level. Raises an exception if the log level is "error".
    """

    def __init__(self, validation_rules: Dict[str, Any], log_level: str = "error") -> None:
        self.validation_rules = validation_rules
        self.log_level = log_level or "error"

    @abstractmethod
    def validate(self, data: Any) -> Optional[bool]:
        pass

    def handle_log_level(self, _error: str, _exception: Exception) -> None:
        if self.log_level == "error":
            raise _exception
        elif self.log_level == "warning":
            logger.warning(_error, exc_info=_exception)
        elif self.log_level == "info":
            logger.info(_error, exc_info=_exception)
        elif self.log_level == "debug":
            logger.debug(_error, exc_info=_exception)
        else:
            raise Exception(f"Invalid log level: {self.log_level} in {self.__class__.__name__}.")
