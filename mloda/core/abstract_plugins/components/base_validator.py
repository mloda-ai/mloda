from abc import ABC, abstractmethod
from typing import Any

from mloda.core.abstract_plugins.components.utils import contained_raise_reason

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
        validate(data: Any) -> None:
            Abstract method to be implemented by subclasses to validate the given data.

        handle_log_level(_error: str, _exception: Exception) -> None:
            Handles logging based on the specified log level. Raises an exception if the log level is "error".
    """

    def __init__(self, validation_rules: dict[str, Any], log_level: str = "error") -> None:
        self.validation_rules = validation_rules
        self.log_level = log_level or "error"

    @abstractmethod
    def validate(self, data: Any) -> None:
        """
        Validate the given data against the validation rules.

        Subclasses must implement this method with their specific validation logic.
        Raise a ValueError with a descriptive message if validation fails.
        If the method returns without raising, validation is considered passed.

        Args:
            data: The data to validate.
        """
        pass

    def handle_log_level(self, _error: str, _exception: Exception) -> None:
        if self.log_level == "error":
            raise _exception
        # The non-error levels contain the raise, so the diagnosis goes on the record as text: exc_info= pins
        # the (type, exception, traceback) triple, and a retained record then keeps the raising frames alive.
        reason = contained_raise_reason(_exception)
        if self.log_level == "warning":
            logger.warning("%s (%s)", _error, reason)
        elif self.log_level == "info":
            logger.info("%s (%s)", _error, reason)
        elif self.log_level == "debug":
            logger.debug("%s (%s)", _error, reason)
        else:
            raise Exception(f"Invalid log level: {self.log_level} in {self.__class__.__name__}.")
