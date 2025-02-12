from abc import ABC, abstractmethod
from typing import Any, List
import logging

from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    ToolFunctionDeclaration,
)

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    """

    @classmethod
    @abstractmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        """
        Returns the ToolFunctionDeclaration associated with this tool.
        """

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs: Any) -> str:
        """
        Executes the tool with the given parameters and returns a string.

        Args:
            **kwargs: Keyword arguments representing the tool's input parameters.

        Returns:
            str: A string describing the result of the tool's execution.
        """
        pass

    @classmethod
    @abstractmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        """
        Creates a result string based on the tool's execution result and input parameters.

        Args:
            result (str): The result of the tool's execution.
            **kwargs (Any): Additional keyword arguments representing the tool's input parameters.

        Returns:
            str: A string describing the result of the tool's execution.
        """
        pass

    @classmethod
    def build_tool_declaration(
        cls,
        name: str,
        description: str,
        parameters: List[InputDataObject],
        required: List[str],
    ) -> ToolFunctionDeclaration:
        """
        Utility method to create ToolFunctionDeclaration objects.

        Args:
            name (str): The name of the tool.
            description (str): A brief description of the tool.
            parameters (List[InputDataObject]): A list of InputDataObject instances representing the tool's input parameters.
            required (List[str]): A list of parameter names that are required for the tool.

        Returns:
            ToolFunctionDeclaration: An instance of ToolFunctionDeclaration containing the tool's metadata and execution function.
        """
        return ToolFunctionDeclaration(
            name=name,
            description=description,
            tool_result=cls.create_result_string,
            parameters=parameters,
            required=required,
            function=cls.execute,
        )

    @classmethod
    def get_class_name(cls) -> str:
        """
        Returns the name of the class.

        Returns:
            str: The name of the class.
        """
        return cls.__name__
