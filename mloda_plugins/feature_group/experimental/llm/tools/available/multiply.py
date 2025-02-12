import logging
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    ToolFunctionDeclaration,
)

logger = logging.getLogger(__name__)


class MultiplyTool(BaseTool):
    """
    MultiplyTool is a BaseTool that provides functionality to multiply two numbers together.

    Methods:
        tool_declaration(cls) -> ToolFunctionDeclaration:
            Returns the tool declaration for the MultiplyTool, including its name, description, and parameters.

        execute(cls, **kwargs: Any) -> str:
            Multiplies two numbers 'a' and 'b' provided in kwargs and returns the result as a string.
            Raises ValueError if either 'a' or 'b' is not provided.

        create_result_string(cls, result: str, **kwargs: Any) -> str:
            Creates a result string that describes the multiplication operation performed.
            Raises ValueError if either 'a' or 'b' is not provided.
    """

    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name=cls.get_class_name(),
            description=""" This tool multiplies two float numbers. It takes two arguments, a and b. """,
            parameters=[
                InputDataObject(name="a", type="float", description="The first number to multiply."),
                InputDataObject(name="b", type="float", description="The second number to multiply."),
            ],
            required=["a", "b"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        """returns a * b."""
        a = kwargs.get("a")
        b = kwargs.get("b")
        if a is None or b is None:
            raise ValueError("Both 'a' and 'b' parameters are required")

        return str(a * b)

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        a = kwargs.get("a")
        b = kwargs.get("b")
        if a is None or b is None:
            raise ValueError("Both 'a' and 'b' parameters are required")
        return f"The result of multiplying {a} and {b} together was: {result}"
