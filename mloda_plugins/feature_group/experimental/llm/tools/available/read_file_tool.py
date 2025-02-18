import os
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    ToolFunctionDeclaration,
    InputDataObject,
)


class ReadFileTool(BaseTool):
    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name="ReadFileTool",
            description="ReadFileTool reads a file.",
            parameters=[
                InputDataObject(name="file_path", type="str", description="The exact file path to be read."),
            ],
            required=["file_path"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path")

        if file_path is None:
            raise ValueError("The 'file_path' parameter is required")

        file_path = cls.validate_file_path(file_path)
        if "Error: " in file_path:
            if isinstance(file_path, str):
                return file_path
            raise ValueError(file_path)

        if not os.path.exists(file_path):
            return f"The file '{file_path}' does not exist."

        with open(file_path, "r") as f:
            content = f.read()

        return f"""Successfully read file {file_path}. Content: 
        '''
        {content}
        '''
        """

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        return result

    @classmethod
    def validate_file_path(cls, file_path: str) -> str:
        cwd = os.getcwd()

        if not os.path.isabs(file_path):  # Check if path is relative
            abs_path = os.path.abspath(file_path)  # Convert to absolute path
            if not abs_path.startswith(cwd):  # Ensure it's within the cwd
                return f"Error: The relative file path '{file_path}' must be inside the current working directory ('{cwd}')."
        else:  # If absolute path
            if not file_path.startswith(cwd):  # Ensure cwd is part of the absolute path
                return (
                    f"Error: The absolute file path '{file_path}' must contain the current working directory ('{cwd}')."
                )

        return file_path
