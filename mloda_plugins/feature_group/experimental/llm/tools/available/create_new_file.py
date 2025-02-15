import os
from typing import Any
from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    ToolFunctionDeclaration,
)


class CreateFileTool(BaseTool):
    """
    CreateFileTool is a BaseTool that provides functionality to create a new file with the provided content.
    """

    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name=cls.get_class_name(),
            description="""This CreateFileTool creates a new file with the provided content. Provide the complete text content for the file, and the desired path to create the file.""",
            parameters=[
                InputDataObject(
                    name="file_content",
                    type="str",
                    description="The complete text content to write to the file.",
                ),
                InputDataObject(
                    name="file_path",
                    type="str",
                    description="The desired path to create the file (e.g., my_file.txt, my_directory/my_file.py).",
                ),
            ],
            required=["file_content", "file_path"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        file_content = kwargs.get("file_content")
        file_path = kwargs.get("file_path")

        if file_content is None:
            raise ValueError("The 'file_content' parameter is required")
        if file_path is None:
            raise ValueError("The 'file_path' parameter is required")

        return cls.create_file_and_return(file_content, file_path)

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        """Creates a result string showing the file was created."""
        file_path = kwargs.get("file_path")
        if file_path is None:
            raise ValueError("The 'file_path' parameter is required for result string creation.")

        result_string = "TOOL: CreateFileTool\n"
        result_string += f"Successfully wrote the following content to {file_path}:\n"
        result_string += "--------------------------------------------------\n"
        result_string += result
        result_string += "\n--------------------------------------------------\n"
        result_string += "END OF TOOL: CreateFileTool\n"
        return result_string

    @staticmethod
    def create_file_and_return(file_content: str, file_path: str) -> str:
        """Write code to file and return the code."""

        while os.path.exists(file_path):
            return f"File already exists at {file_path}. Please provide a different path."

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            return f"Directory does not exist at {directory}. Please provide a different path."

        with open(file_path, "w") as f:
            f.write(file_content)

        return file_content
