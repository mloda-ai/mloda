import os
from typing import Any, List
from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    ToolFunctionDeclaration,
)


class ListFilesTool(BaseTool):
    """
    ListFilesTool is a BaseTool that provides functionality to list files in a specified folder.
    It can optionally filter by file extension or pattern if you choose to implement that logic.
    """

    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name=cls.get_class_name(),
            description="""Lists all files in a specified folder or directory. 
                           Provide a folder path to retrieve the list of files.""",
            parameters=[
                InputDataObject(
                    name="folder_path",
                    type="str",
                    description="The path of the folder from which to list files. (e.g., 'my_directory/')",
                ),
            ],
            required=["folder_path"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        folder_path = kwargs.get("folder_path")

        if folder_path is None:
            raise ValueError("The 'folder_path' parameter is required.")

        return cls.list_files_in_folder(folder_path)

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        """Creates a result string showing the files that were found."""
        folder_path = kwargs.get("folder_path")
        if folder_path is None:
            raise ValueError("The 'folder_path' parameter is required for result string creation.")

        result_string = "TOOL: ListFilesTool\n"
        result_string += f"Files in {folder_path}:\n"
        result_string += "--------------------------------------------------\n"
        result_string += result
        result_string += "\n--------------------------------------------------\n"
        result_string += "END OF TOOL: ListFilesTool\n"
        return result_string

    @staticmethod
    def list_files_in_folder(folder_path: str) -> str:
        """
        List files in the given folder path and return them as a string.
        """

        if not os.path.exists(folder_path):
            return f"Directory does not exist: {folder_path}"

        if not os.path.isdir(folder_path):
            return f"Path is not a directory: {folder_path}"

        try:
            all_items = os.listdir(folder_path)
            files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

            if not files:
                return "No files found in the specified directory."

            # Return the list of files as a newline-separated string
            return "\n".join(files)

        except Exception as e:
            return f"An error occurred while listing files: {e}"
