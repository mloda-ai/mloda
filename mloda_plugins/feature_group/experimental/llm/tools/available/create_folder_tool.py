import os
from typing import Any


from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    ToolFunctionDeclaration,
    InputDataObject,
)


class CreateFolderTool(BaseTool):
    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name="CreateFolderTool",
            description="CreateFolderTool creates a folder if it does not exist.",
            parameters=[
                InputDataObject(name="folder_path", type="str", description="The path of the folder to create.")
            ],
            required=["folder_path"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        folder_path = kwargs.get("folder_path")
        if folder_path is None:
            raise ValueError("The 'folder_path' parameter is required")

        abs_folder_path = os.path.abspath(folder_path)
        if not abs_folder_path.startswith(os.getcwd()):
            raise ValueError(f"The folder path must be within the current working directory. {folder_path}")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return f"Folder '{folder_path}' created."
        return f"Folder '{folder_path}' already exists."

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        return result
