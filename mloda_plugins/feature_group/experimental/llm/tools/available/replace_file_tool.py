import os
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    ToolFunctionDeclaration,
    InputDataObject,
)


class ReplaceFileTool(BaseTool):
    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name="ReplaceFileTool",
            description="ReplaceFileTool replaces the entire content of an existing file.",
            parameters=[
                InputDataObject(name="file_path", type="str", description="The full path of the file to replace."),
                InputDataObject(name="new_content", type="str", description="The new content to write to the file."),
            ],
            required=["file_path", "new_content"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path")
        new_content = kwargs.get("new_content")

        if file_path is None or new_content is None:
            raise ValueError("Both 'file_path' and 'new_content' parameters are required")

        if not os.path.exists(file_path):
            return f"The file '{file_path}' does not exist."

        with open(file_path, "w") as f:
            f.write(new_content)

        return f"Successfully replaced the content of '{file_path}' with the provided new content."

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        return result
