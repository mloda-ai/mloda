import os
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    ToolFunctionDeclaration,
    InputDataObject,
)


class AdjustFileTool(BaseTool):
    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name="AdjustFileTool",
            description="AdjustFileTool replaces content in an existing file.",
            parameters=[
                InputDataObject(name="file_path", type="str", description="The full path of the file to modify."),
                InputDataObject(name="old_content", type="str", description="The exact text to replace."),
                InputDataObject(
                    name="new_content", type="str", description="The text to replace the old content with."
                ),
            ],
            required=["file_path", "old_content", "new_content"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path")
        old_content = kwargs.get("old_content")
        new_content = kwargs.get("new_content")

        if file_path is None or old_content is None or new_content is None:
            raise ValueError("Both 'file_path', 'old_content', and 'new_content' parameters are required")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist")

        with open(file_path, "r") as f:
            content = f.read()

        _saved_content = content

        new_file_content = content.replace(old_content, new_content)

        with open(file_path, "w") as f:
            f.write(new_file_content)

        if _saved_content == new_file_content:
            return f"File '{file_path}' was not modified. No occurrences of '{old_content}' were found. NO CHANGES WERE MADE."

        return f"Successfully replaced all occurrences of '{old_content}' with '{new_content}' in file '{file_path}'."

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        return result
