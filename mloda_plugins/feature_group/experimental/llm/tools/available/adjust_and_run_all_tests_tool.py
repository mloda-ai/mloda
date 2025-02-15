import os
import subprocess  # nosec
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    ToolFunctionDeclaration,
    InputDataObject,
)


class AdjustAndRunAllTestsTool(BaseTool):
    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name="AdjustAndRunAllTestsTool",
            description="AdjustAndRunAllTestsTool replaces content in an existing file and then runs all tests.",
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

        new_file_content = content.replace(old_content, new_content)

        with open(file_path, "w") as f:
            f.write(new_file_content)

        tox_result = cls.run_tox()

        return f"Successfully replaced all occurrences of '{old_content}' with '{new_content}' in file '{file_path}'.\nTox result:\n{tox_result}"

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        return result

    @staticmethod
    def run_tox() -> str:
        command = ["tox"]
        process = subprocess.run(command, capture_output=True, text=True)  # nosec
        if process.returncode == 0:
            return f"Tox run successful.\nStdout:\n{process.stdout}"
        else:
            return f"Tox run failed with return code {process.returncode}.\nStdout:\n{process.stdout}\nStderr:\n{process.stderr}"
