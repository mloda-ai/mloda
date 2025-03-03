import subprocess  # nosec
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    ToolFunctionDeclaration,
)


class GitDiffTool(BaseTool):
    """
    GitDiffTool is a BaseTool that provides functionality to run `git diff` and return its output.
    """

    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name=cls.get_class_name(),
            description="Executes 'git diff' and returns the raw diff output, showing the latest changes in the repository between your working directory and the last committed version.",
            parameters=[
                InputDataObject(
                    name="Adummyobject",
                    type="str",
                    description="Adummyobject",
                ),
            ],
            required=[],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> str:
        result = subprocess.run(["git", "diff"], capture_output=True, text=True)  # nosec
        return result.stdout

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        return f"""
        <git_diff>
        <instruction>This is the LATEST git diff. Do NOT call git_diff again to see the changes.</instruction>
        {result}
        </git_diff>
        """
