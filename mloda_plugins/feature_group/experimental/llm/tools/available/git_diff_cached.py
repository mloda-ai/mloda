import subprocess  # nosec
from typing import Any
from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    ToolFunctionDeclaration,
)


class GitDiffCachedTool(BaseTool):
    """
    GitDiffCachedTool is a BaseTool that provides functionality to run `git diff --cached` and return its output.
    """

    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name=cls.get_class_name(),
            description="Executes 'git diff --cached' and returns the raw diff output, showing the staged changes in the repository.",
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
        result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)  # nosec
        return result.stdout

    @classmethod
    def create_result_string(cls, result: str, **kwargs: Any) -> str:
        return f"""
        <git_diff_cached>
        <instruction>This is the LATEST git diff --cached. Do NOT call git_diff_cached again to see the changes.</instruction>
        {result}
        </git_diff_cached>
        """
