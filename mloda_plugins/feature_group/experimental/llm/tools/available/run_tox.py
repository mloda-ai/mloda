import logging
import os
import subprocess  # nosec
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    PytestResult,
    ToolFunctionDeclaration,
)

logger = logging.getLogger(__name__)


class RunToxTool(BaseTool):
    """
    RunToxTool is a BaseTool that provides functionality to run tox.

    Methods:
        tool_declaration(cls) -> ToolFunctionDeclaration:
            Returns the tool declaration for the RunToxTool, including its name, description, and parameters.

        execute(cls, **kwargs: Any) -> PytestResult:
            Runs tox and returns a PytestResult object.

        create_result_string(cls, result: PytestResult, **kwargs: Any) -> str:
            Creates a result string that summarizes the tox run, including stdout, stderr, and return code.

        run_tox() -> PytestResult:
            Runs tox.
    """

    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name=cls.get_class_name(),
            description="""This RunToxTool runs the 'tox' command to execute tests within isolated environments.  The RunToxTool is designed to run *exactly one* execution of 'tox', and then *always* stop. If the 'tox' command returns a return code of `0`, indicating success, the tool should report success. If the return code is not `0`, indicating failure, the tool should report failure. In *all* cases, the tool must stop after this single execution.""",
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
    def execute(cls, **kwargs: Any) -> PytestResult:  # type: ignore
        return cls.run_tox()

    @classmethod
    def create_result_string(cls, result: str | PytestResult, **kwargs: Any) -> str:
        """Creates a result string summarizing the tox run."""

        if isinstance(result, str):
            raise ValueError(f"Expected PytestResult, got string: {result}")
        result_string = "TOOL: RunToxTool\n"
        result_string += "Tox run:\n"
        result_string += f"  Return Code: {result.return_code}\n"
        result_string += f"  Stdout:\n{result.stdout}\n"
        if result.stderr:
            result_string += f"  Stderr:\n{result.stderr}\n"
        if result.error_message:
            result_string += f"  Error Message:\n{result.error_message}\n"

        result_string += (
            f"\n DO NOT RUN THIS {cls.get_class_name()} TOOL AGAIN. IT IS DESIGNED TO RUN *EXACTLY* ONCE. \n"
        )

        result_string += "END OF TOOL: RunToxTool\n"
        return result_string

    @staticmethod
    def run_tox() -> PytestResult:
        """Runs tox."""

        command = ["tox"]

        custom_env = os.environ.copy()  # Copy existing environment variables
        custom_env["DEACTIVATE_NOTEBOOK_AND_DOC_TESTS"] = (
            "--ignore=tests/test_documentation/ --ignore=tests/test_examples/"  # Set a new environment variable  # Set a new environment variable
        )

        logger.info(command)

        process = subprocess.run(command, capture_output=True, text=True, env=custom_env)  # nosec
        if process.returncode == 0:
            return PytestResult(stdout=process.stdout, return_code=process.returncode)
        else:
            return PytestResult(
                stdout=process.stdout,
                stderr=process.stderr,
                return_code=process.returncode,
                error_message=f"Tox exited with code: {process.returncode}",
            )
