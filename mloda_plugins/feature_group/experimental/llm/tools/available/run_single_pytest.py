import logging
import shlex
import subprocess  # nosec
from typing import Any

from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import (
    InputDataObject,
    PytestResult,
    ToolFunctionDeclaration,
)

logger = logging.getLogger(__name__)


class RunSinglePytestTool(BaseTool):
    """
    RunPytestTool is a BaseTool that provides functionality to run pytest tests.

    Methods:
        tool_declaration(cls) -> ToolFunctionDeclaration:
            Returns the tool declaration for the RunPytestTool, including its name, description, and parameters.

        execute(cls, **kwargs: Any) -> PytestResult:
            Runs pytest with the specified test name provided in kwargs and returns a PytestResult object.
            Raises ValueError if 'test_name' is not provided.

        create_result_string(cls, result: PytestResult, **kwargs: Any) -> str:
            Creates a result string that summarizes the pytest run, including stdout, stderr, and return code.
            Raises ValueError if 'test_name' is not provided.

        run_pytest(test_name: str) -> PytestResult:
            Runs pytest with the specified arguments.
    """

    @classmethod
    def tool_declaration(cls) -> ToolFunctionDeclaration:
        return cls.build_tool_declaration(
            name=cls.get_class_name(),
            description="""This RunSinglePytestTool executes a single pytest test function using the command: `python3 -m pytest -k test_name -s`.  The `test_name` *must* be the exact name of the test function. For example, to run `test_run_single_pytest`, use `-k test_run_single_pytest`.  Do not include class names or module paths. The goal is to run *only one* test and then immediately stop. If the pytest command returns with a return code of `0`, indicating a successful test run, the tool should *stop* and report success.""",
            parameters=[
                InputDataObject(
                    name="test_name",
                    type="str",
                    description="The name of the test to run.",
                ),
            ],
            required=["test_name"],
        )

    @classmethod
    def execute(cls, **kwargs: Any) -> PytestResult:  # type: ignore
        test_name = kwargs.get("test_name")
        if test_name is None:
            raise ValueError("The 'test_name' parameter is required")

        return cls.run_pytest(test_name)

    @classmethod
    def create_result_string(cls, result: str | PytestResult, **kwargs: Any) -> str:
        """Creates a result string summarizing the pytest run."""
        test_name = kwargs.get("test_name")
        if test_name is None:
            raise ValueError("The 'test_name' parameter is required")

        if isinstance(result, str):
            raise ValueError(f"Expected PytestResult, got string: {result}")

        result_string = f"Pytest run for '{test_name}':\n"
        result_string += f"  Return Code: {result.return_code}\n"
        result_string += f"  Stdout:\n{result.stdout}\n"
        if result.stderr:
            result_string += f"  Stderr:\n{result.stderr}\n"
        if result.error_message:
            result_string += f"  Error Message:\n{result.error_message}\n"
        return result_string

    @staticmethod
    def run_pytest(test_name: str) -> PytestResult:
        """Runs pytest with one single test of your chosing from the context."""

        escaped_test_name = shlex.quote(test_name)
        command = ["python3", "-m", "pytest", "-k", escaped_test_name, "-s"]

        logger.info(command)

        process = subprocess.run(command, capture_output=True, text=True)  # nosec

        if process.returncode == 0:
            return PytestResult(stdout=process.stdout, return_code=process.returncode)
        else:
            return PytestResult(
                stdout=process.stdout,
                stderr=process.stderr,
                return_code=process.returncode,
                error_message=f"Pytest exited with code: {process.returncode}",
            )
