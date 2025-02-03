import subprocess  # nosec
import shlex
from typing import Dict
import logging

from mloda_core.abstract_plugins.components.utils import get_all_subclasses
from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import PytestResult, ToolFunctionDeclaration


logger = logging.getLogger(__name__)


class ToolCollection:
    def __init__(self) -> None:
        self.data: Dict[str, ToolFunctionDeclaration] = {}
        subclasses = get_all_subclasses(BaseTool)

        self.tool_mappings = {}

        for sub in subclasses:
            self.tool_mappings[sub.__name__] = sub

    def add_tool(self, tool_name: str) -> None:
        found_mapping = self.tool_mappings.get(tool_name)

        if found_mapping is None:
            raise ValueError(f"Tool {tool_name} not found in tool mappings.")

        self.data[tool_name] = found_mapping.tool_declaration()

    def get_tool(self, tool_name: str) -> ToolFunctionDeclaration:
        return self.data[tool_name]

    def get_all_tools(self) -> Dict[str, ToolFunctionDeclaration]:
        return self.data

    def __str__(self) -> str:
        return str(self.data)

    @staticmethod
    def run_pytest(test_name: str) -> PytestResult:
        """Runs pytest with the specified arguments."""

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

    @staticmethod
    def run_tox() -> PytestResult:
        """Runs tox."""

        command = ["tox"]

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

    @staticmethod
    def create_or_overwrite_new_plugin_and_write_to_file(code: str) -> str:
        """Write code to file and return the file name."""

        with open("tests/test_plugins/new_plugin.py", "w") as f:
            f.write(code)

        return code

    @staticmethod
    def create_or_overwrite_unit_test(code: str) -> str:
        """Write code to file and return the file name."""

        with open("tests/test_plugins/new_plugin_unit_test.py", "w") as f:
            f.write(code)

        return code
