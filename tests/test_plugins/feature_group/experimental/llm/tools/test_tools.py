import os
from typing import List
import logging

import pytest

from mloda_plugins.feature_group.experimental.llm.tools.available.create_new_plugin import (
    CreateFileTool,
)
from mloda_plugins.feature_group.experimental.llm.tools.available.multiply import MultiplyTool
from mloda_plugins.feature_group.experimental.llm.tools.available.run_single_pytest import RunSinglePytestTool
from mloda_plugins.feature_group.experimental.llm.tools.available.run_tox import RunToxTool
from mloda_plugins.feature_group.experimental.llm.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.llm.tools.available.git_diff import GitDiffTool
from mloda_plugins.feature_group.experimental.llm.tools.available.git_diff_cached import GitDiffCachedTool
from mloda_plugins.feature_group.experimental.llm.tools.available.create_folder_tool import CreateFolderTool

logger = logging.getLogger(__name__)


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
class TestSingleTools:
    def get_features(
        self, prompt: str, tool_collection: ToolCollection, target_folder: List[str] = []
    ) -> List[Feature | str]:
        if target_folder == []:
            target_folder = [
                os.getcwd() + "/mloda_plugins",
                os.getcwd() + "/tests/test_plugins/feature_group/experimental/",
            ]

        return [
            Feature(
                name=GeminiRequestLoop.get_class_name(),
                options={
                    "model": "gemini-2.0-flash-exp",
                    "prompt": prompt,
                    "tools": tool_collection,
                    DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
                    "target_folder": frozenset(target_folder),
                    "disallowed_files": frozenset(
                        [
                            "__init__.py",
                            "gemini.py",
                            "llm_base_request.py",
                        ]
                    ),
                    "file_type": "py",
                },
            )
        ]

    def run_test(
        self, prompt: str, tool_classes: str | List[str], target_folder: List[str], expected_output: str
    ) -> None:
        tool_collection = ToolCollection()

        if isinstance(tool_classes, str):
            tool_collection.add_tool(tool_classes)
        else:
            for tool_class in tool_classes:
                tool_collection.add_tool(tool_class)

        features = self.get_features(prompt, tool_collection, target_folder)

        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        for res in results:
            print(res)
            assert expected_output in res[GeminiRequestLoop.get_class_name()].values[0]

    def test_multiply(self) -> None:
        prompt = """ As first step: Multiply 5 by 5. As a second step, multiply the result by 2. """
        target_folder = [
            os.getcwd() + "/tests/test_core/test_index",
        ]
        self.run_test(prompt, MultiplyTool.get_class_name(), target_folder, "50")

    def test_run_single_pytest(self) -> None:
        prompt = """ Run a unit test provided in the context by selecting a test by giving the name following the syntax: python3 -m pytest -k test_name -s. If you receive a 0 as return code, stop."""
        target_folder = [
            os.getcwd() + "/tests/test_core/test_index",
        ]
        self.run_test(prompt, RunSinglePytestTool.get_class_name(), target_folder, "test_add_index_simple")

    def test_run_tox(self) -> None:
        prompt = """ Run tox exactly one time. """
        target_folder = [
            os.getcwd() + "/mloda_plugins",
        ]
        self.run_test(prompt, RunToxTool.get_class_name(), target_folder, "tox")

    def test_create_new_plugin(self) -> None:
        prompt = """ Create a new plugin for a postgres reader and create tests for it.  """
        target_folder = [
            os.getcwd() + "/mloda_plugins",
        ]
        self.run_test(prompt, CreateFileTool.get_class_name(), target_folder, "Result 0 values:")

    def test_create_and_test_new_plugin(self) -> None:
        prompt = """  You are an AI agent tasked with creating a new feature group plugin for reading data from a PostgreSQL database, creating tests for this plugin, and running those tests to ensure the plugin functions correctly.  Your goal is to ensure that you do not run tools twice and handle all errors correctly.

            Follow these steps in order:

            1.  **Create the Plugin:**
                *   Use the 'CreateFileTool' to create the plugin file. 

            2.  **Create Tests for the Plugin:**
                *   Use the 'CreateFileTool' to create a new test file, which tests the created Plugin.

            3.  **Run the Test:**
                *   Identify a single created test of the created plugin and run it using the 'RunSinglePytestTool'.

            4.  **Validation:**
                * After all the actions check one more time.

            5.  **Report success**:
                * After all of it let me know it has been done.
        """
        target_folder = [
            os.getcwd() + "/mloda_core/api",
            os.getcwd() + "/mloda_plugins",
            os.getcwd() + "/tests/test_plugins/",
        ]
        self.run_test(
            prompt,
            [CreateFileTool.get_class_name(), RunSinglePytestTool.get_class_name()],
            target_folder,
            "",
        )

    def test_git_diff_no_cache(self) -> None:
        prompt = """ Run git diff. Return the output. """
        target_folder = [
            os.getcwd() + "/mloda_plugins/function_extender",
        ]
        self.run_test(prompt, GitDiffTool.get_class_name(), target_folder, "diff")

    def test_git_diff_cached(self) -> None:
        prompt = """ Run git diff --cached. Return the output. """
        target_folder = [
            os.getcwd() + "/mloda_plugins/function_extender",
        ]
        self.run_test(prompt, GitDiffCachedTool.get_class_name(), target_folder, "git_diff_cached")

    def test_create_folder(self) -> None:
        prompt = """ Create a folder named 'test_folder' in the 'tests/test_plugins/feature_group/experimental/llm/tools/' directory. """
        target_folder = [
            os.getcwd() + "/tests/test_plugins/feature_group/experimental/llm/tools/",
        ]
        try:
            self.run_test(prompt, CreateFolderTool.get_class_name(), target_folder, "folder")
        finally:
            try:
                os.rmdir("tests/test_plugins/feature_group/experimental/llm/tools/test_folder")
            except Exception:  # nosec
                pass
