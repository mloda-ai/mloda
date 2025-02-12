import os
from typing import List

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
from tests.test_plugins.feature_group.experimental.llm.test_llm import format_array


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

    def test_multiply(self) -> None:
        prompt = """ As first step: Multiply 5 by 5. As a second step, multiply the result by 2. """

        tool_collection = ToolCollection()
        tool_collection.add_tool(MultiplyTool.get_class_name())

        target_folder = [
            os.getcwd() + "/tests/test_core/test_index",
        ]

        features = self.get_features(prompt, tool_collection, target_folder)

        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        for res in results:
            assert "50" in res[GeminiRequestLoop.get_class_name()].values[0]

    def test_run_single_pytest(self) -> None:
        prompt = """ Run a unit test provided in the context by selecting a test by giving the name following the syntax: python3 -m pytest -k test_name -s. If you receive a 0 as return code, stop."""

        tool_collection = ToolCollection()
        tool_collection.add_tool(RunSinglePytestTool.get_class_name())

        target_folder = [
            os.getcwd() + "/tests/test_core/test_index",
        ]

        features = self.get_features(prompt, tool_collection, target_folder)

        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        for res in results:
            assert "test_add_index_simple" in res[GeminiRequestLoop.get_class_name()].values[0]

    def test_run_tox(self) -> None:
        prompt = """ Run tox exactly one time. Propose a solution for the error if the return code is not 0. """

        tool_collection = ToolCollection()
        tool_collection.add_tool(RunToxTool.get_class_name())

        target_folder = [
            os.getcwd() + "/mloda_plugins",
        ]

        features = self.get_features(prompt, tool_collection, target_folder)

        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        # for i, res in enumerate(results):
        #    formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
        #    print(formatted_output)

        for res in results:
            assert "tox" in res[GeminiRequestLoop.get_class_name()].values[0]

    def test_create_new_plugin(self) -> None:
        prompt = """ Create a new plugin for a postgres reader and create tests for it.  """

        tool_collection = ToolCollection()
        tool_collection.add_tool(CreateFileTool.get_class_name())

        target_folder = [
            os.getcwd() + "/mloda_plugins",
        ]

        features = self.get_features(prompt, tool_collection, target_folder)

        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        for i, res in enumerate(results):
            formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
            print(formatted_output)

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

        # prompt = """ Create a new abstractfeaturegroup plugin getting the list and directory of the project. """
        # prompt = """ Can you create a mlodaAPI test fot the plugin ListDirectoryFeatureGroup."""
        # prompt = """you create a mlodaAPI test for the plugin InstalledPackagesFeatureGroup."""
        prompt = """Create a new plugin where an llm is asked to chose from a list of code files, which help the most to answer a question. 
                    You can use the ConcatenatedFileContent feature to get the list of files and GeminiRequestLoop to run the tools.
        """

        tool_collection = ToolCollection()
        tool_collection.add_tool(CreateFileTool.get_class_name())
        tool_collection.add_tool(RunSinglePytestTool.get_class_name())

        target_folder = [
            os.getcwd() + "/mloda_core/api",
            os.getcwd() + "/mloda_plugins",
            os.getcwd() + "/tests/test_plugins/",
        ]

        features = self.get_features(prompt, tool_collection, target_folder)

        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        for i, res in enumerate(results):
            formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
            print(formatted_output)
