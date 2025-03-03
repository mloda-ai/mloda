import logging
import os
import re
from typing import Any, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.llm.llm_api.claude import ClaudeRequestLoop
from mloda_plugins.feature_group.experimental.llm.llm_api.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.experimental.llm.llm_file_selector import LLMFileSelector
from mloda_plugins.feature_group.experimental.llm.tools.available.adjust_file_tool import AdjustFileTool
from mloda_plugins.feature_group.experimental.llm.tools.available.create_new_file import CreateFileTool
from mloda_plugins.feature_group.experimental.llm.tools.available.git_diff import GitDiffTool
from mloda_plugins.feature_group.experimental.llm.tools.available.git_diff_cached import GitDiffCachedTool
from mloda_plugins.feature_group.experimental.llm.tools.available.read_file_tool import ReadFileTool
from mloda_plugins.feature_group.experimental.llm.tools.available.replace_file_tool import ReplaceFileTool
from mloda_plugins.feature_group.experimental.llm.tools.available.run_tox import RunToxTool
from mloda_plugins.feature_group.experimental.llm.tools.base_tool import BaseTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import PytestResult
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

logger = logging.getLogger(__name__)


class RunRefactorGeminiRequestLoop(GeminiRequestLoop):
    @classmethod
    def add_final_part_of_prompt(cls) -> str:
        return """"""


class RunRefactorDiffCached:
    def __init__(self) -> None:
        self.compute_frameworks: Set[Type[ComputeFrameWork]] = {PandasDataframe}

    def run(self) -> None:
        # check tests are passing
        self.run_tox_feature_group()

        # get related file to git diff
        get_diff_cached = self.get_tool_output_by_feature_group_(DiffCachedFeatureGroup)
        diff_cache_relevant_files = self.get_relevant_files_for_refactoring(get_diff_cached)
        related_files_to_git_diff = self.get_related_files_to_diff_cached(diff_cache_relevant_files)

        # identify code smells in the actual code
        split_files = self.split_related_files(related_files_to_git_diff)
        single_code_smell = self.identify_one_code_smell(split_files, get_diff_cached)

        # check if tests are passing
        previous = ""
        actual_git_diff = self.get_tool_output_by_feature_group_(DiffFeatureGroup)
        single_code_smell, previous, actual_git_diff
        for i in range(20):
            previous = self.fix_code_smell(i, split_files, single_code_smell, previous, actual_git_diff)
            if "AnalysisComplete" in previous:
                break

    def fix_code_smell(
        self, run_number: int, files: List[str], code_smell: str, previous: str, current_git_diff: str
    ) -> str:
        prompt = f""" 

                **Objective:**

                Your objective is to automatically refactor code to eliminate a specific code smell while ensuring no existing functionality is broken.  Crucially, your refactoring must *not* involve adding or modifying project dependencies.

                **Task Description:**

                Your task is to iteratively refactor the code using provided tools until the code smell is resolved and all tests pass.

                The following table contains a code smell that you must address in the codebase.  Your task is to refactor the code to eliminate this code smell.  The table includes the following columns:
                - Code Smell Description
                - Location (file_name, function_name, line_numbers)
                - Explanation
                - step by step guide in how to fix this code smell.

                {code_smell}

                You have following tools available:

                *   `AdjustFileTool`:  Modify specific parts of an existing file.  Requires specifying the file path and the changes to be made (e.g., line number, content to replace, content to insert).
                *   `CreateFileTool`: Create a new file with the specified content. Requires specifying the file path and the new file content.
                *   `ReplaceFileTool`: Replace the entire content of an existing file. Requires specifying the file path and the new file content. *NOTE: DO NOT RUN THIS TOOL IF THE FILE DOES NOT EXISTS.*
                *   `RunToxTool`: Execute the test suite.  No parameters needed. Returns pass/fail status.
                *   `ReadFileTool`: Read the content of a file. Requires specifying the exact file path.

                **Refactoring Steps**:
                1. Outline in 2-3 sentences what you are planning to do. Take a second and consider the refactoring log. Do not try out the same step as before as shown in the refactoring log.
                2. Identify the needed fix. You can use ReadFileTool to see the files. This means that these files represent the latest state. Always confirm changes by re-reading the file with ReadFileTool if a previous log indicates that changes were made.
                3. Then, after you check the files, apply the correct tools to apply the fix. You can use the AdjustFileTool, CreateFileTool and ReplaceFileTool.
                4. Afterwards, you must create a refactoring log with the following format.
                **Refactoring Log**:
                - **Summary**: Describe your changes in 100 words. If you used tools, specify the used tool and the parameters precisely.
                - **Reason**: Why you made them in less than 20 words.
                - **Next**: State the next action in a precise way in 50 words. If the next step should be a tool use, specify the tool and the parameters.
                5. If you think that all tests should work, you can use RunToxTool to test the whole application.
                6. If the smell is resolved, respond with AnalysisComplete and no further refactoring log is needed. AnalysisComplete ends the entire process.

                The refactoring log must follow this structure:
                **Refactoring Log**:
                Summary: <Max 100 words>  
                Reason: <Max 20 words>  
                Next: <Max 20 words>

                **Start context refactoring log**:
                {previous}
                **End context refactoring log**:
  
                """  # nosec

        tool_collection = ToolCollection()
        tool_collection.add_tool(AdjustFileTool.get_class_name())
        tool_collection.add_tool(RunToxTool.get_class_name())
        tool_collection.add_tool(ReplaceFileTool.get_class_name())
        tool_collection.add_tool(ReadFileTool.get_class_name())
        tool_collection.add_tool(CreateFileTool.get_class_name())

        expensive_model = ClaudeRequestLoop.get_class_name()
        expensive_model = RunRefactorGeminiRequestLoop.get_class_name()

        model = "claude-3-haiku-20240307"
        model = "gemini-2.0-flash-exp"

        feature = Feature(
            name=expensive_model,
            options={
                "model": model,
                "prompt": prompt,
                DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
                "file_paths": frozenset(files),
                "project_meta_data": True,
                "tools": tool_collection,
            },
        )

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks=self.compute_frameworks,
        )
        res = results[0][expensive_model].values[0]

        res = f"""\n This was iterative {run_number}:{res}\n"""

        res = previous + res

        if isinstance(res, str):
            print(res)
            return res
        raise ValueError("Wrong type of result")

    def split_related_files(self, related_files_to_git_diff: str) -> List[str]:
        files = related_files_to_git_diff.split(",")
        new_files = []
        for file in files:
            new_files.append(file.strip("\n"))
        return new_files

    def identify_one_code_smell(self, files: List[str], git_diff_cached: str) -> str:
        """
        - Potential performance bottlenecks (e.g., inefficient algorithms, unnecessary loops)
            - Dead code (unused variables, functions, or code paths)
            - General code smells (e.g., long parameter lists, excessive class coupling)
            - Functions with high cyclomatic complexity

        """

        prompt = f""" 
            You are an experienced software engineer specializing in code refactoring and quality analysis. Your goal is to identify a *specific, actionable* refactoring target introduced or exacerbated by the code changes described in the following `git diff --cached`.

            Given the following `git diff --cached` output:
            
            {git_diff_cached}
            
            DO NOT INCLUDE code smells related to options.
            
            Identify one code smell that is newly introduced or significantly worsened by the changes in the `git diff --cached`.  Focus on problems that are clearly fixable and would provide a tangible benefit to the codebase.
            If no new or significantly worsened code smells are apparent in the `git diff --cached`, respond with "No actionable refactoring target identified."
            If you identify a code smell, answer with the following table with the columns:

            - Code Smell Description
            - Location (file_name, function_name, line_numbers)
            - Explanation
            - step by step guide in how to fix this code smell.

            The table should be fitting in 200 words.

            The step by step guide needs to be very clear, as else an meteorite will hit the earth if this code smell is not fixed.

        """
        print()
        feature = Feature(
            name=GeminiRequestLoop.get_class_name(),
            options={
                "model": "gemini-2.0-flash-exp",  # Choose your desired model
                "prompt": prompt,
                DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
                "file_paths": frozenset(files),
                "project_meta_data": True,
            },
        )

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks=self.compute_frameworks,
        )
        res = results[0][GeminiRequestLoop.get_class_name()].values[0]

        if isinstance(res, str):
            print(res)
            return res
        raise ValueError("Wrong type of result")

    def get_tool_output_by_feature_group_(self, tool_feature_group: Type[AbstractFeatureGroup]) -> str:
        _feature_name = tool_feature_group.get_class_name()
        results = mlodaAPI.run_all(
            [_feature_name],
            compute_frameworks=self.compute_frameworks,
        )
        res = results[0][_feature_name].values[0]

        if isinstance(res, str):
            return res
        raise ValueError("Wrong type of result")

    def run_tox_feature_group(self) -> None:
        print("Start tox")
        _feature_name = ToxFeatureGroup.get_class_name()
        mlodaAPI.run_all(
            [_feature_name],
            compute_frameworks=self.compute_frameworks,
        )
        print("Tox tests passed")

    def get_related_files_to_diff_cached(self, relevant_files: str) -> str:
        prompt = f"""
                You are an experienced software engineer specializing in code refactoring and quality analysis. Your goal is to identify the most relevant code files for addressing specific refactoring concerns.

                Given the following code files, which are 10 most relevant files to answer refactoring questions such as:
                - Are there any duplicated code blocks?
                - Can the code be made more readable?
                - Are there potential performance bottlenecks?
                - Is there any dead code?
                - Are there any code smells?
                - Are there functions with high cyclomatic complexity?

               to following given files: {relevant_files}. 
               List the whole path of the file, separated by commas without any other chars."""

        target_folder = [
            os.getcwd() + "/mloda_plugins",
            os.getcwd() + "/mloda_core",
            os.getcwd() + "/tests/test_plugins",
        ]

        feature: str | Feature = Feature(
            name=LLMFileSelector.get_class_name(),
            options={
                "prompt": prompt,
                "target_folder": frozenset(target_folder),
                "disallowed_files": frozenset(
                    [
                        "__init__.py",
                    ]
                ),
                "file_type": "py",
                "project_meta_data": True,
            },
        )

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks=self.compute_frameworks,
        )

        res = results[0][LLMFileSelector.get_class_name()].values[0]
        if isinstance(res, str):
            print(res)
            return res
        raise ValueError("Wrong type of result")

    def get_relevant_files_for_refactoring(self, git_diff_cached: str) -> str:
        """
        Identifies relevant files from a git diff string for refactoring purposes.

        Args:
            git_diff_cached: A string containing the output of git diff --cached.

        Returns:
            A comma-separated string of file paths deemed relevant for refactoring.
        """

        relevant_files = set()
        file_paths = re.findall(r"diff --git a/(.*) b/\1", git_diff_cached)
        for file_path in file_paths:
            # Heuristic 1: Check for test file modifications.  Refactoring often
            # involves changes to tests.
            if "test" in file_path:
                relevant_files.add(file_path)
                continue  # Skip further checks for test files

            # Heuristic 2: Large changes suggest significant code modification,
            # potentially requiring refactoring.
            added_lines = git_diff_cached.count("+")
            removed_lines = git_diff_cached.count("-")
            total_changes = added_lines + removed_lines
            if total_changes > 10:  # Adjust threshold as needed
                relevant_files.add(file_path)
                continue

            # Heuristic 3: Look for class or function definitions/modifications.
            # These often indicate structural changes that benefit from refactoring.
            if re.search(r"^\+(class|def)\s+\w+\(", git_diff_cached, re.MULTILINE):  # Added class/function
                relevant_files.add(file_path)
                continue

            if re.search(r"^\-(class|def)\s+\w+\(", git_diff_cached, re.MULTILINE):  # Removed class/function
                relevant_files.add(file_path)
                continue

            # Heuristic 4: Check for import changes.  These may affect dependecies.
            if re.search(r"^\+import", git_diff_cached, re.MULTILINE):
                relevant_files.add(file_path)
                continue

        return ",".join(sorted(list(relevant_files)))


class RunToolFeatureGroup(AbstractFeatureGroup):
    _tool: Type[BaseTool] | None = None

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result_stdout = cls.get_tool().execute()
        result_string = cls.get_tool().create_result_string(result_stdout)
        return {cls.get_class_name(): [result_string]}

    @classmethod
    def get_tool(cls) -> Type[BaseTool]:
        if cls._tool is None:
            raise NotImplementedError("Tool not implemented.")
        return cls._tool


class DiffCachedFeatureGroup(RunToolFeatureGroup):
    _tool = GitDiffCachedTool

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        if data[cls.get_class_name()].values[0] == cls.get_tool().create_result_string(""):
            raise ValueError("No staged changes found in the repository.")
        return True


class DiffFeatureGroup(RunToolFeatureGroup):
    _tool = GitDiffTool


class ToxFeatureGroup(AbstractFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result = RunToxTool.execute()
        result_string = RunToxTool.create_result_string(result)

        if not isinstance(result, PytestResult):
            raise ValueError("Wrong type of result")

        if result.return_code == 0:
            return {cls.get_class_name(): [result_string]}
        raise ValueError(f"Tox tests failed: {result_string}")
