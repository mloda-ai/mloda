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
from mloda_plugins.feature_group.experimental.llm.llm_api.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.experimental.llm.llm_file_selector import LLMFileSelector
from mloda_plugins.feature_group.experimental.llm.tools.available.adjust_file_tool import AdjustFileTool
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

        for i in range(5):
            previous = self.fix_code_smell(split_files, single_code_smell, previous, actual_git_diff)
            if "AnalysisComplete" in previous:
                break

    def fix_code_smell(self, files: List[str], code_smell: str, previous: str, current_git_diff: str) -> str:
        if previous:
            _previous = f"""**Previous Refactoring Steps:**
                            {previous}
            """
        else:
            _previous = """ **Previous Refactoring Steps:**
                              No previous refactoring steps have been taken.
                        """

        prompt = f""" 

                **Objective:**

                Your objective is to automatically refactor code to eliminate a specific code smell while ensuring no existing functionality is broken.  Crucially, your refactoring must *not* involve adding or modifying project dependencies.

                **Task Description:**

                You are an automated code refactoring agent. You will be given information about a codebase and a specific code smell to address. Your task is to iteratively refactor the code using provided tools until the code smell is resolved and all tests pass.

                For each iteration, you will receive:

                1.  **Code Smell Description:**  A textual description of the code smell, including its location (file and potentially line numbers or code snippet).
                2.  **Previous Refactoring Steps:** A summary of refactoring actions already taken (if any). This is for informational context.
                3.  **Current Code Difference (Git Diff):** A `git diff` showing the changes made to the initial codebase so far. This represents the current codebase state.
                4.  **Initial Code Files:** Access to the original codebase files (for reference, though you will be working with the current state as reflected in the diff).

                **Your Actions in each iteration:**

                **Your Actions in each iteration:**

                1.  **Analyze the Code Smell:**  Understand the description and location of the code smell.
                2.  **Plan Refactoring Action:** Based on the code smell and the current code state, determine a *specific refactoring action* to take.  This action must be achievable using the provided tools (`AdjustFileTool`, `ReplaceFileTool`, `CreateFileTool`, `ReadFileTool`). Prioritize actions that directly address the described code smell. Consider if reading file content (`ReadFileTool`) is necessary to inform your refactoring plan.  *Remember: No dependency changes allowed.*
                3.  **Select Tool and Target File(s):** Choose the appropriate tool (`AdjustFileTool`, `ReplaceFileTool`, `CreateFileTool`, or `ReadFileTool`) and identify the file(s) you need to interact with. For `CreateFileTool` and `ReplaceFileTool`, ensure you have the new file content ready. For `AdjustFileTool`, determine the specific changes needed within the file. For `ReadFileTool`, specify the file to read.
                4.  **Execute Tool:**  Use the chosen tool to modify or read the file(s) according to your planned refactoring action.  Provide the necessary parameters for the tool (e.g., file path, content changes for modification tools, file path for reading).
                5.  **Run Tests:** After each code modification (using `AdjustFileTool`, `ReplaceFileTool`, or `CreateFileTool`), immediately use the `RunToxTool` to execute the test suite.  *Note: `ReadFileTool` does not modify code, so no test run is needed immediately after using it, unless it's part of a larger refactoring plan that involves code modification in the same iteration.*
                6.  **Evaluate Test Results:** Check if all tests passed (after using modification tools).
                    *   **If tests pass:** Proceed to the next iteration. Analyze if the code smell is resolved. If the code smell is considered resolved, output "AnalysisComplete". If not resolved, continue iterating from step 1, considering the current code state and the remaining code smell.
                    *   **If tests fail:** Your refactoring action has broken functionality. **Instead of immediately reverting, attempt to diagnose and fix the error.**
                        *   **6.a. Analyze Test Output:** Examine the output from `RunToxTool`. Look for error messages, stack traces, and failed test names. Try to identify the *likely cause* of the test failure based on this output.  *(Initially, this can be simple keyword matching or pattern recognition in error messages.  For example, look for "NameError", "TypeError", "AssertionError", file paths, or line numbers mentioned in the errors.)*
                        *   **6.b. Plan Fix Action (Attempt 1):** Based on your analysis of the test output, formulate a *targeted fix action*. This should be a small, focused change aimed at addressing the likely cause of the failure.  Use `AdjustFileTool` to make this fix.  *(For example, if you see a "NameError" mentioning a variable name, you might try to rename the variable back to its original name or correct a typo in its usage.)*
                        *   **6.c. Execute Fix and Re-run Tests:** Execute the planned fix using `AdjustFileTool` and then immediately run `RunToxTool` again to see if the fix resolved the issue.
                        *   **6.d. Evaluate Fix Attempt:**
                            *   **If tests pass after the fix:**  The error is resolved! Proceed to the next iteration, analyzing if the code smell is now resolved and continuing the refactoring process.
                            *   **If tests still fail after the fix attempt:** The initial fix was not successful. **Now, revert to the code state *before* the *original refactoring action that caused the tests to fail* (not just the fix attempt). Output "TestFailedAfterFixAttempt".** In more advanced scenarios, you could try a different fix, or log the failure for human review. For now, reverting after one fix attempt is a reasonable level of complexity.
                                **Tools Available:**

                *   `AdjustFileTool`:  Modify specific parts of an existing file.  Requires specifying the file path and the changes to be made (e.g., line number, content to replace, content to insert).
                *   `CreateFileTool`: Create a new file with the specified content. Requires specifying the file path and the new file content.
                *   `ReplaceFileTool`: Replace the entire content of an existing file. Requires specifying the file path and the new file content. *NOTE: DO NOT RUN THIS TOOL IF THE FILE DOES NOT EXISTS.*
                *   `RunToxTool`: Execute the test suite.  No parameters needed. Returns pass/fail status.
                *   `ReadFileTool`: Read the content of a file. Requires specifying the exact file path.

                **Constraints:**

                *   **No Dependency Changes:** You *cannot* modify any dependency-related files or add new dependencies.
                *   **Functionality Must Be Preserved:**  All existing tests must pass after each successful refactoring step when code is modified.

                **Input Data Format (for each iteration):**
                Code Smell:
                {code_smell}

                {_previous}

                Current Difference to the Codebase:
                {current_git_diff}

                **Initial Code Files**:
                """  # nosec

        tool_collection = ToolCollection()
        tool_collection.add_tool(AdjustFileTool.get_class_name())
        tool_collection.add_tool(RunToxTool.get_class_name())
        tool_collection.add_tool(ReplaceFileTool.get_class_name())
        tool_collection.add_tool(ReadFileTool.get_class_name())

        feature = Feature(
            name=GeminiRequestLoop.get_class_name(),
            options={
                "model": "gemini-2.0-flash-exp",  # Choose your desired model
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
        res = results[0][GeminiRequestLoop.get_class_name()].values[0]
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

            and the following list of potential code smells:

            - Duplicated code blocks
            - Readability issues (unclear variable names)
            
            DO NOT INCLUDE code smells related to options.

            1.  **Analyze the `git diff --cached` output and the code it modifies.**
            2.  **Identify *one specific* code smell that is newly introduced or significantly worsened by the changes in the `git diff`.**  Focus on problems that are clearly fixable and would provide a tangible benefit to the codebase.
            3.  **Create a concise summary (no more than 100 words) that includes:**
                *   A clear description of the identified code smell.
                *   The specific location(s) in the code where the code smell occurs (e.g., file name, function name, line numbers).
                *   A brief explanation of *why* this is a code smell and what negative impact it has.
                *   A *brief* suggestion for how to refactor the code to address the smell.

            If no new or significantly worsened code smells are apparent in the `git diff`, respond with "No actionable refactoring target identified."

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
