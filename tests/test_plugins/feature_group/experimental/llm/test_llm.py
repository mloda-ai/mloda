import os
from typing import Any, List, Optional, Union

from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.feature_group.experimental.llm.cli import format_array
from mloda_plugins.feature_group.experimental.llm.installed_packages_feature_group import InstalledPackagesFeatureGroup
from mloda_plugins.feature_group.experimental.llm.tools.available.multiply import MultiplyTool
from mloda_plugins.feature_group.input_data.api_data.api_data import ApiInputDataFeature
import pytest

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.experimental.llm.llm_api.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent, find_file_paths
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import ApiInputDataCollection
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda_core.abstract_plugins.components.link import JoinType, Link
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader
from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeature, SourceTuple
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature


class DataTestApiInputData(BaseApiData):
    @classmethod
    def column_names(cls) -> List[str]:
        return [
            "InputData1",
            "InputData2",
            "InputData3",
        ]


class TestReadLLMFiles:
    def test_llm_basics(self) -> None:
        class LLMBasic(ReadFileFeature):
            def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
                return FeatureName("PyFileReader")

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Union[FeatureName, str],
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                if isinstance(feature_name, FeatureName):
                    feature_name = feature_name.name
                if cls.feature_name_equal_to_class_name(feature_name):
                    return True
                return False

        project_root = os.getcwd() + "/mloda_plugins"
        file_paths = find_file_paths([project_root], "py", not_allowed_files_names=["__init__.py"])

        new_features: List[Feature | str] = []
        for f in file_paths:
            feat = Feature(name=LLMBasic.get_class_name(), options={PyFileReader.__name__: f})
            new_features.append(feat)

        result = mlodaAPI.run_all(new_features[:3], compute_frameworks={PandasDataframe})
        assert len(result) == 3

    def test_join_llm_files1(self) -> None:
        class JoinLLMFiles2(SourceInputFeature):
            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                data[cls.get_class_name()] = data[data.columns]
                return data

        project_root = os.getcwd() + "/mloda_plugins"
        file_paths = find_file_paths([project_root], "py", not_allowed_files_names=["__init__.py"])

        file_paths = list(file_paths)[:10]

        short_f_list = [os.path.split(f)[-1] for f in file_paths]

        set_fp = set()
        for cnt, f in enumerate(file_paths):
            short_f = os.path.split(f)[-1]
            right_link, left_link = None, None

            if cnt != len(file_paths) - 1:
                left_link = (ReadFileFeature, short_f_list[cnt])
                right_link = (ReadFileFeature, short_f_list[cnt + 1])

            source_tuple = SourceTuple(
                feature_name=short_f,
                source_class=PyFileReader.get_class_name(),  # type: ignore
                source_value=f,
                left_link=left_link,
                right_link=right_link,
                join_type=JoinType.APPEND,
                merge_index=short_f,
            )
            set_fp.add(source_tuple)

        features: List[Feature | str] = []
        features.append(
            Feature(
                name=JoinLLMFiles2.get_class_name(),
                options={DefaultOptionKeys.mloda_source_feature: frozenset(set_fp)},
            )
        )

        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )
        assert len(results[0]) == 10
        assert len(results) == 1

    def test_join_llm_files2(self) -> None:
        feature = Feature(
            name=ConcatenatedFileContent.get_class_name(),
            options={
                "target_folder": frozenset([os.getcwd() + "/mloda_plugins"]),
            },
        )

        results = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PandasDataframe},
        )
        assert len(results) == 1


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
class TestPlugInLLM:
    def test_llm_gemini_base(self) -> None:
        features: List[Feature | str] = [
            Feature(
                name=GeminiRequestLoop.get_class_name(),
                options={
                    "model": "gemini-1.5-flash-8b",
                    "prompt": "Does this file contain a directory structure and a list of installed packages? Answer with yes or no.",
                    DefaultOptionKeys.mloda_source_feature: frozenset([]),
                    "project_meta_data": True,
                },
            )
        ]

        # Run the API
        results = mlodaAPI.run_all(
            features,
        )
        for i, res in enumerate(results):
            assert GeminiRequestLoop.get_class_name() in res
            assert len(res) == 1
            formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
            assert "yes" in formatted_output

    def test_llm_gemini_given_prompt(self) -> None:
        api_input_key = "TestApiInputData"

        installed = Index(("InstalledPackagesFeatureGroup",))
        api_data_index = Index(("InputData1",))

        link = Link.outer((InstalledPackagesFeatureGroup, installed), (ApiInputDataFeature, api_data_index))

        features: List[Feature | str] = [
            Feature(
                name=GeminiRequestLoop.get_class_name(),
                options={
                    "model": "gemini-1.5-flash-8b",
                    "prompt": "Does this file contain a directory structure and a list of installed packages and Strawberries? Answer with yes or no.",
                    DefaultOptionKeys.mloda_source_feature: frozenset(["InputData1"]),
                    "project_meta_data": True,
                },
                link=link,
            )
        ]

        api_input_data_collection = ApiInputDataCollection(registry={api_input_key: DataTestApiInputData})
        api_data = {
            api_input_key: {
                "InputData1": [" Strawberries "],
            }
        }

        # Run the API
        results = mlodaAPI.run_all(
            features,
            api_input_data_collection=api_input_data_collection,
            api_data=api_data,
            compute_frameworks={PandasDataframe},
        )

        for i, res in enumerate(results):
            assert GeminiRequestLoop.get_class_name() in res
            formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
            assert "yes" in formatted_output.lower()


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
class TestGeminiLLMFiles:
    def test_llm_gemini_prompt(self) -> None:
        prompt = """ 
         
        Can you help me improve the Tools?
        """

        prompt = """
            Can you create me a datacreator feature group which uses the git diff --cached tool to get data?
        """

        features: List[Feature | str] = [
            Feature(
                name=GeminiRequestLoop.get_class_name(),
                options={
                    "model": "gemini-2.0-flash-exp",
                    "prompt": prompt,
                    DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
                    "target_folder": frozenset(
                        [
                            os.getcwd() + "/mloda_plugins",
                            # os.getcwd() + "/mloda_core/abstract_plugins/",
                            os.getcwd() + "/tests/test_plugins/feature_group/experimental//",
                            # os.getcwd() + "/mloda_core/api/",
                        ]
                    ),
                    "disallowed_files": frozenset(["__init__.py"]),
                    "file_type": "py",
                },
            )
        ]

        # Run the API
        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        for i, res in enumerate(results):
            assert GeminiRequestLoop.get_class_name() in res
            assert len(res) == 1

            # print(res[GeminiRequest.get_class_name()].values)
            formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
            print(formatted_output)

    def test_llm_gemini_tool_loop(self) -> None:
        # prompt = """Can you first run test_init_with_all_params, and then tox?"""
        prompt = """
            Perform the following steps in order, ensuring each step completes before proceeding to the next:
            Do not start with the next step until the previous step is completed.  
            Do not start with the unit test.
            The path of the written code is tests/test_plugins/


            1) Use tool `create_or_overwrite_new_plugin_and_write_to_file` to create a gemini api requester with a gemini 1.5 flash model. Orient yourself on existing other FeatureGroup implementations. The file_path is tests/test_plugins/new_plugin.py .
            2) Use tool `create_or_overwrite_unit_test`. The path will be tests/test_plugins/new_plugin_unit_test.py.
            3) Use tool `run_pytest` for the unit test created in step 2.
            4) Repeat step 1,2,3 until the test passes.
            """

        prompt = """ As first step: Multiply 5 by 5. As a second step, multiply the result by 2. """

        tool_collection = ToolCollection()
        tool_collection.add_tool(MultiplyTool.get_class_name())

        features: List[Feature | str] = [
            Feature(
                name=GeminiRequestLoop.get_class_name(),
                options={
                    "model": "gemini-2.0-flash-exp",
                    "prompt": prompt,
                    "tools": tool_collection,
                    DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
                    "target_folder": frozenset(
                        [
                            os.getcwd() + "/mloda_plugins",
                            os.getcwd() + "/mloda_core/",
                            # os.getcwd() + "/mloda_core/abstract_plugins/",
                            os.getcwd() + "/tests/test_plugins/feature_group/experimental/",
                            # os.getcwd() + "/mloda_core/api/",
                        ]
                    ),
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

        # Run the API
        results = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
        )

        for i, res in enumerate(results):
            assert GeminiRequestLoop.get_class_name() in res
            assert len(res) == 1

            formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
            print(formatted_output)
