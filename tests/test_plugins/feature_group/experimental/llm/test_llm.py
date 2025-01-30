import os
from typing import Any, List, Optional, Union

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.experimental.llm.gemini import GeminiRequest
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent, find_file_paths
import pytest

from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.input_data.api.api_input_data_collection import ApiInputDataCollection
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData
from mloda_core.abstract_plugins.components.link import JoinType
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
                name=GeminiRequest.get_class_name(),
                options={
                    "model": "gemini-1.5-flash-8b",
                    "prompt": "What is the meaning of life?",
                    DefaultOptionKeys.mloda_source_feature: frozenset([]),
                },
            )
        ]

        # Run the API
        results = mlodaAPI.run_all(
            features,
        )
        for res in results:
            assert GeminiRequest.get_class_name() in res
            assert len(res) == 1

    def test_llm_gemini_given_prompt(self) -> None:
        api_input_key = "TestApiInputData"

        features: List[Feature | str] = [
            Feature(
                name=GeminiRequest.get_class_name(),
                options={
                    "model": "gemini-1.5-flash-8b",
                    "prompt": "Can you count to 5?",
                    DefaultOptionKeys.mloda_source_feature: frozenset(["InputData1"]),
                },
            )
        ]

        api_input_data_collection = ApiInputDataCollection(registry={api_input_key: DataTestApiInputData})
        api_data = {
            api_input_key: {
                "InputData1": ["Answer to the following question wrong: "],
            }
        }

        # Run the API
        results = mlodaAPI.run_all(
            features,
            api_input_data_collection=api_input_data_collection,
            api_data=api_data,
            compute_frameworks={PandasDataframe},
        )

        for res in results:
            assert GeminiRequest.get_class_name() in res


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
class TestGeminiLLMFiles:
    def test_llm_gemini_files(self) -> None:
        prompt = """  I added the llm features to the mloda_plugins folder. It is still in experimental stage. I want to split it into multiple files. Into which files would you split it?"""

        features: List[Feature | str] = [
            Feature(
                name=GeminiRequest.get_class_name(),
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
            assert GeminiRequest.get_class_name() in res
            assert len(res) == 1

            # print(res[GeminiRequest.get_class_name()].values)
            formatted_output = format_array(f"Result {i} values: ", res[GeminiRequest.get_class_name()].values)
            print(formatted_output)


def format_array(prefix: str, array: Any, indent: int = 2, color: str = "34") -> str:
    """Formats a NumPy array for better console output."""
    indent_str = " " * indent
    formatted_values = ", ".join(map(str, array.tolist()))
    return f"{indent_str}\033[{color}m{prefix} [\033[0m{formatted_values}\033[{color}m]\033[0m"
