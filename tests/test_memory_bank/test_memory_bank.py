import os
from typing import List, Tuple
from memory_bank.prompts.memory_bank_init_prompt import init_prompt
from memory_bank.references import MemoryBankReferences
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.llm.cli import format_array
from mloda_plugins.feature_group.experimental.llm.llm_api.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.experimental.llm.tools.available.create_folder_tool import CreateFolderTool
from mloda_plugins.feature_group.experimental.llm.tools.available.create_new_file import CreateFileTool
from mloda_plugins.feature_group.experimental.llm.tools.available.list_files_tool import ListFilesTool
from mloda_plugins.feature_group.experimental.llm.tools.available.read_file_tool import ReadFileTool
from mloda_plugins.feature_group.experimental.llm.tools.available.replace_file_tool import ReplaceFileTool
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import TextFileReader


class MdFileReader(TextFileReader):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".md",)


class NewGeminiRequestLoop(GeminiRequestLoop):
    @classmethod
    def add_final_part_of_prompt(cls) -> str:
        return ""


class TestMemoryBank:
    def test_memory_bank(self):
        feature = Feature(
            name=MdFileReader.get_class_name(),
            options={MdFileReader.get_class_name(): f"memory_bank/{MemoryBankReferences.PROJECTBRIEF.value}"},
        )

        result = mlodaAPI.run_all([feature], compute_frameworks={PandasDataframe})

        project_brief = result[0][MdFileReader.get_class_name()].values[0]

    def test_prompt_init(self) -> None:
        feature = Feature(
            name=MdFileReader.get_class_name(),
            options={MdFileReader.get_class_name(): f"memory_bank/{MemoryBankReferences.MEMORY_BANK_PROMPT.value}"},
        )

        result = mlodaAPI.run_all([feature], compute_frameworks={PandasDataframe})

        project_brief = result[0][MdFileReader.get_class_name()].values[0]
        # prompt += project_brief

        prompt = init_prompt()

        PluginLoader().all()

        tool_collection = ToolCollection()
        tool_collection.add_tool(ReadFileTool.get_class_name())
        tool_collection.add_tool(CreateFileTool.get_class_name())
        tool_collection.add_tool(CreateFolderTool.get_class_name())
        tool_collection.add_tool(ReplaceFileTool.get_class_name())
        tool_collection.add_tool(ListFilesTool.get_class_name())

        _test_classes = [NewGeminiRequestLoop]
        for _cls in _test_classes:
            features: List[Feature | str] = [
                Feature(
                    name=_cls.get_class_name(),
                    options={
                        "model": "gemini-2.0-flash-exp",
                        "prompt": prompt,
                        "tools": tool_collection,
                        DefaultOptionKeys.mloda_source_feature: frozenset(["ListDirectoryFeatureGroup"]),
                        "target_folder": frozenset(
                            [
                                # os.getcwd() + "/docs",
                                # os.getcwd() + "/mloda_plugins",
                            ]
                        ),
                        "disallowed_files": frozenset(["__init__.py"]),
                        # "project_meta_data": True,
                        "extra_ignore_patterns": frozenset(
                            ["LICENSE.TXT", "CONTRIBUTING.md", "NOTICE.md", "memory_bank/"]
                        ),
                    },
                )
            ]

            results = mlodaAPI.run_all(
                features,
                compute_frameworks={PandasDataframe},
            )

            for i, res in enumerate(results):
                assert _cls.get_class_name() in res
                assert len(res) == 1

                # print(res[GeminiRequest.get_class_name()].values)
                formatted_output = format_array(f"Result {i} values: ", res[_cls.get_class_name()].values)
                # print(formatted_output)
