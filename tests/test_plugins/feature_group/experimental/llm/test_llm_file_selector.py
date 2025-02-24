import os
from typing import List

from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_plugins.feature_group.experimental.llm.cli import format_array
from mloda_plugins.feature_group.experimental.llm.llm_api.openai import OpenAIRequestLoop
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.llm.llm_file_selector import LLMFileSelector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
def test_llm_file_selector() -> None:
    target_folder = [
        os.getcwd() + "/mloda_plugins",
        os.getcwd() + "/mloda_core",
        os.getcwd() + "/tests/test_plugins",
    ]

    # prompt = """Given the following code files, which are most relevant files to answering the question
    #            'How do mloda feature groups work?? List the whole path of the file, separated by commas without any other chars."""
    # prompt = """Given the following code files, what are the most relevant files to answering the question
    #             'What is cool of mloda for data engineering? Exclude llm' List the whole path of the file, separated by commas without any other chars."""
    # prompt = """Given the following code files, what are the most relevant files to answering the question
    #             'what of this framework mloda is innovative? Exclude llm' List the whole path of the file, separated by commas without any other chars."""

    base_prompt_1 = "Given the following .py code files, which are most relevant files to answering the question"
    base_prompt_2 = "Do not include *.md or .ipynb files. List the whole full path of the file, separated by commas without any other chars."

    prompt = """ 'How do I create PostgresReader and test it?' """

    PluginLoader().all()

    features: List[Feature | str] = [
        Feature(
            name="LLMFileSelector",
            options={
                "prompt": base_prompt_1 + prompt + base_prompt_2,
                "target_folder": frozenset(target_folder),
                "disallowed_files": frozenset(
                    [
                        "__init__.py",
                        "geminipy.py",
                    ]
                ),
                "file_type": "py",
                "project_meta_data": True,
            },
        )
    ]

    results = mlodaAPI.run_all(
        features,
        compute_frameworks={PandasDataframe},
    )

    # prompt = "Given the following code files, which code smells do you see?"
    # prompt = "Given the following code files, choose a code smell and fix it. Do not focus on docs. Show the result as a whole file."
    # prompt = "Given the following code files, explain why this is innovative?"
    prompt = (
        "Given the following code files, give me an example for how to integrate feature groups? Show only the code."
    )

    files = None
    for res in results:
        print(res[LLMFileSelector.get_class_name()].values[0])
        files = res[LLMFileSelector.get_class_name()].values[0]

    assert files is not None

    files = files.split(",")
    new_files = []
    for file in files:
        new_files.append(file.strip("\n"))

        if "\n" in file:
            print(file)

    _cls = OpenAIRequestLoop.get_class_name()

    llm_feature = Feature(
        name=_cls,
        options={
            "model": "gemini-2.0-flash-exp",  # Choose your desired model
            "prompt": prompt,
            DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
            "file_paths": frozenset(files),
            "project_meta_data": True,
        },
    )

    try:
        results = mlodaAPI.run_all(
            [llm_feature],
            compute_frameworks={PandasDataframe},
        )
    except ValueError as e:
        print(e)
        print(frozenset(files))
        return

    for i, res in enumerate(results):
        formatted_output = format_array(f"Result {i} values: ", res[_cls].values)
        print(formatted_output)
