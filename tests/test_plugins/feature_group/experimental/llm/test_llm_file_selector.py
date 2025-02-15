import os
from typing import List

from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda_plugins.feature_group.experimental.llm.cli import format_array
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.llm.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.experimental.llm.llm_file_selector import LLMFileSelector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
def test_llm_file_selector() -> None:
    target_folder = [
        os.getcwd() + "/mloda_plugins",
        os.getcwd() + "/mloda_core",
    ]

    # prompt = """Given the following code files, which are most relevant files to answering the question
    #            'How do mloda feature groups work?? List the whole path of the file, separated by commas without any other chars."""
    # prompt = """Given the following code files, what are the most relevant files to answering the question
    #             'What is cool of mloda for data engineering? Exclude llm' List the whole path of the file, separated by commas without any other chars."""
    prompt = """Given the following code files, what are the most relevant files to answering the question 
                 'what of this framework mloda is innovative? Exclude llm' List the whole path of the file, separated by commas without any other chars."""

    PluginLoader().all()

    features: List[Feature | str] = [
        Feature(
            name="LLMFileSelector",
            options={
                "prompt": prompt,
                "target_folder": frozenset(target_folder),
                "disallowed_files": frozenset(
                    [
                        "__init__.py",
                        "run.py",
                        "request.py",
                        "compute_frame_work.py",
                        "execution_plan.py",
                        "source_input_feature.py",
                        "engine.py",
                        "abstract_feature_group.py",
                    ]
                ),
                "file_type": "py",
            },
        )
    ]

    for i in range(5):
        print(f"\nAttempt {i}")
        try:
            results = mlodaAPI.run_all(
                features,
                compute_frameworks={PandasDataframe},
            )
        except ValueError:
            continue
        break

    # prompt = "Given the following code files, which code smells do you see?"
    # prompt = "Given the following code files, choose a code smell and fix it. Do not focus on docs. Show the result as a whole file."
    prompt = "Given the following code files, explain why this is innovative?"

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

    llm_feature = Feature(
        name=GeminiRequestLoop.get_class_name(),
        options={
            "model": "gemini-2.0-flash-exp",  # Choose your desired model
            "prompt": prompt,
            DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
            "file_paths": frozenset(files),
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
        formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
        print(formatted_output)
