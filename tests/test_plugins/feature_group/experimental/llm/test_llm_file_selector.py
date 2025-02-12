import os
from typing import List

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.llm.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.experimental.llm.llm_file_selector import LLMFileSelector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from tests.test_plugins.feature_group.experimental.llm.test_llm import format_array


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
def test_llm_file_selector() -> None:
    target_folder = [
        os.getcwd() + "/mloda_plugins",
        os.getcwd() + "/mloda_core",
    ]

    prompt = """Given the following code files, which are most relevant to answering the question 
                'How does mlodaAPI work?'? List the whole path of the file, separated by commata without any other chars."""

    features: List[Feature | str] = [
        Feature(
            name="LLMFileSelector",
            options={
                "prompt": prompt,
                "target_folder": frozenset(target_folder),
                "disallowed_files": frozenset(["__init__.py", "run.py"]),
                "file_type": "py",
            },
        )
    ]

    results = mlodaAPI.run_all(
        features,
        compute_frameworks={PandasDataframe},
    )

    # prompt = "Given the following code files, which code smells do you see?"
    prompt = "Given the following code files, choose a code smell and fix it. Show the result as a whole file."

    files = None
    for res in results:
        print(res[LLMFileSelector.get_class_name()].values[0])
        files = res[LLMFileSelector.get_class_name()].values[0]

    assert files is not None

    files = files.split(",")

    llm_feature = Feature(
        name=GeminiRequestLoop.get_class_name(),
        options={
            "model": "gemini-2.0-flash-exp",  # Choose your desired model
            "prompt": prompt,
            DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
            "file_paths": frozenset(files),
        },
    )

    results = mlodaAPI.run_all(
        [llm_feature],
        compute_frameworks={PandasDataframe},
    )

    for i, res in enumerate(results):
        formatted_output = format_array(f"Result {i} values: ", res[GeminiRequestLoop.get_class_name()].values)
        print(formatted_output)
