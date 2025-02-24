from copy import copy
from typing import Any, Set


from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.llm.installed_packages_feature_group import InstalledPackagesFeatureGroup
from mloda_plugins.feature_group.experimental.llm.list_directory_feature_group import ListDirectoryFeatureGroup
from mloda_plugins.feature_group.experimental.llm.llm_api.llm_base_request import LLMBaseRequest

from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeatureComposite


try:
    import pandas as pd
except ImportError:
    pd = None


class RequestLoop(LLMBaseRequest):
    """
    A feature group that interacts with LLMs via a request loop.
    It works by sending requests as long as tools are called.

    If the `project_meta_data` option is set to True, the feature group requires that the requested of the
    feature GeminiRequestLoop set the link between SourceInputFeatureComposite and one of the other feature groups.

    The specific SourceInputFeatureComposite depends on your projects access, thus we don t know it here.
    If you encounter this feature and want an automatic link setting, we could develop this.
    However, it was deprioritized due to the complexity of the feature.

    Example in test test_llm_gemini_given_prompt:
        installed = Index(("InstalledPackagesFeatureGroup",))
        api_data = Index(("InputData1",))
        link = Link.outer((InstalledPackagesFeatureGroup, installed), (ApiInputDataFeature, api_data))

        Feature(
                    name=GeminiRequestLoop.get_class_name(),
                    options={
                        ...
                        "project_meta_data": True,
                    },
                    link=link
        )
    """

    @classmethod
    def api(cls) -> Any:
        NotImplementedError

    def input_features(self, options: Options, feature_name: FeatureName) -> Set[Feature] | None:
        features = SourceInputFeatureComposite.input_features(options, feature_name) or set()

        if options.get("project_meta_data") is not None:
            idx_installed = Index(("InstalledPackagesFeatureGroup",))
            idx_list_dir = Index(("ListDirectoryFeatureGroup",))

            link = Link.append(
                (ListDirectoryFeatureGroup, idx_list_dir), (InstalledPackagesFeatureGroup, idx_installed)
            )

            list_dir = Feature(
                name=ListDirectoryFeatureGroup.get_class_name(),
                link=link,
                index=idx_list_dir,
            )
            installed = Feature(
                name=InstalledPackagesFeatureGroup.get_class_name(),
                index=idx_installed,
            )

            features.add(list_dir)
            features.add(installed)

        return features

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        model, initial_prompt, model_parameters, tools = cls.read_properties(data, features)

        current_prompt = copy(initial_prompt)

        all_results = []
        all_tool_result = ""
        while True:
            print("\n############################################\n")

            response = cls.api().request(model, current_prompt, model_parameters, tools)
            response_text, tool_result = cls.api().handle_response(response, features, tools)

            if response_text:
                response_text = response_text.strip()
            all_results.append(response_text)

            if not tool_result:
                break
            else:
                print(tool_result)

            current_prompt = cls.loop_prompt(initial_prompt, tool_result, all_tool_result)

            all_tool_result += tool_result

        return pd.DataFrame({cls.get_class_name(): ["\n".join(all_results)]})

    @classmethod
    def read_properties(cls, data: Any, features: FeatureSet) -> Any:
        data.iloc[0, 0] = "\n".join(data.stack().dropna().astype(str).tolist())  # Combine non-NaN values

        model = cls.get_model_from_config(features)
        prompt = copy(cls.handle_prompt(data, features))
        model_parameters = cls.get_model_parameters(features)
        tools = cls.get_tools(features)

        return model, prompt, model_parameters, tools

    @classmethod
    def loop_prompt(cls, current_prompt: str, tool_result: str, all_tool_result: str) -> str:
        prompt_parts = [
            current_prompt,
            "\n\n--------------------------------------------------\n",
            "**Previous Steps:**",
        ]

        if all_tool_result:
            prompt_parts.extend(
                [
                    "\nResults from all prior tool executions:\n",
                    all_tool_result,
                    "\n\n--------------------------------------------------\n",
                ]
            )

        if all_tool_result != "":
            current_prompt += f"\n\n Previous steps were completed with the following results: {all_tool_result}. \n\n"

        prompt_parts.extend(
            [
                "\n**Most Recent Step:**",
                "\nResult from the last tool execution:\n",
                tool_result,
                "\n\n--------------------------------------------------\n",
                """
            **Instructions:**

            You are an expert reasoning agent. Given the information above (the original instructions, prior steps, and the most recent step result), carefully analyze the situation and determine the next action to take. 

            *   If the goal is complete, respond with the `Final Answer: ` followed by the final answer.
            *   If another tool is needed, determine the correct tool to use, and what input it needs.
            """,
            ]
        )
        improved_prompt = "".join(prompt_parts)
        print(len(improved_prompt))
        return improved_prompt
