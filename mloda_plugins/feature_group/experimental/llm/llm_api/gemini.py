from copy import copy
from dataclasses import asdict
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple


from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.llm.installed_packages_feature_group import InstalledPackagesFeatureGroup
from mloda_plugins.feature_group.experimental.llm.list_directory_feature_group import ListDirectoryFeatureGroup
from mloda_plugins.feature_group.experimental.llm.llm_api.llm_base_request import LLMBaseRequest

from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import ToolFunctionDeclaration
from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeatureComposite

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import FunctionCall, Content, Part
    import google.api_core.exceptions
except ImportError:
    genai, FunctionCall, Content, Part, functionDeclarations, google = None, None, None, None, None, None  # type: ignore

import logging

logger = logging.getLogger(__name__)


def python_type_to_gemini_type(python_type: str) -> str:
    """Converts Python type strings to Gemini API type strings."""
    type_mapping = {
        "float": "NUMBER",
        "int": "INTEGER",
        "str": "STRING",
        "bool": "BOOLEAN",
        "number": "NUMBER",
    }
    return type_mapping.get(python_type, "STRING")  # Default to STRING if not found


def parse_tool_function_easier(function_declaration: ToolFunctionDeclaration) -> Dict[str, Any]:
    """Parses a ToolFunctionDeclaration into a dict using asdict."""
    # convert the entire tool_function to a dictionary

    if not isinstance(function_declaration, ToolFunctionDeclaration):
        raise ValueError(f"Tool function {function_declaration} does not return a ToolFunctionDeclaration.")

    output = asdict(function_declaration)

    output["parameters"] = {
        "type": "OBJECT",
        "properties": {
            param["name"]: {"type": python_type_to_gemini_type(param["type"]), "description": param["description"]}
            for param in output["parameters"]
        },
        "required": output["required"],
    }
    # remove the function as it's not part of the gemini protobuf schema
    del output["function"]
    del output["required"]
    del output["tool_result"]

    return output


class GeminiRequestLoop(LLMBaseRequest):
    """
    A feature group that interacts with Gemini, iteratively using tools
    until no tool is called by the llm anymore.

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
    def request(cls, model: str, prompt: str, model_parameters: Dict[str, Any], tools: ToolCollection | None) -> Any:
        try:
            gemini_model = cls._setup_model_if_needed(model)
            if gemini_model is not None:
                _tools = cls.parse_tools(tools)

                result = generate_content_with_retry(gemini_model, prompt, model_parameters, _tools)
                # result = gemini_model.generate_content(prompt, generation_config=model_parameters, tools=_tools)
                return result
        except Exception as e:
            logger.error(f"Error during Gemini request: {e}")
            raise

        raise ValueError("Gemini model is not set.")

    @classmethod
    def parse_tools(cls, tools: ToolCollection | None) -> List[Dict[str, Any]]:
        """Parses all tools in the ToolCollection."""

        parsed_tools: List[Dict[str, Any]] = []

        if tools is None:
            return parsed_tools
        for _, tool in tools.get_all_tools().items():
            parsed_tools.append(parse_tool_function_easier(tool))
        return parsed_tools

    @classmethod
    def _setup_model_if_needed(cls, model: str) -> "genai.GenerativeModel":  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        if genai is None:
            raise ImportError("Please install google.generativeai to use this feature.")

        genai.configure(api_key=api_key)  # type: ignore
        return genai.GenerativeModel(model)  # type: ignore

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data.iloc[0, 0] = "\n".join(data.stack().dropna().astype(str).tolist())  # Combine non-NaN values

        model = cls.get_model_from_config(features)
        prompt = cls.handle_prompt(data, features)
        model_parameters = cls.get_model_parameters(features)
        tools = cls.get_tools(features)

        all_results = []
        initial_prompt = prompt
        current_prompt = copy(initial_prompt)

        all_tool_result = ""
        while True:
            print("\n############################################\n")

            response = cls.request(model, current_prompt, model_parameters, tools)
            response_text, tool_result = cls.handle_response(response, features, tools)

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
        improved_prompt = "".join(prompt_parts)  # Efficient string joining
        print(len(improved_prompt))
        return improved_prompt

    @classmethod
    def handle_response(
        cls, response: Any, features: FeatureSet, tools: ToolCollection | None
    ) -> Tuple[str, Optional[str]]:
        response_text = ""
        tool_result = None
        if hasattr(response, "parts"):
            # Handle google.generativeai.types.generation_types.GenerateContentResponse
            tool_calls = []
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(part.function_call)

            if tool_calls:
                if tools is None:
                    raise ValueError("Tools are not set.")
                print(tool_calls)
                tool_result = cls._execute_tools(tool_calls, features, tools)
                return "", tool_result

            for part in response.parts:
                if hasattr(part, "text") and len(part.text) > 0:
                    response_text += part.text
        elif hasattr(response, "text"):
            response_text = response.text
        else:
            logger.warning(f"Response has no text or parts attribute: {response}")
            return str(response), None

        return response_text, None


def generate_content_with_retry(
    gemini_model: Any,
    prompt: str,
    generation_config: Any,
    tools: List[Dict[str, Any]],
    max_retries: int = int(os.environ.get("GEMINI_MAX_RETRIES", "5")),
    initial_retry_delay: int = int(os.environ.get("GEMINI_INITIAL_RETRY_DELAY", "10")),
    max_retry_delay: int = int(os.environ.get("GEMINI_MAX_RETRY_DELAY", "60")),
) -> Any:
    """
    Generates content with retry logic to handle ResourceExhausted (429) errors due to rate limits.

    Args:
        gemini_model: Your Gemini model object.
        prompt: The prompt for content generation.
        generation_config: Generation configuration parameters.
        tools: Tools to be used (if any).
        max_retries: Maximum number of retry attempts.
        initial_retry_delay: Initial delay in seconds before the first retry.
        max_retry_delay: Maximum delay in seconds between retries.

    Returns:
        The generated content result if successful, or raises the last exception if retries fail.
    """
    retry_attempt = 0
    while retry_attempt <= max_retries:
        try:
            result = gemini_model.generate_content(prompt, generation_config=generation_config, tools=tools)
            return result  # Successful generation, return the result

        except google.api_core.exceptions.ResourceExhausted as e:
            if e.code == 429:  # Check if it's specifically a 429 Resource Exhausted error
                retry_attempt += 1
                if retry_attempt > max_retries:
                    print(f"Maximum retries ({max_retries}) reached.  Still getting rate limited. Raising exception.")
                    raise  # Re-raise the last exception if max retries exceeded

                # Exponential backoff: delay increases with each retry
                delay = min(initial_retry_delay * (2 ** (retry_attempt - 1)), max_retry_delay)
                print(f"Rate limit hit (429). Retrying in {delay:.2f} seconds (Attempt {retry_attempt}/{max_retries}).")
                time.sleep(delay)
            else:
                # Re-raise other ResourceExhausted errors (if any other codes exist)
                raise  # It's a ResourceExhausted error but not 429, re-raise it

        except Exception as e:
            # Handle other potential exceptions (optional, depending on your needs)
            print(f"An unexpected error occurred: {e}")
            raise  # Re-raise the unexpected exception
