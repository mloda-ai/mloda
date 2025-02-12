from typing import Any, Dict, Set, Type, Union, List


from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import PytestResult, ToolFunctionDeclaration


try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import FunctionCall, Content, Part
except ImportError:
    genai, FunctionCall, Content, Part = None, None, None, None  # type: ignore

import logging

logger = logging.getLogger(__name__)


class LLMBaseRequest(AbstractFeatureGroup):
    model = "model"
    prompt = "prompt"
    temperature = "temperature"
    model_parameters = "model_parameters"
    tools = "tools"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        model = cls.get_model_from_config(features)
        prompt = cls.handle_prompt(data, features)
        model_parameters = cls.get_model_parameters(features)
        tools = cls.get_tools(features)

        response = cls.request(model, prompt, model_parameters, tools)
        transformed_data = cls.handle_response(response, features, tools)

        return transformed_data

    @classmethod
    def get_tools(cls, features: FeatureSet) -> ToolCollection | None:
        tools = features.get_options_key(cls.tools)

        if tools is None:
            return None

        if not isinstance(tools, ToolCollection):
            raise ValueError(f"Tools must be a ToolCollection. {tools}")

        return tools

    @classmethod
    def get_model_from_config(cls, features: FeatureSet) -> str:
        model = features.get_options_key(cls.model)
        if model is None:
            raise ValueError(f"Model was not set for {cls.__name__}")
        if not isinstance(model, str):
            raise ValueError(f"Model must be a string. {model}")
        return model

    @classmethod
    def get_model_parameters(cls, features: FeatureSet) -> Dict[str, Any]:
        model_parameters = features.get_options_key(cls.model_parameters) or {}
        if not isinstance(model_parameters, dict):
            raise ValueError(f"Model parameters must be a dict. {model_parameters}")

        return model_parameters

    @classmethod
    def request(cls, model: str, prompt: str, model_parameters: Dict[str, Any], tools: ToolCollection | None) -> Any:
        """
        Abstract method to make the request to the LLM
        """
        raise NotImplementedError

    @classmethod
    def handle_prompt(cls, data: Any, features: FeatureSet) -> str:
        data_prompt = "" if data is None or data.empty else str(data.iloc[0, 0])
        option_prompt = features.get_options_key(cls.prompt) or ""

        if not option_prompt and not data_prompt:
            raise ValueError(f"Prompt was not set for {cls.__name__}")

        if option_prompt != "":
            option_prompt = f""" {option_prompt} """

        return f"{option_prompt}\nContext:\n{data_prompt} End Context\n "

    @classmethod
    def handle_response(cls, response: Any, features: FeatureSet, tools: ToolCollection | None) -> pd.DataFrame:
        feature_name = next(iter(features.get_all_names()))
        response_text = ""

        if hasattr(response, "parts"):
            # Handle google.generativeai.types.generation_types.GenerateContentResponse
            tool_calls = []
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append(part.function_call)

            if tool_calls:
                if tools is None:
                    raise ValueError("Tools are not set.")
                tool_result = cls._execute_tools(tool_calls, features, tools)
                return pd.DataFrame({feature_name: [tool_result]})

            for part in response.parts:
                if hasattr(part, "text") and len(part.text) > 0:
                    response_text += part.text

        elif hasattr(response, "text"):
            response_text = response.text
        else:
            logger.warning(f"Response has no text or parts attribute: {response}")
            return pd.DataFrame({feature_name: [str(response)]})

        return pd.DataFrame({feature_name: [response_text]})

    @classmethod
    def _execute_tools(cls, tool_calls: List[FunctionCall], features: FeatureSet, tools: ToolCollection) -> str:
        """Executes all tool calls."""
        tool_results = []
        for tool_call in tool_calls:
            tool_results.append(cls._parse_response(tool_call, features, tools))

        return "\n ".join(tool_results)

    @classmethod
    def _parse_response(cls, response: Any, features: FeatureSet, tools: ToolCollection) -> str:
        if isinstance(response, FunctionCall):
            tool_call: FunctionCall = response
        else:
            raise ValueError(f"Response is not a FunctionCall: {response}, {type(response)}")

        args_dict = dict(tool_call.args)
        tool_result = cls._execute_tool(tool_call.name, args_dict, tools)

        return tool_result

    @classmethod
    def _execute_tool(cls, tool_name: str, args: Dict[str, Any], tools: ToolCollection) -> str:
        """
        Executes the tool and apppend the tool_result string.
        """
        tool = tools.get_tool(tool_name)
        if not isinstance(tool, ToolFunctionDeclaration):
            raise ValueError(f"Tool {tool_name} not found in tool mappings.")

        tool_result = tool.function(**args)
        if not isinstance(tool_result, str) and not isinstance(tool_result, PytestResult):
            raise ValueError(f"Tool result must be a string or PytestResult. {tool_name}, {tool_result}")

        return_tool_result = tool.tool_result(tool_result, **args)
        if not isinstance(return_tool_result, str):
            raise ValueError(f"Tool result must be a string. {tool_name}, {return_tool_result}")
        return return_tool_result

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}
