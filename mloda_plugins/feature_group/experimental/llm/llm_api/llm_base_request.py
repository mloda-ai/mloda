from abc import ABC
from typing import Any, Dict, Set, Type, Union, List


from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import PytestResult, ToolFunctionDeclaration


import logging

logger = logging.getLogger(__name__)


class LLMBaseApi(ABC):
    @classmethod
    def request(
        cls,
        model: str,
        prompt: Union[str, List[Dict[str, str]]],
        model_parameters: Dict[str, Any],
        tools: ToolCollection | None,
    ) -> Any:
        """
        Abstract method to make the request to the LLM
        """
        raise NotImplementedError

    @classmethod
    def _execute_tools(cls, tool_calls: List[Any], features: FeatureSet, tools: ToolCollection) -> str:
        """Executes all tool calls."""
        tool_results = []
        for tool_call in tool_calls:
            tool_results.append(cls._parse_response(tool_call, features, tools))

        return "\n ".join(tool_results)

    @classmethod
    def _parse_response(cls, response: Any, features: FeatureSet, tools: ToolCollection) -> str:
        if hasattr(response, "args"):
            args_dict = dict(response.args)
            _name = response.name
        else:
            args_dict = response["args"]
            _name = response["name"]

        tool_result = cls._execute_tool(_name, args_dict, tools)
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


class LLMBaseRequest(AbstractFeatureGroup):
    model = "model"
    prompt = "prompt"
    temperature = "temperature"
    model_parameters = "model_parameters"
    tools = "tools"

    @classmethod
    def api(cls) -> Type[LLMBaseApi]:
        raise NotImplementedError

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise NotImplementedError

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
    def handle_prompt(cls, data: Any, features: FeatureSet) -> str:
        data_prompt = "" if data is None or data.empty else str(data.iloc[0, 0])
        option_prompt = features.get_options_key(cls.prompt) or ""

        if not option_prompt and not data_prompt:
            raise ValueError(f"Prompt was not set for {cls.__name__}")

        if option_prompt != "":
            option_prompt = f""" {option_prompt} """

        return f"{option_prompt}\nContext:\n{data_prompt} End Context\n "

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}
