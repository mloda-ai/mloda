from dataclasses import asdict
import os
import time
from typing import Any, Dict, List, Tuple


from mloda_core.abstract_plugins.components.feature_set import FeatureSet

from mloda_plugins.feature_group.experimental.llm.llm_api.llm_base_request import LLMBaseApi

from mloda_plugins.feature_group.experimental.llm.llm_api.request_loop import RequestLoop
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import ToolFunctionDeclaration


try:
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import FunctionCall, Content, Part
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


class GeminiAPI(LLMBaseApi):
    @classmethod
    def request(
        cls,
        model: str,
        prompt: str | List[Dict[str, str]],
        model_parameters: Dict[str, Any],
        tools: ToolCollection | None,
    ) -> Any:
        if not isinstance(prompt, str):
            raise ValueError("Gemini does not support multiple prompts. Provide a single")

        try:
            gemini_model = cls._setup_model_if_needed(model)
            if gemini_model is not None:
                _tools = cls.parse_tools(tools)

                result = cls.generate_response(gemini_model, prompt, model_parameters, _tools)
                return result
        except Exception as e:
            logger.error(f"Error during Gemini request: {e}")
            raise

        raise ValueError("Gemini model is not set.")

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
    def parse_tools(cls, tools: ToolCollection | None) -> List[Dict[str, Any]]:
        """Parses all tools in the ToolCollection."""

        parsed_tools: List[Dict[str, Any]] = []

        if tools is None:
            return parsed_tools
        for _, tool in tools.get_all_tools().items():
            parsed_tools.append(parse_tool_function_easier(tool))
        return parsed_tools

    @classmethod
    def handle_response(
        cls, response: Any, features: FeatureSet, tools: ToolCollection | None
    ) -> Tuple[List[Dict[str, str]], str]:
        responses = []
        used_tool = ""

        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "text") and part.text:
                    responses.append({"text": part.text})

                if hasattr(part, "function_call") and part.function_call:
                    if tools is None:
                        raise ValueError("Tools are not set.")

                    tool_dict = {
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                    }

                    tool_result = cls._execute_tools([tool_dict], features, tools)
                    if tool_result:
                        responses.append({"tool": tool_result})
                        used_tool += tool_result
        else:
            logger.warning(f"Response has no text or parts attribute: {response}")
            return [], ""

        return responses, used_tool

    def generate_response(
        llm_model: Any,
        prompt: str,  # Single prompt expected for Gemini
        generation_config: Dict[str, Any],
        tools: Any,
        max_retries: int = 5,
        initial_retry_delay: int = 10,
        max_retry_delay: int = 60,
    ) -> Any:
        """
        Generates
        content from Gemini with retry logic for rate limits.
        """
        # Override defaults with environment variables if present
        max_retries = int(os.environ.get("GEMINI_MAX_RETRIES", str(max_retries)))
        initial_retry_delay = int(os.environ.get("GEMINI_INITIAL_RETRY_DELAY", str(initial_retry_delay)))
        max_retry_delay = int(os.environ.get("GEMINI_MAX_RETRY_DELAY", str(max_retry_delay)))

        retry_attempt = 0
        while retry_attempt <= max_retries:
            try:
                if isinstance(prompt, list):
                    raise ValueError("Gemini does not support multiple prompts. Provide a single prompt.")
                result = llm_model.generate_content(prompt, generation_config=generation_config, tools=tools)
                return result
            except Exception as e:
                is_rate_limit_error = False
                if e.code == 429:  # type: ignore
                    is_rate_limit_error = True

                if is_rate_limit_error:
                    retry_attempt += 1
                    if retry_attempt > max_retries:
                        print(f"Maximum retries ({max_retries}) reached for GEMINI. Raising exception.")
                        raise
                    delay = min(initial_retry_delay * (2 ** (retry_attempt - 1)), max_retry_delay)
                    print(
                        f"Rate limit hit for GEMINI. Retrying in {delay:.2f} seconds (Attempt {retry_attempt}/{max_retries})."
                    )
                    time.sleep(delay)
                else:
                    print(f"An unexpected error occurred during GEMINI request: {e}")
                    raise

        raise Exception(f"Maximum retries ({max_retries}) reached for GEMINI without a successful response.")


class GeminiRequestLoop(RequestLoop):
    @classmethod
    def api(cls) -> Any:
        return GeminiAPI
