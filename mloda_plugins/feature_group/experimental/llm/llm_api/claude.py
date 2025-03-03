from copy import copy
from dataclasses import asdict
import os
import time
from typing import Any, Dict, List, Tuple, Union


from mloda_core.abstract_plugins.components.feature_set import FeatureSet

from mloda_plugins.feature_group.experimental.llm.llm_api.llm_base_request import LLMBaseApi
from mloda_plugins.feature_group.experimental.llm.llm_api.request_loop import RequestLoop
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import ToolFunctionDeclaration

try:
    import anthropic
    from anthropic.types.tool_use_block import ToolUseBlock
except ImportError:
    anthropic, ToolUseBlock = None, None

try:
    import pandas as pd
except ImportError:
    pd = None


import logging

logger = logging.getLogger(__name__)


def python_type_to_claude_type(python_type: str) -> str:
    """Converts Python type strings to Claude API type strings."""
    type_mapping = {
        "float": "number",
        "int": "integer",
        "str": "string",
        "bool": "boolean",
        "number": "number",
    }
    return type_mapping.get(python_type, "string")  # Default to string if not found


def parse_tool_function_for_claude(function_declaration: ToolFunctionDeclaration) -> Dict[str, Any]:
    """Parses a ToolFunctionDeclaration into a dict formatted for Claude function calling.

    The output will have the following structure compatible with Anthropic's API:
    {
        "name": <function name>,
        "description": <function description>,
        "input_schema": {
            "type": "object",
            "properties": {
                <param_name>: {
                    "type": <claude json schema type>,
                    "description": <param description>
                },
                ...
            },
            "required": [<list of required param names>]
        }
    }
    """
    if not isinstance(function_declaration, ToolFunctionDeclaration):
        raise ValueError(f"Tool function {function_declaration} does not return a ToolFunctionDeclaration.")

    # Build the parameters schema
    input_schema = {
        "type": "object",
        "properties": {},
        "required": function_declaration.required,
    }

    for param in function_declaration.parameters:
        param_dict = asdict(param)
        input_schema["properties"][param_dict["name"]] = {  # type: ignore
            "type": python_type_to_claude_type(param_dict["type"]),
            "description": param_dict["description"],
        }

    # Construct the final tool structure
    tool_dict = {
        "name": function_declaration.name,
        "description": function_declaration.description,
        "input_schema": input_schema,
    }

    return tool_dict


class ClaudeAPI(LLMBaseApi):
    @classmethod
    def request(
        cls,
        model: str,
        prompt: Union[str, List[Dict[str, str]]],
        model_parameters: Dict[str, Any],
        tools: ToolCollection | None,
    ) -> Any:
        if isinstance(prompt, str):
            raise ValueError("Claude requires a list of messages, not a single string.")

        try:
            claude_client = cls._setup_model_if_needed(model)
            if claude_client is not None:
                _tools = cls.parse_tools(tools)
                result = cls.generate_response(claude_client, model, prompt, _tools)
                return result
        except Exception as e:
            logger.error(f"Error during Claude request: {e}")
            raise

        raise ValueError("Claude model is not set.")

    @classmethod
    def _setup_model_if_needed(cls, model: str) -> Any:
        api_key = os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY environment variable is not set.")

        if anthropic is None:
            raise ImportError("Please install the anthropic package to use Claude.")

        claude_client = anthropic.Anthropic(api_key=api_key)
        return claude_client

    @classmethod
    def parse_tools(cls, tools: ToolCollection | None) -> List[Dict[str, Any]]:
        """Parses all tools in the ToolCollection for Claude."""
        parsed_tools: List[Dict[str, Any]] = []
        if tools is None:
            return parsed_tools

        for _, tool in tools.get_all_tools().items():
            parsed_tools.append(parse_tool_function_for_claude(tool))

        return parsed_tools

    @classmethod
    def handle_response(
        cls, response: Any, features: FeatureSet, tools: ToolCollection | None
    ) -> Tuple[List[Dict[str, str]], str]:
        responses = []
        used_tool = ""

        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    responses.append({"text": block.text})

                if isinstance(block, ToolUseBlock):
                    if tools is None:
                        raise ValueError("Tools are not set.")

                    tool_dict = {"name": block.name, "args": block.input}
                    tool_result = cls._execute_tools([tool_dict], features, tools)
                    if tool_result:
                        responses.append({"tool": tool_result})
                        used_tool += tool_result
        else:
            logger.warning(f"Response has unexpected structure: {response}")
            return str(response), ""  # type: ignore

        return responses, used_tool

    def generate_response(
        client: Any,
        model: str,
        messages: List[Dict[str, str]],
        tools: Any,
        max_retries: int = 5,
        initial_retry_delay: int = 10,
        max_retry_delay: int = 60,
    ) -> Any:
        """
        Generates content from Claude with retry logic for rate limits.
        """
        # Override defaults with environment variables if present
        max_retries = int(os.environ.get("CLAUDE_MAX_RETRIES", str(max_retries)))
        initial_retry_delay = int(os.environ.get("CLAUDE_INITIAL_RETRY_DELAY", str(initial_retry_delay)))
        max_retry_delay = int(os.environ.get("CLAUDE_MAX_RETRY_DELAY", str(max_retry_delay)))

        retry_attempt = 0
        while retry_attempt <= max_retries:
            try:
                message_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000,
                }

                if tools:
                    message_params["tools"] = tools

                result = client.messages.create(**message_params)
                return result
            except Exception as e:
                is_rate_limit_error = False
                if hasattr(e, "status_code") and e.status_code == 429:
                    is_rate_limit_error = True

                if is_rate_limit_error:
                    retry_attempt += 1
                    if retry_attempt > max_retries:
                        print(f"Maximum retries ({max_retries}) reached for CLAUDE. Raising exception.")
                        raise
                    delay = min(initial_retry_delay * (2 ** (retry_attempt - 1)), max_retry_delay)
                    print(
                        f"Rate limit hit for CLAUDE. Retrying in {delay:.2f} seconds (Attempt {retry_attempt}/{max_retries})."
                    )
                    time.sleep(delay)
                else:
                    print(f"An unexpected error occurred during CLAUDE request: {e}")
                    raise

        raise Exception(f"Maximum retries ({max_retries}) reached for CLAUDE without a successful response.")


class ClaudeRequestLoop(RequestLoop):
    @classmethod
    def api(cls) -> Any:
        return ClaudeAPI

    @classmethod
    def initial_prompt_message(cls, messages: Any, initial_prompt: str) -> Tuple[Any, Any]:
        if not messages:
            messages = [{"role": "user", "content": initial_prompt}]
            _messages = copy(messages)
        else:
            _messages = copy(messages)
            _messages.append({"role": "user", "content": cls.add_final_part_of_prompt()})

        return messages, _messages

    @classmethod
    def add_tool_response_to_messages(cls, messages: Any, response: str) -> Any:
        messages.append({"role": "user", "content": response})
        return messages

    @classmethod
    def add_text_response_to_messages(cls, messages: Any, response: str) -> Any:
        messages.append({"role": "assistant", "content": response})
        return messages
