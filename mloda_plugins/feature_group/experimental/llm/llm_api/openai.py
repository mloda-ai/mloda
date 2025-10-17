from copy import copy
from dataclasses import asdict
import json
import os
import time
from typing import Any, Dict, List, Tuple, Union


from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.experimental.llm.llm_api.llm_base_request import LLMBaseApi
from mloda_plugins.feature_group.experimental.llm.llm_api.request_loop import RequestLoop
from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import ToolCollection
from mloda_plugins.feature_group.experimental.llm.tools.tool_data_classes import ToolFunctionDeclaration

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import pandas as pd
except ImportError:
    pd = None


import logging

logger = logging.getLogger(__name__)


def python_type_to_openapi_type(python_type: str) -> str:
    """Converts Python type strings to OpenAPI/JSON Schema type strings."""
    type_mapping = {
        "float": "number",
        "int": "integer",
        "str": "string",
        "bool": "boolean",
        "number": "number",
    }
    return type_mapping.get(python_type, "string")  # Default to string if not found


def parse_tool_function_for_openai(function_declaration: ToolFunctionDeclaration) -> Dict[str, Any]:
    """Parses a ToolFunctionDeclaration into a dict formatted for OpenAI function calling.

    The output will have the following structure:
    {
        "type": "function",
        "function": {
            "name": <function name>,
            "description": <function description>,
            "parameters": {
                "type": "object",
                "properties": {
                    <param_name>: {
                        "type": <openai json schema type>,
                        "description": <param description>
                    },
                    ...
                },
                "required": [<list of required param names>],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    """
    if not isinstance(function_declaration, ToolFunctionDeclaration):
        raise ValueError(f"Tool function {function_declaration} does not return a ToolFunctionDeclaration.")

    # Build the parameters schema.
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": function_declaration.required,
        "additionalProperties": False,
    }

    for param in function_declaration.parameters:
        param_dict = asdict(param)
        parameters_schema["properties"][param_dict["name"]] = {  # type: ignore
            "type": python_type_to_openapi_type(param_dict["type"]),
            "description": param_dict["description"],
        }

    # Construct the final tool structure.
    tool_dict = {
        "type": "function",
        "function": {
            "name": function_declaration.name,
            "description": function_declaration.description,
            "parameters": parameters_schema,
            "strict": True,
        },
    }

    return tool_dict


class OpenAIAPI(LLMBaseApi):
    @classmethod
    def request(
        cls,
        model: str,
        prompt: Union[str, List[Dict[str, str]]],
        model_parameters: Dict[str, Any],
        tools: ToolCollection | None,
    ) -> Any:
        if isinstance(prompt, str):
            raise ValueError("OpenAI requires a list of messages, not a single prompt.")

        try:
            openai_client = cls._setup_model_if_needed(model)
            if openai_client is not None:
                tools = cls.parse_tools(tools)  # type: ignore
                result = cls.generate_response(openai_client, model, prompt, tools)
                return result
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise
        raise ValueError("OpenAI model is not set.")

    @classmethod
    def _setup_model_if_needed(cls, model: str) -> Any:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        return client

    @classmethod
    def parse_tools(cls, tools: ToolCollection | None) -> List[Dict[str, Any]]:
        """Parses all tools in the ToolCollection for OpenAI."""
        parsed_tools: List[Dict[str, Any]] = []
        if tools is None:
            return parsed_tools
        for _, tool in tools.get_all_tools().items():
            parsed_tools.append(parse_tool_function_for_openai(tool))
        return parsed_tools

    @classmethod
    def handle_response(
        cls, response: Any, features: FeatureSet, tools: ToolCollection | None
    ) -> Tuple[List[Dict[str, str]], str]:
        responses = []
        used_tool = ""

        if hasattr(response, "choices") and response.choices:
            for choice in response.choices:
                message = choice.message

                if hasattr(message, "content") and message.content:
                    responses.append({"text": message.content})

                if message.tool_calls:  # Check for function call
                    if tools is None:
                        raise ValueError("Tools are not set.")

                    new_tools = []
                    for e in message.tool_calls:
                        tool_dict = {"name": e.function.name, "args": json.loads(e.function.arguments)}
                        new_tools.append(tool_dict)

                    tool_result = cls._execute_tools(new_tools, features, tools)
                    if tool_result:
                        responses.append({"tool": tool_result})
                        used_tool += tool_result

        else:
            logger.warning(f"Response has no text or choices attribute: {response}")
            return [], ""

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
        Generates content from OpenAI with retry logic for rate limits.
        """
        # Override defaults with environment variables if present
        max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", str(max_retries)))
        initial_retry_delay = int(os.environ.get("OPENAI_INITIAL_RETRY_DELAY", str(initial_retry_delay)))
        max_retry_delay = int(os.environ.get("OPENAI_MAX_RETRY_DELAY", str(max_retry_delay)))

        retry_attempt = 0
        while retry_attempt <= max_retries:
            try:
                result = client.chat.completions.create(model=model, messages=messages, tools=tools)
                return result
            except Exception as e:
                # Check for an OpenAI rate limit error; adjust the error check as needed
                is_rate_limit_error = False
                if e.code == 429:  # type: ignore
                    is_rate_limit_error = True

                if is_rate_limit_error:
                    retry_attempt += 1
                    if retry_attempt > max_retries:
                        print(f"Maximum retries ({max_retries}) reached for OPENAI. Raising exception.")
                        raise
                    delay = min(initial_retry_delay * (2 ** (retry_attempt - 1)), max_retry_delay)
                    print(
                        f"Rate limit hit for OPENAI. Retrying in {delay:.2f} seconds (Attempt {retry_attempt}/{max_retries})."
                    )
                    time.sleep(delay)
                else:
                    print(f"An unexpected error occurred during OPENAI request: {e}")
                    raise

        raise Exception(f"Maximum retries ({max_retries}) reached for OPENAI without a successful response.")


class OpenAIRequestLoop(RequestLoop):
    """
    Base class for integrating OpenAI LLM API into mloda feature pipelines.

    This feature group provides integration with OpenAI-compatible APIs (including
    Gemini's OpenAI compatibility layer), handling chat completion formatting,
    response parsing, function calling, rate limiting, and conversation management.

    ## Key Capabilities

    - Chat completions with message history
    - Function/tool calling with strict schema validation
    - Automatic retry logic with exponential backoff for rate limits
    - Support for OpenAI-compatible endpoints (including Gemini proxy)
    - Multi-turn conversation support
    - Integration with mloda's ToolCollection framework

    ## Common Use Cases

    - Chat-based interactions and conversations
    - Function calling for structured outputs
    - Code generation and analysis
    - Question answering with context
    - Multi-step reasoning with tool use
    - API-agnostic LLM integration (works with compatible providers)

    ## Usage Examples

    ### Basic Chat Completion

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="OpenAIRequestLoop",
        options=Options(
            context={
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "Explain this code snippet"}
                ],
            }
        )
    )
    ```

    ### Multi-Turn Conversation

    ```python
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "How do I read a CSV file?"},
        {"role": "assistant", "content": "You can use pandas.read_csv()..."},
        {"role": "user", "content": "Show me an example"},
    ]

    feature = Feature(
        name="OpenAIRequestLoop",
        options=Options(
            context={
                "model": "gpt-4",
                "messages": messages,
            }
        )
    )
    ```

    ### Function Calling

    ```python
    from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import (
        ToolCollection
    )

    feature = Feature(
        name="OpenAIRequestLoop",
        options=Options(
            context={
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": "What's the weather in NYC?"}
                ],
                "tools": tool_collection,  # Contains weather functions
            }
        )
    )
    ```

    ### Using Gemini via OpenAI Compatibility

    ```python
    # Note: This implementation uses Gemini's OpenAI-compatible endpoint
    feature = Feature(
        name="OpenAIRequestLoop",
        options=Options(
            context={
                "model": "gemini-1.5-flash",
                "messages": [
                    {"role": "user", "content": "Analyze this data"}
                ],
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)

    - `model` (required): Model name (e.g., "gpt-4", "gemini-1.5-flash")
    - `messages` (required): List of message dicts with "role" and "content"
    - `tools`: ToolCollection instance for function calling
    - Additional OpenAI chat completion parameters

    ### Group Parameters

    Currently none for OpenAIRequestLoop.

    ## Configuration

    ### Environment Variables

    - `GEMINI_API_KEY` (required): API key (defaults to Gemini endpoint)
    - `OPENAI_MAX_RETRIES`: Maximum retry attempts (default: 5)
    - `OPENAI_INITIAL_RETRY_DELAY`: Initial retry delay in seconds (default: 10)
    - `OPENAI_MAX_RETRY_DELAY`: Maximum retry delay in seconds (default: 60)

    ### Base URL

    Defaults to Gemini's OpenAI-compatible endpoint:
    `https://generativelanguage.googleapis.com/v1beta/openai/`

    Can be modified to use OpenAI or other compatible endpoints.

    ## Output Format

    Returns chat completion text in a DataFrame column named `OpenAIRequestLoop`.
    For function calls, returns the tool execution results.

    ## Message Format

    OpenAI chat completion format with roles:
    ```python
    [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "AI response"}
    ]
    ```

    ## Rate Limiting

    Implements exponential backoff retry logic:
    - Detects HTTP 429 rate limit errors
    - Retries with increasing delays: 10s, 20s, 40s, 60s (capped)
    - Configurable via environment variables
    - Raises exception after max retries exceeded

    ## Requirements

    - `openai` package installed (`pip install openai`)
    - Valid API key in GEMINI_API_KEY (or OPENAI_API_KEY for OpenAI)
    - Internet connection for API access
    - Sufficient API credits/quota

    ## Error Handling

    - Rate limits: Automatic retry with exponential backoff
    - Invalid API key: Raises ValueError
    - Missing package: Raises ImportError (if openai package not installed)
    - Invalid message format: Validates list of dicts (not single string)
    - Network errors: Propagates exception after retries exhausted

    ## Implementation Notes

    - Uses OpenAI SDK's chat.completions.create() interface
    - Tool schemas use strict=True for validation
    - Function arguments are JSON-parsed from response
    - Multiple tool calls in single response are supported
    - Inherits core functionality from RequestLoop base class
    - Default configuration points to Gemini's OpenAI-compatible endpoint

    ## Related Classes

    - `OpenAIAPI`: Low-level API wrapper for OpenAI requests
    - `RequestLoop`: Base class providing request/response loop logic
    - `ToolCollection`: Manages available tools for function calling
    - `ClaudeRequestLoop`: Alternative provider with native SDK
    - `GeminiRequestLoop`: Native Gemini API (non-OpenAI compatible)
    """

    @classmethod
    def api(cls) -> Any:
        return OpenAIAPI

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
