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
    """
    Base class for integrating Anthropic Claude LLM API into mloda feature pipelines.

    This feature group provides integration with Anthropic's Claude models, handling
    message formatting, response parsing, tool calling, rate limiting, and multi-turn
    conversation management for Claude API interactions.

    ## Key Capabilities

    - Multi-turn conversation support with message history
    - Claude-specific message format handling (role-based)
    - Tool/function calling with proper tool result formatting
    - Automatic retry logic with exponential backoff for rate limits
    - Support for all Claude model versions (e.g., claude-3-opus, claude-3-sonnet)
    - Integration with mloda's ToolCollection framework

    ## Common Use Cases

    - Complex reasoning and analysis tasks
    - Multi-step problem solving with tools
    - Code review and refactoring suggestions
    - Extended conversations with context preservation
    - Document analysis and summarization
    - Interactive AI workflows with tool augmentation

    ## Usage Examples

    ### Basic Text Generation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="ClaudeRequestLoop",
        options=Options(
            context={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "user", "content": "Explain how this function works"}
                ],
            }
        )
    )
    ```

    ### Multi-Turn Conversation

    ```python
    messages = [
        {"role": "user", "content": "What is feature engineering?"},
        {"role": "assistant", "content": "Feature engineering is..."},
        {"role": "user", "content": "Give me an example with pandas"},
    ]

    feature = Feature(
        name="ClaudeRequestLoop",
        options=Options(
            context={
                "model": "claude-3-5-sonnet-20241022",
                "messages": messages,
            }
        )
    )
    ```

    ### Tool-Augmented Workflow

    ```python
    from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import (
        ToolCollection
    )

    feature = Feature(
        name="ClaudeRequestLoop",
        options=Options(
            context={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "user", "content": "Read config.py and check for issues"}
                ],
                "tools": tool_collection,
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)

    - `model` (required): Claude model name (e.g., "claude-3-5-sonnet-20241022")
    - `messages` (required): List of message dicts with "role" and "content" keys
    - `tools`: ToolCollection instance for function calling
    - `max_tokens`: Maximum tokens in response (default: 1000)
    - Additional parameters for multi-turn conversations

    ### Group Parameters

    Currently none for ClaudeRequestLoop.

    ## Configuration

    ### Environment Variables

    - `CLAUDE_API_KEY` (required): Anthropic API key for Claude access
    - `CLAUDE_MAX_RETRIES`: Maximum retry attempts (default: 5)
    - `CLAUDE_INITIAL_RETRY_DELAY`: Initial retry delay in seconds (default: 10)
    - `CLAUDE_MAX_RETRY_DELAY`: Maximum retry delay in seconds (default: 60)

    ## Output Format

    Returns generated text from Claude in a DataFrame column named `ClaudeRequestLoop`.
    For tool calls, returns the tool execution results.

    ## Message Format

    Claude requires a list of message dictionaries:
    ```python
    [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "AI response"},
        {"role": "user", "content": "Follow-up message"}
    ]
    ```

    ## Rate Limiting

    Implements exponential backoff retry logic:
    - Detects HTTP 429 rate limit errors
    - Retries with increasing delays: 10s, 20s, 40s, 60s (capped)
    - Configurable via environment variables
    - Raises exception after max retries exceeded

    ## Requirements

    - `anthropic` package installed (`pip install anthropic`)
    - Valid CLAUDE_API_KEY environment variable
    - Internet connection for API access
    - Sufficient API credits/quota

    ## Error Handling

    - Rate limits: Automatic retry with exponential backoff
    - Invalid API key: Raises ValueError
    - Missing package: Raises ImportError
    - Invalid message format: Validates list of dicts (not single string)
    - Network errors: Propagates exception after retries exhausted

    ## Implementation Notes

    - Claude requires message arrays (not single string prompts like Gemini)
    - Tool results are added to messages as user role content
    - Assistant responses are tracked separately
    - Response parsing handles ToolUseBlock objects from Anthropic SDK
    - Inherits core functionality from RequestLoop base class

    ## Related Classes

    - `ClaudeAPI`: Low-level API wrapper for Claude requests
    - `RequestLoop`: Base class providing request/response loop logic
    - `ToolCollection`: Manages available tools for function calling
    - `GeminiRequestLoop`: Alternative LLM provider with different message format
    """

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
