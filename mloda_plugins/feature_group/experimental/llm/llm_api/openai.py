from copy import copy
from dataclasses import asdict
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union


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
    ) -> Tuple[str, Optional[str]]:
        response_text = ""
        tool_result = None

        if hasattr(response, "choices") and len(response.choices) > 0:  # OpenAI response structure
            message = response.choices[0].message
            if message.content:
                response_text = message.content

            if message.tool_calls:  # Check for function call
                if tools is None:
                    raise ValueError("Tools are not set.")

                new_tools = []
                for e in message.tool_calls:
                    tool_dict = {"name": e.function.name, "args": json.loads(e.function.arguments)}
                    new_tools.append(tool_dict)

                tool_result = cls._execute_tools(new_tools, features, tools)
                return "", tool_result

        elif hasattr(response, "text"):  # Fallback - might not be needed for chat models but for robustness
            response_text = response.text
        else:
            logger.warning(f"Response has no text or choices attribute: {response}")
            return str(response), None

        return response_text, None

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
    @classmethod
    def api(cls) -> Any:
        return OpenAIAPI

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        model, initial_prompt, model_parameters, tools = cls.read_properties(data, features)

        current_prompt = copy(initial_prompt)

        all_results = []
        messages: List[Dict[str, str]] = []  # Initialize messages for OpenAI

        all_tool_result = ""
        while True:
            print("\n############################################\n")
            api_instance = cls.api()

            if not messages:
                messages = [{"role": "user", "content": current_prompt}]

            response = api_instance.request(model, messages, model_parameters, tools)

            response_text, tool_result = cls.api().handle_response(response, features, tools)

            if response_text:
                response_text = response_text.strip()
            all_results.append(response_text)

            if not tool_result:
                break
            else:
                print(tool_result)

                messages = cls.loop_prompt(messages, tool_result, all_tool_result)

            all_tool_result += tool_result
        return pd.DataFrame({cls.get_class_name(): ["\n".join(all_results)]})

    @classmethod
    def loop_prompt(
        cls, prompt_or_messages: Any, tool_result: str, all_tool_result: str
    ) -> Any:  # Accepting either prompt string or messages list
        messages: List[Dict[str, str]] = prompt_or_messages  # Type hint for clarity
        messages.append(
            {"role": "assistant", "content": f"Tool Result: {tool_result}"}
        )  # Add tool result as assistant message
        messages.append(
            {
                "role": "user",
                "content": """
            **Instructions:**

            You are an expert reasoning agent. Given the information above (the original instructions, prior steps, and the most recent step result), carefully analyze the situation and determine the next action to take.

            *   If the goal is complete, respond with the `Final Answer: ` followed by the final answer.
            *   If another tool is needed, determine the correct tool to use, and what input it needs.
            """,
            }
        )  # Add user instruction for next turn, incorporating previous instructions
        return messages  # Return the updated messages list
