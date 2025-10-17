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
    genai, FunctionCall, Content, Part, functionDeclarations, google = None, None, None, None, None, None

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
    def _setup_model_if_needed(cls, model: str) -> "genai.GenerativeModel":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        if genai is None:
            raise ImportError("Please install google.generativeai to use this feature.")

        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model)

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
    """
    Base class for integrating Google Gemini LLM API into mloda feature pipelines.

    This feature group provides a bridge between mloda's feature engineering framework
    and Google's Gemini generative AI models. It handles request formatting, response
    parsing, tool calling, rate limiting, and error handling for Gemini API interactions.

    ## Key Capabilities

    - Send prompts to Gemini models and receive generated text
    - Support for multiple Gemini model versions (e.g., gemini-2.0-flash-exp)
    - Automatic retry logic with exponential backoff for rate limits
    - Tool/function calling support for interactive workflows
    - Integration with mloda's ToolCollection framework
    - Response parsing and validation

    ## Common Use Cases

    - Natural language processing and text generation
    - Code analysis and generation
    - File content analysis for intelligent selection
    - Multi-turn conversations with context preservation
    - Tool-augmented AI workflows (function calling)
    - Automated documentation generation

    ## Usage Examples

    ### Basic Text Generation

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature
    from mloda_core.abstract_plugins.components.options import Options

    feature = Feature(
        name="GeminiRequestLoop",
        options=Options(
            context={
                "model": "gemini-2.0-flash-exp",
                "prompt": "Explain what this code does",
                "mloda_source_feature": frozenset(["CodeContentFeature"]),
            }
        )
    )
    ```

    ### File Analysis with LLM

    ```python
    from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
    from mloda_plugins.feature_group.input_data.read_context_files import (
        ConcatenatedFileContent
    )

    feature = Feature(
        name="GeminiRequestLoop",
        options=Options(
            context={
                "model": "gemini-2.0-flash-exp",
                "prompt": "List the file paths that contain authentication logic",
                DefaultOptionKeys.mloda_source_feature: frozenset([
                    ConcatenatedFileContent.get_class_name()
                ]),
                "target_folder": frozenset(["/workspace/src"]),
                "file_type": "py",
            }
        )
    )
    ```

    ### Tool-Augmented Workflow

    ```python
    from mloda_plugins.feature_group.experimental.llm.tools.tool_collection import (
        ToolCollection
    )

    # Assuming tools are configured
    feature = Feature(
        name="GeminiRequestLoop",
        options=Options(
            context={
                "model": "gemini-2.0-flash-exp",
                "prompt": "Read file.py and suggest improvements",
                "tools": tool_collection,  # ToolCollection instance
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)

    - `model` (required): Gemini model name (e.g., "gemini-2.0-flash-exp")
    - `prompt` (required): Single string prompt for the LLM
    - `mloda_source_feature`: Frozenset of upstream feature dependencies
    - `model_parameters`: Dict of generation config (temperature, top_p, etc.)
    - `tools`: ToolCollection instance for function calling
    - Additional parameters passed to upstream features (target_folder, file_type, etc.)

    ### Group Parameters

    Currently none for GeminiRequestLoop.

    ## Configuration

    ### Environment Variables

    - `GEMINI_API_KEY` (required): Google API key for Gemini access
    - `GEMINI_MAX_RETRIES`: Maximum retry attempts (default: 5)
    - `GEMINI_INITIAL_RETRY_DELAY`: Initial retry delay in seconds (default: 10)
    - `GEMINI_MAX_RETRY_DELAY`: Maximum retry delay in seconds (default: 60)

    ## Output Format

    Returns generated text from Gemini in a DataFrame column named `GeminiRequestLoop`.
    If tools are used, returns the tool execution results.

    ## Rate Limiting

    Implements exponential backoff retry logic:
    - Detects HTTP 429 rate limit errors
    - Retries with increasing delays: 10s, 20s, 40s, 60s (capped)
    - Configurable via environment variables
    - Raises exception after max retries exceeded

    ## Requirements

    - `google.generativeai` package installed (`pip install google-generativeai`)
    - Valid GEMINI_API_KEY environment variable
    - Internet connection for API access
    - Sufficient API quota/credits

    ## Error Handling

    - Rate limits: Automatic retry with exponential backoff
    - Invalid API key: Raises ValueError
    - Missing package: Raises ImportError
    - Network errors: Propagates exception after retries exhausted
    - Invalid prompts: Validates single string prompt (no list support)

    ## Implementation Notes

    - Gemini only supports single string prompts (not message arrays)
    - Tool functions are converted from ToolFunctionDeclaration format
    - Response parsing handles both text and function call responses
    - Inherits core functionality from RequestLoop base class

    ## Related Classes

    - `GeminiAPI`: Low-level API wrapper for Gemini requests
    - `RequestLoop`: Base class providing request/response loop logic
    - `ToolCollection`: Manages available tools for function calling
    - `LLMFileSelector`: Example feature using GeminiRequestLoop
    """

    @classmethod
    def api(cls) -> Any:
        return GeminiAPI
