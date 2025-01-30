import os
from typing import Any, Dict, Set


from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.llm.llm_base_request import LLMBaseRequest
from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeatureComposite

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

import logging

logger = logging.getLogger(__name__)


class GeminiRequest(LLMBaseRequest):
    def input_features(self, options: Options, feature_name: FeatureName) -> Set[Feature] | None:
        return SourceInputFeatureComposite.input_features(options, feature_name)

    @classmethod
    def request(cls, model: str, prompt: str, model_parameters: Dict[str, Any]) -> Any:
        try:
            gemini_model = cls._setup_model_if_needed(model)
            if gemini_model is not None:
                result = gemini_model.generate_content(prompt, generation_config=model_parameters)
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
