import json
from typing import Any, Dict, Set, Type, Union


from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe


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


class LLMBaseRequest(AbstractFeatureGroup):
    model = "model"
    prompt = "prompt"
    temperature = "temperature"
    model_parameters = "model_parameters"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        model = cls.get_model_from_config(features)
        prompt = cls.handle_prompt(data, features)
        model_parameters = cls.get_model_parameters(features)

        response = cls.request(model, prompt, model_parameters)
        transformed_data = cls.handle_response(response, features)

        return transformed_data

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
    def request(cls, model: str, prompt: str, model_parameters: Dict[str, Any]) -> Any:
        raise NotImplementedError("Request method must be implemented in the child class.")

    @classmethod
    def handle_prompt(cls, data: Any, features: FeatureSet) -> str:
        data_prompt = "" if data is None or data.empty else str(data.iloc[0, 0])
        option_prompt = features.get_options_key(cls.prompt) or ""

        if not option_prompt and not data_prompt:
            raise ValueError(f"Prompt was not set for {cls.__name__}")

        return f"{data_prompt} {option_prompt}".strip()

    @classmethod
    def handle_response(cls, response: Any, features: FeatureSet) -> pd.DataFrame:
        feature_name = next(iter(features.get_all_names()))

        if hasattr(response, "text"):
            response_text = response.text
            try:
                json_response = json.loads(response_text)
                return pd.DataFrame({feature_name: [json_response]})
            except json.JSONDecodeError:
                return pd.DataFrame({feature_name: [response_text]})
        else:
            logger.warning(f"Response has no text attribute: {response}")
            return pd.DataFrame({feature_name: [response]})

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}
