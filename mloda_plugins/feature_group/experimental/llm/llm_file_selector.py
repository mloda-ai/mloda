import logging
import os
from typing import Any, Optional, Set

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.llm.llm_api.gemini import GeminiRequestLoop
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

logger = logging.getLogger(__name__)


class LLMFileSelector(AbstractFeatureGroup):
    """
    A feature group that uses an LLM to select relevant files from a directory.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        prompt = options.get("prompt")
        if not prompt:
            raise ValueError("Prompt is required for LLMFileSelector")

        target_folder = options.get("target_folder")
        if not target_folder:
            raise ValueError("Target folder is required for LLMFileSelector")

        disallowed_files = list(options.get("disallowed_files")) if options.get("disallowed_files") else ["__init__.py"]

        file_type = options.get("file_type")

        llm_feature = Feature(
            name=GeminiRequestLoop.get_class_name(),
            options={
                "model": "gemini-2.0-flash-exp",  # Choose your desired model
                "prompt": prompt,
                DefaultOptionKeys.mloda_source_feature: frozenset([ConcatenatedFileContent.get_class_name()]),
                "target_folder": frozenset(target_folder),
                "disallowed_files": frozenset(disallowed_files),
                "file_type": file_type,
            },
        )
        return {llm_feature}

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        if GeminiRequestLoop.get_class_name() not in data.columns:
            raise ValueError(f"Feature {GeminiRequestLoop.get_class_name()} not found in input data.")

        for str_paths in data[GeminiRequestLoop.get_class_name()].values:
            paths = str_paths.split(",")
            for path in paths:
                if "\n" in path:
                    raise ValueError(f"File path {path} contains a newline character.")

                if not os.path.exists(path):
                    raise ValueError(f"File <{path}> does not exist.")

        return True

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data = data.rename(
            columns={GeminiRequestLoop.get_class_name(): LLMFileSelector.get_class_name()}, inplace=False
        )
        return data

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:
        for str_paths in data[cls.get_class_name()].values:
            paths = str_paths.split(",")
            for path in paths:
                if "\n" in path:
                    raise ValueError(f"File path {path} contains a newline character.")

                if not os.path.exists(path):
                    raise ValueError(f"File <{path}> does not exist.")
        return True
