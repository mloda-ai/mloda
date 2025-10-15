"""
Configuration loader for converting parsed config to Feature objects.

This module handles the conversion from validated configuration data
to mloda Feature instances.
"""

from typing import List, Union
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.config.feature.parser import parse_json
from mloda_plugins.config.feature.models import FeatureConfig


def load_features_from_config(config_str: str, format: str = "json") -> List[Union[Feature, str]]:
    """Load features from a configuration string.

    Args:
        config_str: Configuration string in the specified format
        format: Configuration format (currently only "json" is supported)

    Returns:
        List of Feature objects and/or feature name strings
    """
    if format != "json":
        raise ValueError(f"Unsupported format: {format}")

    config_items = parse_json(config_str)

    features: List[Union[Feature, str]] = []
    for item in config_items:
        if isinstance(item, str):
            features.append(item)
        elif isinstance(item, FeatureConfig):
            features.append(Feature(name=item.name, options=item.options))
        else:
            raise ValueError(f"Unexpected config item type: {type(item)}")

    return features
