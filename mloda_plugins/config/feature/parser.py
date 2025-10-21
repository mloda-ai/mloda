"""
Format parsers for feature configuration files.

This module provides parsers for configuration files in JSON format.
"""

import json
from typing import List
from mloda_plugins.config.feature.models import FeatureConfig, FeatureConfigItem


def parse_json(config_str: str) -> List[FeatureConfigItem]:
    """Parse a JSON configuration string into feature config items.

    Args:
        config_str: JSON string containing feature configuration

    Returns:
        List of feature configuration items (strings or FeatureConfig objects)
    """
    data = json.loads(config_str)

    if not isinstance(data, list):
        raise ValueError("Configuration must be a JSON array")

    result: List[FeatureConfigItem] = []
    for item in data:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            result.append(FeatureConfig(**item))
        else:
            raise ValueError(f"Invalid configuration item: {item}")

    return result
