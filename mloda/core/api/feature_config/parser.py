"""
Format parsers for feature configuration files.

This module provides parsers for configuration files in JSON format.
"""

import json
from mloda.core.api.feature_config.models import (
    FeatureConfig,
    FeatureConfigItem,
    describe_offender,
    error_prefix,
    feature_config_error_prefix,
    validate_top_level_in_features,
)


def parse_json(config_str: str) -> list[FeatureConfigItem]:
    """Parse a JSON configuration string into feature config items.

    Args:
        config_str: JSON string containing feature configuration

    Returns:
        List of feature configuration items (strings or FeatureConfig objects)
    """
    data = json.loads(config_str)

    if not isinstance(data, list):
        raise ValueError("Configuration must be a JSON array")

    result: list[FeatureConfigItem] = []
    for position, item in enumerate(data, start=1):
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(
                    f"{feature_config_error_prefix(position, None)}: a feature name string must be non-empty, "
                    f"got {describe_offender(item)}"
                )
            result.append(item)
        elif isinstance(item, dict):
            with error_prefix(feature_config_error_prefix(position, item.get("name"))):
                config = FeatureConfig(**item)
                validate_top_level_in_features(config.in_features)
            result.append(config)
        else:
            raise ValueError(
                f"{feature_config_error_prefix(position, None)}: must be a feature name string or an object, "
                f"got {type(item).__name__}"
            )

    return result
