"""
Configuration loader for converting parsed config to Feature objects.

This module handles the conversion from validated configuration data
to mloda Feature instances.
"""

from typing import Any

# Import from the component modules, NOT the `mloda.user` facade: that facade
# imports load_features_from_config from this module, so a facade import here
# creates a circular import (guarded by test_import_isolation.py).
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.api.feature_config.parser import parse_json
from mloda.core.api.feature_config.models import FeatureConfig, validate_feature_group_scope
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


def process_nested_features(options: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert nested in_features dicts to Feature objects.

    Args:
        options: Dictionary of options that may contain nested feature definitions

    Returns:
        Dictionary with nested dicts converted to Feature objects
    """
    processed: dict[str, Any] = {}
    for key, value in options.items():
        if key == "in_features" and isinstance(value, dict):
            # This is a nested feature definition - convert it to a Feature object
            feature_name = value.get("name")
            if not feature_name:
                raise ValueError(f"Nested in_features must have a 'name' field: {value}")

            # Recursively process nested options
            nested_options = value.get("options", {})
            processed_nested_options = process_nested_features(nested_options)

            # Handle nested in_features (can also be a dict)
            in_features = value.get("in_features")
            if in_features:
                if isinstance(in_features, list):
                    # For list, convert each to string (single sources) or keep as-is
                    processed_nested_options["in_features"] = in_features if len(in_features) > 1 else in_features[0]
                elif isinstance(in_features, dict):
                    # Recursively create Feature for in_features
                    in_features = process_nested_features({"in_features": in_features})["in_features"]
                    processed_nested_options["in_features"] = in_features
                else:
                    processed_nested_options["in_features"] = in_features

            # Create the Feature object; the nested dict carries its own resolution scope,
            # validated in the config layer so it fails config-style, not Python-surface.
            nested_feature_group = value.get("feature_group")
            validate_feature_group_scope(nested_feature_group)
            processed[key] = Feature(
                name=feature_name,
                options=processed_nested_options,
                feature_group=nested_feature_group,
            )
        elif isinstance(value, dict):
            # Recursively process nested dicts
            processed[key] = process_nested_features(value)
        else:
            processed[key] = value

    return processed


def load_features_from_config(config_str: str, format: str = "json") -> list[Feature | str]:
    """Load features from a configuration string.

    Args:
        config_str: Configuration string in the specified format
        format: Configuration format (currently only "json" is supported)

    Returns:
        List of Feature objects and/or feature name strings
    """
    if format != "json":
        raise ValueError(f"Unsupported format: '{format}'. Only 'json' is currently supported.")

    config_items = parse_json(config_str)

    features: list[Feature | str] = []

    for item in config_items:
        if isinstance(item, str):
            features.append(item)
        elif isinstance(item, FeatureConfig):
            # Build feature name with column index suffix if present
            feature_name = item.name
            if item.column_index is not None:
                feature_name = f"{item.name}~{item.column_index}"

            # Check if group_options or context_options exist
            if item.group_options is not None or item.context_options is not None:
                # Use new Options architecture with group/context separation
                context = item.context_options or {}
                # Handle in_features if present
                if item.in_features:
                    # Always convert to frozenset for consistency
                    context[DefaultOptionKeys.in_features] = frozenset(item.in_features)
                options = Options(
                    group=item.group_options or {},
                    context=context,
                    propagate_context_keys=frozenset(item.propagate_context_keys)
                    if item.propagate_context_keys
                    else None,
                )
                feature = Feature(name=feature_name, options=options, feature_group=item.feature_group)
                features.append(feature)
            # Check if in_features exists and create Options accordingly
            elif item.in_features:
                # Process nested features in options before creating Feature
                processed_options = process_nested_features(item.options)
                # Always convert to frozenset for consistency (even single items)
                source_value = frozenset(item.in_features)
                options = Options(group=processed_options, context={DefaultOptionKeys.in_features: source_value})
                feature = Feature(name=feature_name, options=options, feature_group=item.feature_group)
                features.append(feature)
            else:
                # Process nested features in options before creating Feature
                processed_options = process_nested_features(item.options)
                feature = Feature(name=feature_name, options=processed_options, feature_group=item.feature_group)
                features.append(feature)
        else:
            raise ValueError(f"Unexpected config item type: {type(item)}")

    return features
