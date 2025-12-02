"""
Configuration loader for converting parsed config to Feature objects.

This module handles the conversion from validated configuration data
to mloda Feature instances.
"""

from typing import List, Union, Dict, Any
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.config.feature.parser import parse_json
from mloda_plugins.config.feature.models import FeatureConfig
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


def process_nested_features(options: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert nested in_features dicts to Feature objects.

    Args:
        options: Dictionary of options that may contain nested feature definitions

    Returns:
        Dictionary with nested dicts converted to Feature objects
    """
    processed: Dict[str, Any] = {}
    for key, value in options.items():
        if key == "in_features" and isinstance(value, dict):
            # This is a nested feature definition - convert it to a Feature object
            feature_name = value.get("name")
            if not feature_name:
                raise ValueError(f"Nested in_features must have a 'name' field: {value}")

            # Recursively process nested options
            nested_options = value.get("options", {})
            processed_nested_options = process_nested_features(nested_options)

            # Handle nested mloda_sources (can also be a dict)
            mloda_sources = value.get("mloda_sources")
            if mloda_sources:
                if isinstance(mloda_sources, list):
                    # For list, convert each to string (single sources) or keep as-is
                    processed_nested_options["in_features"] = (
                        mloda_sources if len(mloda_sources) > 1 else mloda_sources[0]
                    )
                elif isinstance(mloda_sources, dict):
                    # Recursively create Feature for mloda_sources
                    in_features = process_nested_features({"in_features": mloda_sources})["in_features"]
                    processed_nested_options["in_features"] = in_features
                else:
                    processed_nested_options["in_features"] = mloda_sources

            # Create the Feature object
            processed[key] = Feature(name=feature_name, options=processed_nested_options)
        elif isinstance(value, dict):
            # Recursively process nested dicts
            processed[key] = process_nested_features(value)
        else:
            processed[key] = value

    return processed


def load_features_from_config(config_str: str, format: str = "json") -> List[Union[Feature, str]]:
    """Load features from a configuration string.

    Uses a two-pass strategy to support feature references:
    - Pass 1: Create all Feature objects and build a name registry
    - Pass 2: Resolve @feature_name references to actual Feature objects

    Args:
        config_str: Configuration string in the specified format
        format: Configuration format (currently only "json" is supported)

    Returns:
        List of Feature objects and/or feature name strings
    """
    if format != "json":
        raise ValueError(f"Unsupported format: {format}")

    config_items = parse_json(config_str)

    # Pass 1: Create all Feature objects and build registry
    features: List[Union[Feature, str]] = []
    feature_registry: Dict[str, Feature] = {}

    for item in config_items:
        if isinstance(item, str):
            # Create a Feature object for string entries so they can be referenced
            feature = Feature(name=item, options={})
            features.append(item)
            feature_registry[item] = feature
        elif isinstance(item, FeatureConfig):
            # Build feature name with column index suffix if present
            feature_name = item.name
            if item.column_index is not None:
                feature_name = f"{item.name}~{item.column_index}"

            # Check if group_options or context_options exist
            if item.group_options is not None or item.context_options is not None:
                # Use new Options architecture with group/context separation
                context = item.context_options or {}
                # Handle mloda_sources if present
                if item.mloda_sources:
                    # Always convert to frozenset for consistency
                    context[DefaultOptionKeys.in_features] = frozenset(item.mloda_sources)
                options = Options(group=item.group_options or {}, context=context)
                feature = Feature(name=feature_name, options=options)
                features.append(feature)
                feature_registry[feature_name] = feature
            # Check if mloda_sources exists and create Options accordingly
            elif item.mloda_sources:
                # Process nested features in options before creating Feature
                processed_options = process_nested_features(item.options)
                # Always convert to frozenset for consistency (even single items)
                source_value = frozenset(item.mloda_sources)
                options = Options(group=processed_options, context={DefaultOptionKeys.in_features: source_value})
                feature = Feature(name=feature_name, options=options)
                features.append(feature)
                feature_registry[feature_name] = feature
            else:
                # Process nested features in options before creating Feature
                processed_options = process_nested_features(item.options)
                feature = Feature(name=feature_name, options=processed_options)
                features.append(feature)
                feature_registry[feature_name] = feature
        else:
            raise ValueError(f"Unexpected config item type: {type(item)}")

    # Pass 2: Resolve @feature_name references to Feature objects
    for feat in features:
        if isinstance(feat, Feature):
            mloda_source = feat.options.context.get(DefaultOptionKeys.in_features)
            if mloda_source:
                # Handle both single string and frozenset of strings
                if isinstance(mloda_source, str) and mloda_source.startswith("@"):
                    # Single reference string
                    referenced_name = mloda_source[1:]
                    if referenced_name in feature_registry:
                        feat.options.context[DefaultOptionKeys.in_features] = feature_registry[referenced_name]
                    else:
                        raise ValueError(f"Feature reference '@{referenced_name}' not found in configuration")
                elif isinstance(mloda_source, frozenset):
                    # Frozenset of sources - resolve any @ references
                    resolved_sources = []
                    for source in mloda_source:
                        if isinstance(source, str) and source.startswith("@"):
                            referenced_name = source[1:]
                            if referenced_name in feature_registry:
                                resolved_sources.append(feature_registry[referenced_name])
                            else:
                                raise ValueError(f"Feature reference '@{referenced_name}' not found in configuration")
                        else:
                            resolved_sources.append(source)
                    # Only replace if we actually resolved any references
                    if any(isinstance(s, str) and s.startswith("@") for s in mloda_source):
                        feat.options.context[DefaultOptionKeys.in_features] = frozenset(resolved_sources)

    return features
