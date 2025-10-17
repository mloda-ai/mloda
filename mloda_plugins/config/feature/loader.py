"""
Configuration loader for converting parsed config to Feature objects.

This module handles the conversion from validated configuration data
to mloda Feature instances.
"""

from typing import List, Union, Dict
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.config.feature.parser import parse_json
from mloda_plugins.config.feature.models import FeatureConfig
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


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
                    context[DefaultOptionKeys.mloda_source_feature] = frozenset(item.mloda_sources)
                options = Options(group=item.group_options or {}, context=context)
                feature = Feature(name=feature_name, options=options)
                features.append(feature)
                feature_registry[feature_name] = feature
            # Check if mloda_sources exists and create Options accordingly
            elif item.mloda_sources:
                # Convert mloda_sources list to frozenset
                options = Options(
                    group=item.options, context={DefaultOptionKeys.mloda_source_feature: frozenset(item.mloda_sources)}
                )
                feature = Feature(name=feature_name, options=options)
                features.append(feature)
                feature_registry[feature_name] = feature
            # Check if mloda_source exists and create Options accordingly
            elif item.mloda_source:
                # Temporarily store mloda_source as string (will resolve in pass 2)
                options = Options(
                    group=item.options, context={DefaultOptionKeys.mloda_source_feature: item.mloda_source}
                )
                feature = Feature(name=feature_name, options=options)
                features.append(feature)
                feature_registry[feature_name] = feature
            else:
                feature = Feature(name=feature_name, options=item.options)
                features.append(feature)
                feature_registry[feature_name] = feature
        else:
            raise ValueError(f"Unexpected config item type: {type(item)}")

    # Pass 2: Resolve @feature_name references to Feature objects
    for feat in features:
        if isinstance(feat, Feature):
            mloda_source = feat.options.context.get(DefaultOptionKeys.mloda_source_feature)
            if mloda_source and isinstance(mloda_source, str) and mloda_source.startswith("@"):
                # Extract the feature name (remove @ prefix)
                referenced_name = mloda_source[1:]
                # Look up the Feature object in the registry
                if referenced_name in feature_registry:
                    # Replace the string reference with the actual Feature object
                    feat.options.context[DefaultOptionKeys.mloda_source_feature] = feature_registry[referenced_name]
                else:
                    raise ValueError(f"Feature reference '@{referenced_name}' not found in configuration")

    return features
