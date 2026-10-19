"""
Configuration loader for converting parsed config to Feature objects.

This module handles the conversion from validated configuration data
to mloda Feature instances.
"""

import json
from typing import Any

# Import from the component modules, NOT the `mloda.user` facade: that facade
# imports load_features_from_config from this module, so a facade import here
# creates a circular import (guarded by test_import_isolation.py).
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.api.feature_config.parser import parse_json
from mloda.core.api.feature_config.models import (
    FeatureConfig,
    parse_nested_feature_config,
    validate_nested_in_features,
)
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


def build_nested_feature(value: dict[str, Any]) -> Feature:
    """Build a Feature from a nested in_features dict, validated through FeatureConfig.

    Args:
        value: Nested feature definition under an "in_features" key

    Returns:
        Feature object for the nested definition
    """
    config = parse_nested_feature_config(value)
    validate_nested_in_features(config.in_features)

    processed_nested_options = process_nested_features(config.options)

    # Sibling in_features: a list of more than one source name stays a list, a 1-element list collapses to its
    # element, a dict recurses into another nested Feature, a string is a single source name stored as-is.
    in_features = config.in_features
    if in_features:
        if isinstance(in_features, list):
            processed_nested_options["in_features"] = in_features if len(in_features) > 1 else in_features[0]
        elif isinstance(in_features, dict):
            processed_nested_options["in_features"] = build_nested_feature(in_features)
        else:
            processed_nested_options["in_features"] = in_features

    return Feature(name=config.name, options=processed_nested_options, feature_group=config.feature_group)


def process_nested_features(options: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert nested in_features dicts to Feature objects.

    Args:
        options: Dictionary of options that may contain nested feature definitions

    Returns:
        Dictionary with nested dicts converted to Feature objects
    """
    processed: dict[str, Any] = {}
    for key, value in options.items():
        # str() on both sides: the enum value is the option key, so comparing
        # through it rather than a literal keeps this branch matching if that
        # value is renamed. Not `key == DefaultOptionKeys.in_features`, which
        # holds only while DefaultOptionKeys mixes in str (see #1063).
        if str(key) == str(DefaultOptionKeys.in_features) and isinstance(value, dict):
            processed[key] = build_nested_feature(value)
        elif isinstance(value, dict):
            # Recursively process nested dicts
            processed[key] = process_nested_features(value)
        else:
            processed[key] = value

    return processed


def validate_no_in_features_collision(container: dict[str, Any] | None, container_name: str) -> None:
    """The top-level in_features field is written into context, so an in_features container key would collide."""
    if container and DefaultOptionKeys.in_features in container:
        raise ValueError(
            f"'in_features' cannot be used both as a top-level feature config field and as a key inside "
            f"'{container_name}'; declare the source features in one place."
        )


def validate_single_in_features_container(
    group_options: dict[str, Any] | None, context_options: dict[str, Any] | None
) -> None:
    """Group and context share one key space, so the same in_features key in both would collide inside Options."""
    if (
        group_options
        and context_options
        and DefaultOptionKeys.in_features in group_options
        and DefaultOptionKeys.in_features in context_options
    ):
        raise ValueError(
            "'in_features' cannot be a key of both 'group_options' and 'context_options'; "
            "declare the source features in one place."
        )


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

            # One in_features declaration per feature: the top-level field is written into context, so an
            # in_features key in any container collides, and so does the same key in both modern containers.
            if item.in_features:
                validate_no_in_features_collision(item.options, "options")
                validate_no_in_features_collision(item.group_options, "group_options")
                validate_no_in_features_collision(item.context_options, "context_options")
            validate_single_in_features_container(item.group_options, item.context_options)

            # Check if group_options or context_options exist
            if item.group_options is not None or item.context_options is not None:
                # Use new Options architecture with group/context separation
                # Nested features are processed in both containers, as they are under the legacy 'options' key.
                group = process_nested_features(item.group_options or {})
                context = process_nested_features(item.context_options or {})
                if item.in_features:
                    context[DefaultOptionKeys.in_features] = tuple(item.in_features)
                options = Options(
                    group=group,
                    context=context,
                    propagate_context_keys=frozenset(item.propagate_context_keys)
                    if item.propagate_context_keys
                    else None,
                    cascade_context_keys=frozenset(item.cascade_context_keys) if item.cascade_context_keys else None,
                )
                feature = Feature(name=feature_name, options=options, feature_group=item.feature_group)
                features.append(feature)
            # Check if in_features exists and create Options accordingly
            elif item.in_features:
                # Process nested features in options before creating Feature
                processed_options = process_nested_features(item.options)
                source_value = tuple(item.in_features)
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


def load_run_definition(config_str: str, format: str = "json") -> tuple[list[Feature | str], list[Any]]:
    """Load a run definition (features plus links) from a configuration string.

    The top level is an object with a ``format`` version, a ``features`` array
    (same item shape as ``load_features_from_config``) and an optional ``links``
    array. A bare array is still accepted and yields no links.

    Args:
        config_str: Configuration string in the specified format
        format: Configuration format (currently only "json" is supported)

    Returns:
        Tuple of (features, links). Links are returned as raw dicts; building
        ``Link`` objects needs feature group classes and is left to the caller.
    """
    if format != "json":
        raise ValueError(f"Unsupported format: '{format}'. Only 'json' is currently supported.")

    data = json.loads(config_str)
    if isinstance(data, list):
        return load_features_from_config(config_str, format=format), []
    if not isinstance(data, dict):
        raise ValueError("Run definition must be a JSON object or array")

    version = data.get("format", 1)
    if version != 1:
        raise ValueError(f"Unsupported run definition format version: {version!r}")

    features_data = data.get("features", [])
    if not isinstance(features_data, list):
        raise ValueError("'features' must be a JSON array")
    features = load_features_from_config(json.dumps(features_data), format=format)

    links = data.get("links", [])
    if not isinstance(links, list):
        raise ValueError("'links' must be a JSON array")
    return features, links


def _run_definition_item(feature: Feature | str) -> Any:
    """One run definition item: a bare string for a feature name, an object otherwise."""
    if isinstance(feature, str):
        return feature

    item: dict[str, Any] = {"name": str(feature.name)}
    options = feature.options
    in_features_key = str(DefaultOptionKeys.in_features)
    in_features = next((value for key, value in options.context.items() if str(key) == in_features_key), None)
    context = {str(key): value for key, value in options.context.items() if str(key) != in_features_key}
    if in_features is not None:
        # The loader keeps the sources as a tuple in context; the file format writes an array.
        item["in_features"] = list(in_features) if isinstance(in_features, (list, tuple)) else in_features
    if context:
        item["context_options"] = context
    if options.group:
        item["group_options"] = {str(key): value for key, value in options.group.items()}
    if options.propagate_context_keys:
        item["propagate_context_keys"] = sorted(options.propagate_context_keys)
    if options.cascade_context_keys:
        item["cascade_context_keys"] = sorted(options.cascade_context_keys)
    scope = feature.feature_group_scope
    if scope is not None:
        # A class object cannot live in JSON; the class name is the config form the loader accepts.
        item["feature_group"] = scope if isinstance(scope, str) else scope.__name__
    return item


def dump_run_definition(features: list[Feature | str], links: list[Any] | None = None) -> str:
    """Dump features and links to a run definition JSON string.

    Feature names are written as bare strings; features with options are
    written as objects carrying ``name`` plus the config fields the loader
    accepts (``in_features``, ``group_options``, ``context_options``,
    ``propagate_context_keys``, ``cascade_context_keys``, ``feature_group``),
    so a load of the dump rebuilds the same feature. Links are expected to
    already be JSON-safe dicts.

    Args:
        features: Feature objects and/or feature name strings
        links: Optional list of JSON-safe link dicts

    Returns:
        JSON string in the run definition format

    Raises:
        ValueError: If an option value cannot be represented in JSON.
    """
    items = [_run_definition_item(feature) for feature in features]
    try:
        return json.dumps({"format": 1, "features": items, "links": links or []}, indent=2)
    except TypeError as error:
        # Loud, not silent: a skipped option would dump a run that loads back different.
        raise ValueError(
            f"Run definition is not JSON-serialisable: {error}. "
            "Options must be JSON-safe values; tuples are written as arrays, class objects cannot be written."
        ) from error
