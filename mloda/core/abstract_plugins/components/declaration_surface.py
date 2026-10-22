"""The per-key validator behind ``FeatureGroup.PROPERTY_MAPPING``."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.property_spec import PropertySpec


def validate_property_spec(owner: str, key: str, spec: Any) -> PropertySpec:
    """Reject a non-PropertySpec value for one key."""
    if not isinstance(spec, PropertySpec):
        raise ValueError(
            f"{owner}.PROPERTY_MAPPING['{key}'] is a {type(spec).__name__}, not a PropertySpec. "
            f"Construct PropertySpec(...) or use the property_spec(...) helper."
        )
    return spec
