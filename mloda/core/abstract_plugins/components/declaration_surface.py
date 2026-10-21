"""The per-key validator behind ``FeatureGroup.PROPERTY_MAPPING``."""

from __future__ import annotations

from enum import Enum
from typing import Any

from mloda.core.abstract_plugins.components.property_spec import PropertySpec


class DeclarationSurface(Enum):
    """The class-body attribute whose per-key specs are validated: which name."""

    FEATURE_GROUP = "PROPERTY_MAPPING"

    @property
    def attr(self) -> str:
        return self.value


def validate_property_spec(owner: str, key: str, spec: Any, surface: DeclarationSurface) -> PropertySpec:
    """Reject a non-PropertySpec value for one key."""
    if not isinstance(spec, PropertySpec):
        raise ValueError(
            f"{owner}.{surface.attr}['{key}'] is a {type(spec).__name__}, not a PropertySpec. "
            f"Construct PropertySpec(...) or use the property_spec(...) helper."
        )
    return spec
