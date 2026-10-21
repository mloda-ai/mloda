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


def _feature_group_checks(spec: PropertySpec) -> tuple[tuple[bool, str], ...]:
    """Fields that do not apply to a FeatureGroup spec."""
    return (
        (
            spec.framework_set,
            "declares framework_set=True, which marks a framework-written key; PROPERTY_MAPPING keys on a "
            "FeatureGroup are user-set.",
        ),
        (
            spec.scalar_only,
            "declares scalar_only=True, which rejects collection values; PROPERTY_MAPPING keys on a "
            "FeatureGroup always unpack element-wise.",
        ),
    )


def validate_property_spec(
    owner: str, key: str, spec: Any, surface: DeclarationSurface, via: str | None = None
) -> PropertySpec:
    """The per-key rules for one surface; via appends the reached-through suffix to any raised message."""
    suffix = "" if via is None else f" (reached defining {via})"
    prefix = f"{owner}.{surface.attr}['{key}']"
    if not isinstance(spec, PropertySpec):
        raise ValueError(
            f"{prefix} is a {type(spec).__name__}, not a PropertySpec. "
            f"Construct PropertySpec(...) or use the property_spec(...) helper.{suffix}"
        )

    for triggered, message in _feature_group_checks(spec):
        if triggered:
            raise ValueError(f"{prefix} {message}{suffix}")

    return spec
