"""Shared mechanics behind ``FeatureGroup.PROPERTY_MAPPING`` and ``BaseInputData.READER_OPTIONS``:
one per-key validator and one merge helper (the reader merges, the feature group reads its
attribute); two names, one spec type.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from mloda.core.abstract_plugins.components.property_spec import PropertySpec, is_no_default


class DeclarationSurface(Enum):
    """The two class-body attributes that share the merge mechanics: which name, which cache."""

    FEATURE_GROUP = ("PROPERTY_MAPPING", "_property_mapping_cache")
    READER = ("READER_OPTIONS", "_reader_option_specs_cache")

    def __init__(self, attr: str, cache_attr: str) -> None:
        self._surface_attr = attr
        self._surface_cache_attr = cache_attr

    @property
    def attr(self) -> str:
        return self._surface_attr

    @property
    def cache_attr(self) -> str:
        return self._surface_cache_attr


def own_declaration(klass: type, surface: DeclarationSurface) -> dict[str, Any]:
    """The class's own declaration on this surface, rejected loudly when it is None or not a dict."""
    declared = klass.__dict__.get(surface.attr, {})
    if declared is None:
        raise ValueError(
            f"{klass.__name__}.{surface.attr} is None. Declarations merge across the class "
            f"hierarchy, so None cannot clear inherited keys. Remove the assignment or declare a dict."
        )
    if not isinstance(declared, dict):
        raise ValueError(
            f"{klass.__name__}.{surface.attr} is a {type(declared).__name__}, not a dict. "
            f"Declare a dict mapping option keys to PropertySpec instances."
        )
    return declared


class _MergedDeclaration(dict[str, PropertySpec]):
    """The merge cache; a private type so reject_merge_cache_assignment can tell a framework-written
    cache from an authored one."""


def merged_declaration(cls: type, surface: DeclarationSurface) -> dict[str, PropertySpec]:
    """The MRO-merged declaration of cls on this surface, cached in cls's OWN __dict__; hands
    back the same object, not a copy. A cache present but not framework-written is poisoned and
    raises the same way reject_merge_cache_assignment would."""
    cached = cls.__dict__.get(surface.cache_attr)
    if isinstance(cached, _MergedDeclaration):
        return cached
    if cached is not None:
        reject_merge_cache_assignment(cls, surface)

    merged = _MergedDeclaration()
    for klass in reversed(cls.__mro__):
        merged.update(own_declaration(klass, surface))
    setattr(cls, surface.cache_attr, merged)
    return merged


def reject_merge_cache_assignment(cls: type, surface: DeclarationSurface) -> None:
    """The merge cache is framework-written; assigning it in a class body is rejected. A cache warmed
    by a cooperative __init_subclass__ hook before super() is a _MergedDeclaration and is not blamed."""
    if surface.cache_attr not in cls.__dict__:
        return
    if isinstance(cls.__dict__[surface.cache_attr], _MergedDeclaration):
        return
    raise ValueError(
        f"{cls.__name__} assigns {surface.cache_attr} in its class body; the merge cache is "
        f"framework-written and must never be declared."
    )


def _reader_inert_checks(spec: PropertySpec) -> tuple[tuple[bool, str], ...]:
    """Fields inert on a reader spec regardless of framework_set."""
    return (
        (
            spec.match_guard is not None,
            "declares a match_guard, which is name-matching machinery and silently inert on a reader.",
        ),
        (
            spec.deferred_binding,
            "declares deferred_binding=True, which is the name-capture exemption; a reader key has no name path.",
        ),
        (spec.context is False, "declares context=False, which places a materialized value; readers place none."),
    )


def _reader_framework_set_checks(spec: PropertySpec) -> tuple[tuple[bool, str], ...]:
    """Combinations that are inert once a reader spec declares framework_set=True."""
    return (
        (
            spec.strict_validation,
            "combines framework_set=True with strict_validation=True; the framework-written key is "
            "exempt from user-value validation, so strictness would be silently inert.",
        ),
        (
            spec.required_when is not None,
            "combines framework_set=True with a required_when predicate; the framework-written key is "
            "exempt from user-value validation, so the predicate would be silently inert.",
        ),
        (
            spec.allow_explicit_none,
            "combines framework_set=True with allow_explicit_none=True; the admit path skips the "
            "framework-written key, so the flag cannot affect reader selection.",
        ),
        (
            is_no_default(spec.default),
            "declares framework_set=True without a declared default; the framework-written key must "
            "declare its absent-state default explicitly, None included.",
        ),
    )


def _feature_group_checks(spec: PropertySpec) -> tuple[tuple[bool, str], ...]:
    """Fields that are reader-only and therefore inert on a FeatureGroup spec."""
    return (
        (
            spec.framework_set,
            "declares framework_set=True, which marks a reader-only field; PROPERTY_MAPPING keys on a "
            "FeatureGroup are user-set.",
        ),
        (
            spec.scalar_only,
            "declares scalar_only=True, which marks a reader-only field; PROPERTY_MAPPING keys on a "
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

    if surface is DeclarationSurface.FEATURE_GROUP:
        checks = _feature_group_checks(spec)
    elif surface is DeclarationSurface.READER:
        checks = _reader_inert_checks(spec)
    else:
        raise ValueError(f"Unknown DeclarationSurface {surface!r}.")
    for triggered, message in checks:
        if triggered:
            raise ValueError(f"{prefix} {message}{suffix}")

    if surface is DeclarationSurface.READER and spec.framework_set:
        for triggered, message in _reader_framework_set_checks(spec):
            if triggered:
                raise ValueError(f"{prefix} {message}{suffix}")

    return spec


def validate_declaration(cls: type, surface: DeclarationSurface, root: type | None) -> None:
    """Validate cls's own declaration on this surface, plus every plain mixin's reached through the
    MRO. A class whose own MRO already contains root is skipped; root=None walks every ancestor."""
    for key, spec in own_declaration(cls, surface).items():
        validate_property_spec(cls.__name__, key, spec, surface)

    for klass in cls.__mro__[1:]:
        # Real inheritance only: a class whose own MRO already contains root validated itself
        # at its own definition; an ABC.register virtual subclass never ran __init_subclass__ either way.
        if root is not None and root in klass.__mro__:
            continue
        for key, spec in own_declaration(klass, surface).items():
            validate_property_spec(klass.__name__, key, spec, surface, via=cls.__name__)
