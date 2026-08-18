"""The PROPERTY_MAPPING declaration surface shared by FeatureGroup and BaseInputData: MRO merge,
per-class cache, and author-time rules. One attribute, one spec type; DeclarationSurface only
changes which fields are inert.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, is_no_default


class DeclarationSurface(Enum):
    FEATURE_GROUP = "FeatureGroup"
    READER = "BaseInputData"


SURFACE_ATTR = "PROPERTY_MAPPING_SURFACE"
CACHE_ATTR = "_property_mapping_cache"
PROPERTY_MAPPING_ATTR = "PROPERTY_MAPPING"


def own_property_mapping(klass: type) -> dict[str, Any]:
    """The class's own PROPERTY_MAPPING, rejected loudly when it is None or not a dict."""
    declared = klass.__dict__.get(PROPERTY_MAPPING_ATTR, {})
    if declared is None:
        raise ValueError(
            f"{klass.__name__}.PROPERTY_MAPPING is None. Declarations merge across the class "
            f"hierarchy, so None cannot clear inherited keys. Remove the assignment or declare a dict."
        )
    if not isinstance(declared, dict):
        raise ValueError(
            f"{klass.__name__}.PROPERTY_MAPPING is a {type(declared).__name__}, not a dict. "
            f"Declare a dict mapping option keys to PropertySpec instances."
        )
    return declared


def surface_base(cls: type) -> type | None:
    """The first class in cls.__mro__ that declares the surface in its own __dict__."""
    for klass in cls.__mro__:
        if SURFACE_ATTR in klass.__dict__:
            return klass
    return None


class _MergedMapping(dict[str, PropertySpec]):
    """The framework merge cache: the merged mapping itself, plus whether some class beyond the
    surface base declared it, so a cooperative pre-super() read is never mistaken for an assignment."""

    declared_beyond_base: bool = False


def _merged_property_mapping(cls: type) -> _MergedMapping:
    """The MRO-merged declarations of cls, cached in cls's OWN __dict__; internal, never handed out."""
    cached: _MergedMapping | None = cls.__dict__.get(CACHE_ATTR)
    if cached is not None:
        return cached

    base = surface_base(cls)
    merged = _MergedMapping()
    for klass in reversed(cls.__mro__):
        merged.update(own_property_mapping(klass))
    merged.declared_beyond_base = any(
        klass is not base and PROPERTY_MAPPING_ATTR in klass.__dict__ for klass in cls.__mro__
    )
    setattr(cls, CACHE_ATTR, merged)
    return merged


def merged_property_mapping(cls: type) -> dict[str, PropertySpec]:
    """The MRO-merged declarations of cls, cached in cls's OWN __dict__; internal, never handed out."""
    return _merged_property_mapping(cls)


def declares_property_mapping(cls: type) -> bool:
    """True when some class OTHER THAN the surface base declares its own PROPERTY_MAPPING."""
    return _merged_property_mapping(cls).declared_beyond_base


def configuration_property_mapping(cls: type) -> dict[str, PropertySpec] | None:
    """The cached merged mapping itself when declared beyond the surface base, else None; no copy,
    since internal callers only ever read it."""
    merged = _merged_property_mapping(cls)
    return merged if merged.declared_beyond_base else None


def reject_merge_cache_assignment(cls: type) -> None:
    """The merge cache is framework-written; a class body assigning it is a mistake. A cache warmed
    by a cooperative __init_subclass__ hook before calling super() is a _MergedMapping and is not
    blamed: only a class body literally assigning the attribute produces something else."""
    if CACHE_ATTR not in cls.__dict__:
        return
    if isinstance(cls.__dict__[CACHE_ATTR], _MergedMapping):
        return
    raise ValueError(
        f"{cls.__name__} assigns {CACHE_ATTR} in its class body; the merge cache is "
        f"framework-written and must never be declared."
    )


def reject_surface_marker_assignment(cls: type) -> None:
    """The surface marker is framework-written; only FeatureGroup and BaseInputData set it."""
    if SURFACE_ATTR in cls.__dict__:
        raise ValueError(
            f"{cls.__name__} assigns {SURFACE_ATTR} in its class body; the marker is set only by "
            f"the framework's own base classes (FeatureGroup, BaseInputData)."
        )


def _reject_conflicting_surface_bases(cls: type) -> None:
    """Exactly one surface may govern cls; two distinct surface-marked classes in its MRO is an
    author mistake, whether that is FeatureGroup and BaseInputData both, or a plain mixin carrying
    the marker alongside either. Runs before any per-key validation, so a reader-only key merged
    from the wrong surface never produces a misleading error first."""
    bases = [klass for klass in cls.__mro__ if SURFACE_ATTR in klass.__dict__]
    if len(bases) <= 1:
        return
    names = " and ".join(base.__name__ for base in bases)
    raise ValueError(f"{cls.__name__} inherits both {names}; a class declares on one {SURFACE_ATTR} surface only.")


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
    if not isinstance(spec, PropertySpec):
        raise ValueError(
            f"{owner}.PROPERTY_MAPPING['{key}'] is a {type(spec).__name__}, not a PropertySpec. "
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
            raise ValueError(f"{owner}.PROPERTY_MAPPING['{key}'] {message}{suffix}")

    if surface is DeclarationSurface.READER and spec.framework_set:
        for triggered, message in _reader_framework_set_checks(spec):
            if triggered:
                raise ValueError(f"{owner}.PROPERTY_MAPPING['{key}'] {message}{suffix}")

    return spec


def validate_property_mapping(cls: type) -> None:
    """Validate cls's own declaration, plus every plain mixin's, against the surface cls declares."""
    _reject_conflicting_surface_bases(cls)
    base = surface_base(cls)
    if base is None:
        return
    surface: DeclarationSurface = base.__dict__[SURFACE_ATTR]

    for key, spec in own_property_mapping(cls).items():
        validate_property_spec(cls.__name__, key, spec, surface)

    for klass in cls.__mro__[1:]:
        # Real inheritance only: an ABC.register virtual subclass answers issubclass True without
        # ever running __init_subclass__, so it never validated itself.
        if base in klass.__mro__:
            continue
        for key, spec in own_property_mapping(klass).items():
            validate_property_spec(klass.__name__, key, spec, surface, via=cls.__name__)
