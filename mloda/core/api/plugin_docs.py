"""
Plugin documentation discovery functions.

These functions return documentation and metadata for currently loaded plugins.
They report the current state - ensure plugins are loaded before calling.

Example:
    from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
    from mloda.core.api.plugin_docs import get_feature_group_docs

    # Load plugins first
    PluginLoader.all()

    # Then get documentation
    docs = get_feature_group_docs()
"""

from typing import Any, Optional

from mloda.core.abstract_plugins.components.base_feature_group_version import SOURCE_INTROSPECTION_ERRORS
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses, safe_field
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.api.plugin_info import ComputeFrameworkInfo, ExtenderInfo, FeatureGroupInfo, ResolvedFeature
from mloda.core.prepare.accessible_plugins import (
    RedefinitionConflictError,
    dedup_feature_group_subclasses,
    registry_for,
)
from mloda.core.prepare.identify_feature_group import split_frameworks_by_capability


def list_registered(plugin_type: type[Any]) -> list[type[Any]]:
    """Return the default-registry classes for a plugin base type, sorted by registry key."""
    return PluginRegistry.default().list_registered(plugin_type)


def _as_str(value: Any) -> str:
    """Return `value` unchanged, raising TypeError if a plugin returned a non-str, so the guarded read degrades."""
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def _safe_version(fg_class: type[FeatureGroup]) -> str:
    """Return the feature group version or "unavailable" if source introspection fails or version() is not a str."""
    return safe_field(lambda: _as_str(fg_class.version()), "unavailable", catching=SOURCE_INTROSPECTION_ERRORS)


def _dedup_degrading_on_conflict(
    feature_groups: set[type[FeatureGroup]], allow_redefinition: bool
) -> set[type[FeatureGroup]]:
    """Dedup feature groups, collapsing each redefinition conflict to its live class instead of raising."""
    try:
        return dedup_feature_group_subclasses(feature_groups, allow_redefinition=allow_redefinition)
    except RedefinitionConflictError:
        return dedup_feature_group_subclasses(feature_groups, allow_redefinition=True)


def _match_recording_validation_errors(
    fg_class: type[FeatureGroup],
    feature_name: FeatureName,
    options: Options,
    validation_errors: list[str],
) -> bool:
    """Match a feature group, degrading a validation ValueError into "not a match" and recording its message.

    resolve_feature is a non-throwing debug API, but matching can raise ValueError once caller
    options reach the feature chain parser (e.g. a forwarded option contradicting the feature name).
    Such a candidate is not a match, and the reason is recorded so it can be surfaced in the error.
    """
    try:
        return fg_class.match_feature_group_criteria(feature_name, options, None)
    except ValueError as exc:
        message = str(exc)
        if message not in validation_errors:
            validation_errors.append(message)
        return False


def get_feature_group_docs(
    name: Optional[str] = None,
    search: Optional[str] = None,
    compute_framework: Optional[str | type[ComputeFramework]] = None,
    version_contains: Optional[str] = None,
    plugin_collector: Optional[PluginCollector] = None,
    registered_only: bool = False,
) -> list[FeatureGroupInfo]:
    """
    Get documentation for feature groups with optional filtering.

    Returns the current state of loaded feature groups. Ensure plugins are loaded
    before calling this function (e.g., via PluginLoader.all()).

    Returns gracefully on redefinition conflicts: instead of raising, each
    conflicting class is documented as its most recently defined (live) version.

    Degrades per field instead of failing the call: a read that raises, or that
    returns a non-str where the field feeds a filter, falls back to a
    base-class-derived value ("unavailable" for version).

    Args:
        name: Filter by feature group name (case-insensitive partial match).
        search: Search in feature group description (case-insensitive partial match).
        compute_framework: Filter by compute framework name or class (case-insensitive match).
        version_contains: Filter by version substring.
        plugin_collector: Filter using plugin collector's applicability check.
        registered_only: If True, only document classes in the collector's injected registry, else the default registry.

    Returns:
        List of FeatureGroupInfo objects sorted by name.
    """
    allow_redefinition = plugin_collector.allow_redefinition if plugin_collector is not None else False
    all_feature_groups: set[type[FeatureGroup]] = get_all_subclasses(FeatureGroup)
    if registered_only:
        registered = registry_for(plugin_collector).registered_classes()
        all_feature_groups = {fg for fg in all_feature_groups if fg in registered}
    if plugin_collector is not None:
        all_feature_groups = {fg for fg in all_feature_groups if plugin_collector.applicable_feature_group_class(fg)}
    all_feature_groups = _dedup_degrading_on_conflict(all_feature_groups, allow_redefinition=allow_redefinition)
    results = []

    for fg_class in all_feature_groups:
        if fg_class.__module__ == "__main__":
            continue

        cls_name = fg_class.__name__
        fg_name = safe_field(lambda: _as_str(fg_class.get_class_name()), cls_name, field=f"{cls_name}.get_class_name")
        doc_fallback = (fg_class.__doc__ or "").strip() or cls_name
        description = safe_field(lambda: _as_str(fg_class.description()), doc_fallback, field=f"{cls_name}.description")
        version = _safe_version(fg_class)
        module = fg_class.__module__
        compute_frameworks: list[str] = safe_field(
            lambda: [cfw.__name__ for cfw in fg_class.compute_framework_definition()],
            [],
            field=f"{cls_name}.compute_framework_definition",
        )
        supported_feature_names: set[str] = safe_field(
            lambda: fg_class.feature_names_supported(), set(), field=f"{cls_name}.feature_names_supported"
        )
        prefix = safe_field(lambda: fg_class.prefix(), f"{cls_name}_", field=f"{cls_name}.prefix")

        if name is not None and name.lower() not in fg_name.lower():
            continue

        if search is not None and search.lower() not in description.lower():
            continue

        if compute_framework is not None:
            cfw_name = compute_framework if isinstance(compute_framework, str) else compute_framework.__name__
            if cfw_name.lower() not in {c.lower() for c in compute_frameworks}:
                continue

        if version_contains is not None and version_contains not in version:
            continue

        results.append(
            FeatureGroupInfo(
                name=fg_name,
                description=description,
                version=version,
                module=module,
                compute_frameworks=compute_frameworks,
                supported_feature_names=supported_feature_names,
                prefix=prefix,
            )
        )

    return sorted(results, key=lambda x: x.name)


def get_compute_framework_docs(
    name: Optional[str] = None,
    search: Optional[str] = None,
    available_only: bool = False,
    registered_only: bool = False,
) -> list[ComputeFrameworkInfo]:
    """
    Get documentation for compute frameworks with optional filtering.

    Returns the current state of loaded compute frameworks. Ensure plugins are loaded
    before calling this function (e.g., via PluginLoader.all()).

    Args:
        name: Filter by compute framework name (case-insensitive partial match).
        search: Search in compute framework description (case-insensitive partial match).
        available_only: If True, only return importable frameworks. By default (False) all frameworks
            are listed, with is_available as the flag indicating importability.
        registered_only: If True, only document classes registered in the default registry.

    Returns:
        List of ComputeFrameworkInfo objects sorted by name.
    """
    all_compute_frameworks = get_all_subclasses(ComputeFramework)
    if registered_only:
        registered = PluginRegistry.default().registered_classes()
        all_compute_frameworks = {cfw for cfw in all_compute_frameworks if cfw in registered}
    results = []

    for cfw_class in all_compute_frameworks:
        if cfw_class.__module__ == "__main__":
            continue

        cfw_name = cfw_class.__name__
        description = (cfw_class.__doc__ or "").strip() or cfw_class.__name__
        module = cfw_class.__module__

        is_available = safe_field(lambda: cfw_class.is_available(), False)
        expected_data_framework = safe_field(lambda: str(cfw_class.expected_data_framework()), "unavailable")
        has_merge_engine = safe_field(lambda: cfw_class.merge_engine() is not None, False)
        has_filter_engine = safe_field(lambda: cfw_class.filter_engine() is not None, False)

        if available_only and not is_available:
            continue

        if name is not None and name.lower() not in cfw_name.lower():
            continue

        if search is not None and search.lower() not in description.lower():
            continue

        results.append(
            ComputeFrameworkInfo(
                name=cfw_name,
                description=description,
                module=module,
                is_available=is_available,
                expected_data_framework=expected_data_framework,
                has_merge_engine=has_merge_engine,
                has_filter_engine=has_filter_engine,
            )
        )

    return sorted(results, key=lambda x: x.name)


def get_extender_docs(
    name: Optional[str] = None,
    search: Optional[str] = None,
    wraps: Optional[str] = None,
    registered_only: bool = False,
) -> list[ExtenderInfo]:
    """
    Get documentation for extenders with optional filtering.

    Returns the current state of loaded extenders. Ensure plugins are loaded
    before calling this function (e.g., via PluginLoader.all()).

    Args:
        name: Filter by extender name (case-insensitive partial match).
        search: Search in extender description (case-insensitive partial match).
        wraps: Filter by wrapped function type (case-insensitive exact match).
        registered_only: If True, only document classes registered in the default registry.

    Returns:
        List of ExtenderInfo objects sorted by name.
    """
    all_extenders = get_all_subclasses(Extender)
    if registered_only:
        registered = PluginRegistry.default().registered_classes()
        all_extenders = {ext for ext in all_extenders if ext in registered}
    results = []

    for ext_class in all_extenders:
        if ext_class.__module__ == "__main__":
            continue

        ext_name = ext_class.__name__
        description = (ext_class.__doc__ or "").strip() or ext_class.__name__
        module = ext_class.__module__

        if ext_name in ("Extender", "_CompositeExtender"):
            continue

        wraps_list: list[str] = safe_field(lambda: [w.value for w in ext_class().wraps()], [])

        if name is not None and name.lower() not in ext_name.lower():
            continue

        if search is not None and search.lower() not in description.lower():
            continue

        if wraps is not None:
            wraps_lower = wraps.lower()
            if not any(wraps_lower == w.lower() for w in wraps_list):
                continue

        results.append(
            ExtenderInfo(
                name=ext_name,
                description=description,
                module=module,
                wraps=wraps_list,
            )
        )

    return sorted(results, key=lambda x: x.name)


def resolve_feature(
    feature_name: str,
    *,
    options: Optional[Options] = None,
    plugin_collector: Optional[PluginCollector] = None,
) -> ResolvedFeature:
    """
    Resolve a feature name to its matching FeatureGroup class.

    This function searches all loaded FeatureGroups to find those that match
    the given feature name. It applies subclass filtering to prefer more
    specific (child) classes over parent classes.

    Never raises: matching errors are reported in the returned ResolvedFeature.error.

    Args:
        feature_name: The name of the feature to resolve.
        options: Keyword-only. Options used for matching and capability checks. An empty or omitted Options is
            the documented default.
        plugin_collector: Keyword-only. Its ``allow_redefinition`` flag is threaded into deduplication.

    Returns:
        ResolvedFeature containing the resolved FeatureGroup (if found),
        all matching candidates, and any error message.
    """
    resolved_options = options if options is not None else Options()
    is_default_options = not resolved_options.group and not resolved_options.context
    options_caveat = "default options" if is_default_options else "the provided options"
    allow_redefinition = plugin_collector.allow_redefinition if plugin_collector is not None else False
    validation_errors: list[str] = []
    fg_classes: set[type[FeatureGroup]] = get_all_subclasses(FeatureGroup)
    if plugin_collector is not None:
        fg_classes = {fg for fg in fg_classes if plugin_collector.applicable_feature_group_class(fg)}
    try:
        all_fgs = list(dedup_feature_group_subclasses(fg_classes, allow_redefinition=allow_redefinition))
    except ValueError as exc:
        raw_conflicts = list(getattr(exc, "conflicts", []))
        feature_name_obj = FeatureName(feature_name)
        matching_conflicts = [
            fg
            for fg in raw_conflicts
            if _match_recording_validation_errors(fg, feature_name_obj, resolved_options, validation_errors)
        ]
        return ResolvedFeature(
            feature_name=feature_name,
            feature_group=None,
            candidates=matching_conflicts,
            error=str(exc),
        )
    candidates: list[type[FeatureGroup]] = []
    feature_name_obj = FeatureName(feature_name)

    for fg in all_fgs:
        if _match_recording_validation_errors(fg, feature_name_obj, resolved_options, validation_errors):
            candidates.append(fg)

    if not candidates:
        error = f"No FeatureGroup found for feature name: {feature_name}"
        if validation_errors:
            error = f"{error} Rejected during matching: {' | '.join(validation_errors)}"
        return ResolvedFeature(
            feature_name=feature_name,
            feature_group=None,
            candidates=[],
            error=error,
        )

    supported, rejected = split_frameworks_by_capability(candidates, feature_name_obj, resolved_options)
    supported_names = sorted(c.get_class_name() for c in supported)
    rejected_names = sorted(c.get_class_name() for c in rejected)

    if not supported and rejected:
        return ResolvedFeature(
            feature_name=feature_name,
            feature_group=None,
            candidates=candidates,
            error=(
                f"Feature '{feature_name}' matches {[c.get_class_name() for c in candidates]} "
                f"but is unsupported on all installed compute frameworks "
                f"(evaluated under {options_caveat}): {rejected_names}."
            ),
            supported_compute_frameworks=[],
            unsupported_compute_frameworks=rejected_names,
        )

    filtered = _filter_subclasses(candidates)

    if len(filtered) == 1:
        return ResolvedFeature(
            feature_name=feature_name,
            feature_group=filtered[0],
            candidates=candidates,
            error=None,
            supported_compute_frameworks=supported_names,
            unsupported_compute_frameworks=rejected_names,
        )

    conflicting_names = [fg.__name__ for fg in filtered]
    return ResolvedFeature(
        feature_name=feature_name,
        feature_group=None,
        candidates=candidates,
        error=f"Multiple FeatureGroups match feature name '{feature_name}': {conflicting_names}",
        supported_compute_frameworks=supported_names,
        unsupported_compute_frameworks=rejected_names,
    )


def _filter_subclasses(feature_groups: list[type[FeatureGroup]]) -> list[type[FeatureGroup]]:
    """Prefer more specific (child) classes over parent classes."""
    fgs_to_pop: set[type[FeatureGroup]] = set()

    for i_fg in feature_groups:
        for o_fg in feature_groups:
            if i_fg == o_fg:
                continue
            if issubclass(i_fg, o_fg):
                fgs_to_pop.add(o_fg)

    return [fg for fg in feature_groups if fg not in fgs_to_pop]
