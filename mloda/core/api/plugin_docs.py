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

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.api.plugin_info import ComputeFrameworkInfo, ExtenderInfo, FeatureGroupInfo, ResolvedFeature
from mloda.core.prepare.accessible_plugins import dedup_feature_group_subclasses
from mloda.core.prepare.identify_feature_group import split_frameworks_by_capability


def list_registered(plugin_type: type[Any]) -> list[type[Any]]:
    """Return the default-registry classes for a plugin base type, sorted by registry key."""
    return PluginRegistry.default().list_registered(plugin_type)


def _safe_version(fg_class: type[FeatureGroup]) -> str:
    """Return the feature group version or "unavailable" if source introspection fails."""
    try:
        return fg_class.version()
    except (OSError, TypeError):
        return "unavailable"


def _dedup_degrading_on_conflict(
    feature_groups: set[type[FeatureGroup]], allow_redefinition: bool
) -> set[type[FeatureGroup]]:
    """Dedup feature groups, keeping all conflicting versions instead of raising on redefinition."""
    try:
        return dedup_feature_group_subclasses(feature_groups, allow_redefinition=allow_redefinition)
    except ValueError as exc:
        conflicts: set[type[FeatureGroup]] = set(getattr(exc, "conflicts", []))
        survivors = dedup_feature_group_subclasses(feature_groups - conflicts, allow_redefinition=allow_redefinition)
        return survivors | conflicts


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

    Returns gracefully on redefinition conflicts (all conflicting versions are
    documented) and reports "unavailable" as version for classes whose source
    cannot be introspected.

    Args:
        name: Filter by feature group name (case-insensitive partial match).
        search: Search in feature group description (case-insensitive partial match).
        compute_framework: Filter by compute framework name or class.
        version_contains: Filter by version substring.
        plugin_collector: Filter using plugin collector's applicability check.
        registered_only: If True, only document classes registered in the default registry.

    Returns:
        List of FeatureGroupInfo objects sorted by name.
    """
    allow_redefinition = plugin_collector.allow_redefinition if plugin_collector is not None else False
    all_feature_groups: set[type[FeatureGroup]] = get_all_subclasses(FeatureGroup)
    if registered_only:
        registered = {entry.cls for entry in PluginRegistry.default().snapshot().values()}
        all_feature_groups = {fg for fg in all_feature_groups if fg in registered}
    if plugin_collector is not None:
        all_feature_groups = {fg for fg in all_feature_groups if plugin_collector.applicable_feature_group_class(fg)}
    all_feature_groups = _dedup_degrading_on_conflict(all_feature_groups, allow_redefinition=allow_redefinition)
    results = []

    for fg_class in all_feature_groups:
        if fg_class.__module__ == "__main__":
            continue

        fg_name = fg_class.get_class_name()
        description = fg_class.description()
        version = _safe_version(fg_class)
        module = fg_class.__module__
        compute_frameworks = [cfw.__name__ for cfw in fg_class.compute_framework_definition()]
        supported_feature_names = fg_class.feature_names_supported()
        prefix = fg_class.prefix()

        if name is not None and name.lower() not in fg_name.lower():
            continue

        if search is not None and search.lower() not in description.lower():
            continue

        if compute_framework is not None:
            cfw_name = compute_framework if isinstance(compute_framework, str) else compute_framework.__name__
            if cfw_name not in compute_frameworks:
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
    available_only: bool = True,
    registered_only: bool = False,
) -> list[ComputeFrameworkInfo]:
    """
    Get documentation for compute frameworks with optional filtering.

    Returns the current state of loaded compute frameworks. Ensure plugins are loaded
    before calling this function (e.g., via PluginLoader.all()).

    Args:
        name: Filter by compute framework name (case-insensitive partial match).
        search: Search in compute framework description (case-insensitive partial match).
        available_only: If True, only return available frameworks (default True).
        registered_only: If True, only document classes registered in the default registry.

    Returns:
        List of ComputeFrameworkInfo objects sorted by name.
    """
    all_compute_frameworks = get_all_subclasses(ComputeFramework)
    if registered_only:
        registered = {entry.cls for entry in PluginRegistry.default().snapshot().values()}
        all_compute_frameworks = {cfw for cfw in all_compute_frameworks if cfw in registered}
    results = []

    for cfw_class in all_compute_frameworks:
        if cfw_class.__module__ == "__main__":
            continue

        cfw_name = cfw_class.__name__
        description = (cfw_class.__doc__ or "").strip() or cfw_class.__name__
        module = cfw_class.__module__

        is_available = cfw_class.is_available()

        try:
            expected_data_framework = str(cfw_class.expected_data_framework())
        except Exception:  # nosec
            expected_data_framework = "unavailable"

        try:
            has_merge_engine = cfw_class.merge_engine() is not None
        except Exception:  # nosec
            has_merge_engine = False

        try:
            has_filter_engine = cfw_class.filter_engine() is not None
        except Exception:  # nosec
            has_filter_engine = False

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
        registered = {entry.cls for entry in PluginRegistry.default().snapshot().values()}
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

        wraps_list: list[str] = []
        try:
            instance = ext_class()
            wraps_list = [w.value for w in instance.wraps()]
        except Exception:  # nosec
            pass

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


def resolve_feature(feature_name: str, plugin_collector: Optional[PluginCollector] = None) -> ResolvedFeature:
    """
    Resolve a feature name to its matching FeatureGroup class.

    This function searches all loaded FeatureGroups to find those that match
    the given feature name. It applies subclass filtering to prefer more
    specific (child) classes over parent classes.

    Args:
        feature_name: The name of the feature to resolve.
        plugin_collector: Optional PluginCollector. When provided, its
            ``allow_redefinition`` flag is threaded into deduplication so that
            redefined FeatureGroup classes (e.g. via reload) do not raise.

    Returns:
        ResolvedFeature containing the resolved FeatureGroup (if found),
        all matching candidates, and any error message.
    """
    allow_redefinition = plugin_collector.allow_redefinition if plugin_collector is not None else False
    fg_classes: set[type[FeatureGroup]] = get_all_subclasses(FeatureGroup)
    if plugin_collector is not None:
        fg_classes = {fg for fg in fg_classes if plugin_collector.applicable_feature_group_class(fg)}
    try:
        all_fgs = list(dedup_feature_group_subclasses(fg_classes, allow_redefinition=allow_redefinition))
    except ValueError as exc:
        raw_conflicts = list(getattr(exc, "conflicts", []))
        feature_name_obj = FeatureName(feature_name)
        matching_conflicts = [
            fg for fg in raw_conflicts if fg.match_feature_group_criteria(feature_name_obj, Options(), None)
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
        if fg.match_feature_group_criteria(feature_name_obj, Options(), None):
            candidates.append(fg)

    if not candidates:
        return ResolvedFeature(
            feature_name=feature_name,
            feature_group=None,
            candidates=[],
            error=f"No FeatureGroup found for feature name: {feature_name}",
        )

    supported, rejected = split_frameworks_by_capability(candidates, feature_name_obj, Options())
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
                f"(evaluated under default options): {rejected_names}."
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
