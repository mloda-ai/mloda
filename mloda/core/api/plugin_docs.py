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
from mloda.core.abstract_plugins.components.subtype_declaration import SubtypeDeclaration
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses, safe_field, safe_field_with_error
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature, normalize_feature_group_scope
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.api.plugin_info import ComputeFrameworkInfo, ExtenderInfo, FeatureGroupInfo, ResolvedFeature
from mloda.core.prepare.accessible_plugins import (
    EnvironmentPreconditionError,
    FeatureGroupEnvironmentMapping,
    FrameworkDeclarationError,
    PreFilterPlugins,
    RedefinitionConflictError,
    dedup_feature_group_subclasses,
    registry_for,
)
from mloda.core.prepare.identify_feature_group import (
    IdentifyFeatureGroupClass,
    render_resolution_failure,
    scope_callout,
)


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


def _subtype_support_or_error(fg_class: type[FeatureGroup]) -> tuple[dict[str, list[str]], Optional[str]]:
    """Return the sorted subtype support matrix, degrading a raising class to ({}, message)."""
    empty: dict[str, frozenset[str]] = {}
    matrix, error = safe_field_with_error(lambda: fg_class.subtype_support_matrix(), empty)
    if error is not None:
        return {}, error
    return {framework: sorted(supported) for framework, supported in matrix.items()}, None


def _resolved_subtype_fields(
    fg_class: type[FeatureGroup], feature_name: FeatureName, options: Options
) -> tuple[Optional[str], Optional[str]]:
    """Resolve (subtype, subtype_family) for a candidate, degrading a raising declaration to (None, None)."""
    subtype: Optional[str] = safe_field(lambda: fg_class.resolve_subtype(feature_name, options), None)
    if subtype is None:
        return None, None
    raw_subtype = subtype
    canonical = safe_field(lambda: fg_class.canonical_subtype(raw_subtype), raw_subtype)
    if canonical != raw_subtype:
        return subtype, canonical
    return subtype, None


def _dedup_degrading_on_conflict(
    feature_groups: set[type[FeatureGroup]], allow_redefinition: bool
) -> set[type[FeatureGroup]]:
    """Dedup feature groups, collapsing each redefinition conflict to its live class instead of raising."""
    try:
        return dedup_feature_group_subclasses(feature_groups, allow_redefinition=allow_redefinition)
    except RedefinitionConflictError:
        return dedup_feature_group_subclasses(feature_groups, allow_redefinition=True)


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

        declaration: Optional[SubtypeDeclaration] = safe_field(
            lambda: fg_class.SUBTYPES, None, field=f"{cls_name}.SUBTYPES"
        )
        subtype_key = declaration.key if declaration is not None else None
        subtypes: list[str] = safe_field(
            lambda: sorted(fg_class.subtype_universe()), [], field=f"{cls_name}.subtype_universe"
        )
        parametric_subtypes = sorted(declaration.family_names()) if declaration is not None else []
        subtype_support, subtype_error = _subtype_support_or_error(fg_class)

        results.append(
            FeatureGroupInfo(
                name=fg_name,
                description=description,
                version=version,
                module=module,
                compute_frameworks=compute_frameworks,
                supported_feature_names=supported_feature_names,
                prefix=prefix,
                subtype_key=subtype_key,
                subtypes=subtypes,
                parametric_subtypes=parametric_subtypes,
                subtype_support=subtype_support,
                subtype_error=subtype_error,
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
    feature: str | Feature,
    *,
    options: Optional[Options] = None,
    plugin_collector: Optional[PluginCollector] = None,
    feature_group: str | type[FeatureGroup] | None = None,
    links: Optional[set[Link]] = None,
    data_access_collection: Optional[DataAccessCollection] = None,
    compute_frameworks: Optional[set[type[ComputeFramework]]] = None,
) -> ResolvedFeature:
    """
    Resolve a feature name (or a Feature object) to its matching FeatureGroup class.

    This function searches all loaded FeatureGroups to find those that match
    the given feature name. It applies subclass filtering to prefer more
    specific (child) classes over parent classes.

    Never raises for matching errors: those are reported in the returned ResolvedFeature.error.
    Signature misuse (options/feature_group alongside a Feature) is a programmer error and raises TypeError.

    Design note: resolve_feature is a thin adapter. It only normalizes the standalone request, builds the
    canonical accessible-plugins environment once, delegates one evaluation to
    IdentifyFeatureGroupClass.evaluate, and projects the result. ANY environment-build failure (including
    redefinition conflicts) is projected fail-closed from the failure itself into ``error`` with no
    candidates and no re-matching; the seam owns name/domain/scope/abstract/subclass filtering, the
    winner, candidates, and the failure texts.

    Engine inputs now covered: name, options, domain and compute-framework pin (carried on the Feature),
    scope (via the Feature's feature_group_scope or the feature_group argument for the string form), and
    the ``links`` / ``data_access_collection`` arguments threaded into the seam. No engine input is left out.

    Args:
        feature: A feature name string, or a Feature object used directly as the single source of truth for
            name, options, domain, compute-framework pin and scope.
        options: Keyword-only. Options used for matching and capability checks (string form only). An empty or
            omitted Options is the documented default. Passing it alongside a Feature raises TypeError.
        plugin_collector: Keyword-only. Its ``allow_redefinition`` flag is threaded into deduplication.
        feature_group: Keyword-only. Scope resolution to a FeatureGroup subclass or a class-name string
            (string form only), same forms and semantics as ``Feature(..., feature_group=...)``: the string
            form matches the named class and its subclasses; None (or a whitespace-only string) means unscoped.
            Passing it alongside a Feature raises TypeError.
        links: Keyword-only. Threaded to the seam's link filter.
        data_access_collection: Keyword-only. Threaded to the seam so reader / input-data groups can resolve.
        compute_frameworks: Keyword-only. Restrict the candidate universe's compute-framework set (default: all available frameworks).

    Returns:
        ResolvedFeature containing the resolved FeatureGroup (if found),
        all matching candidates, and any error message.
    """
    if isinstance(feature, Feature):
        if options is not None or feature_group is not None:
            raise TypeError(
                "resolve_feature(Feature(...)) is the single source of truth for name, options and scope; "
                "set them on the Feature, do not also pass 'options' or 'feature_group'"
            )
        feature_obj: Optional[Feature] = feature
        feature_name = str(feature.name)
        scope = feature.feature_group_scope
        resolved_options = feature.options
    else:
        feature_obj = None
        feature_name = feature
        try:
            scope = normalize_feature_group_scope(feature_group)
        except TypeError as exc:
            return ResolvedFeature(
                feature_name=feature_name,
                feature_group=None,
                candidates=[],
                error=str(exc),
            )
        resolved_options = options if options is not None else Options()

    callout = scope_callout(scope)
    scope_suffix = f" {callout}" if callout else ""
    feature_name_obj = FeatureName(feature_name)

    restricted_frameworks = (
        compute_frameworks if compute_frameworks is not None else get_all_subclasses(ComputeFramework)
    )
    try:
        accessible_plugins: FeatureGroupEnvironmentMapping = PreFilterPlugins(
            restricted_frameworks, plugin_collector, attribute_declaration_failures=True
        ).get_accessible_plugins()
    except (RedefinitionConflictError, EnvironmentPreconditionError, FrameworkDeclarationError) as exc:
        # mloda's own environment failure: already a complete sentence. Project it bare, no candidates.
        return ResolvedFeature(feature_name, None, [], error=f"{exc}{scope_suffix}")
    except Exception as exc:  # noqa: BLE001  (never-raising debug API; projects a broken plugin's build failure)
        return ResolvedFeature(
            feature_name,
            None,
            [],
            error=f"Failed to build the plugin environment: {type(exc).__name__}: {exc}{scope_suffix}",
        )

    if feature_obj is None:
        feature_obj, feature_error = safe_field_with_error(
            lambda: Feature(feature_name, options=resolved_options, feature_group=scope),
            None,
        )
        if feature_obj is None:
            return ResolvedFeature(feature_name, None, [], error=f"{feature_error}{scope_suffix}")

    # The seam does the name/domain/scope/abstract/subclass filtering. Matching or a capability hook can raise;
    # resolve_feature must not, so degrade any raise into an error result (never-raising contract).
    evaluation = safe_field_with_error(
        lambda: IdentifyFeatureGroupClass.evaluate(
            feature_obj, accessible_plugins, links=links, data_access_collection=data_access_collection
        ),
        None,
    )
    result, eval_error = evaluation
    if result is None:
        return ResolvedFeature(feature_name, None, [], error=f"{eval_error}{scope_suffix}")

    candidates = sorted(result.criteria_matched, key=lambda c: c.get_class_name())

    if result.failure_kind is None:
        winner, supported_frameworks = next(iter(result.identified.items()))
        available = accessible_plugins.get(winner, set())
        supported_names = sorted(c.get_class_name() for c in supported_frameworks)
        unsupported_names = sorted(c.get_class_name() for c in (available - supported_frameworks))
        subtype, subtype_family = _resolved_subtype_fields(winner, feature_name_obj, resolved_options)
        return ResolvedFeature(
            feature_name,
            winner,
            candidates,
            None,
            supported_compute_frameworks=supported_names,
            unsupported_compute_frameworks=unsupported_names,
            subtype=subtype,
            subtype_family=subtype_family,
        )

    # A projection of the evaluation already in hand: the same result the engine renders from, and the
    # renderer emits the scope callout itself, so scope_suffix must not be appended a second time.
    error = render_resolution_failure(result, feature_obj)
    return ResolvedFeature(feature_name, None, candidates, error=error)
