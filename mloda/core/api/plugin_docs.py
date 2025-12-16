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

from typing import List, Optional, Type, Union

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.api.plugin_info import ComputeFrameworkInfo, ExtenderInfo, FeatureGroupInfo


def get_feature_group_docs(
    name: Optional[str] = None,
    search: Optional[str] = None,
    compute_framework: Optional[Union[str, Type[ComputeFramework]]] = None,
    version_contains: Optional[str] = None,
    plugin_collector: Optional[PluginCollector] = None,
) -> List[FeatureGroupInfo]:
    """
    Get documentation for feature groups with optional filtering.

    Returns the current state of loaded feature groups. Ensure plugins are loaded
    before calling this function (e.g., via PluginLoader.all()).

    Args:
        name: Filter by feature group name (case-insensitive partial match).
        search: Search in feature group description (case-insensitive partial match).
        compute_framework: Filter by compute framework name or class.
        version_contains: Filter by version substring.
        plugin_collector: Filter using plugin collector's applicability check.

    Returns:
        List of FeatureGroupInfo objects sorted by name.
    """
    all_feature_groups = get_all_subclasses(FeatureGroup)
    results = []

    for fg_class in all_feature_groups:
        fg_name = fg_class.get_class_name()
        description = fg_class.description()
        version = fg_class.version()
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

        if plugin_collector is not None and not plugin_collector.applicable_feature_group_class(fg_class):
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
) -> List[ComputeFrameworkInfo]:
    """
    Get documentation for compute frameworks with optional filtering.

    Returns the current state of loaded compute frameworks. Ensure plugins are loaded
    before calling this function (e.g., via PluginLoader.all()).

    Args:
        name: Filter by compute framework name (case-insensitive partial match).
        search: Search in compute framework description (case-insensitive partial match).
        available_only: If True, only return available frameworks (default True).

    Returns:
        List of ComputeFrameworkInfo objects sorted by name.
    """
    all_compute_frameworks = get_all_subclasses(ComputeFramework)
    results = []

    for cfw_class in all_compute_frameworks:
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
) -> List[ExtenderInfo]:
    """
    Get documentation for extenders with optional filtering.

    Returns the current state of loaded extenders. Ensure plugins are loaded
    before calling this function (e.g., via PluginLoader.all()).

    Args:
        name: Filter by extender name (case-insensitive partial match).
        search: Search in extender description (case-insensitive partial match).
        wraps: Filter by wrapped function type (case-insensitive exact match).

    Returns:
        List of ExtenderInfo objects sorted by name.
    """
    all_extenders = get_all_subclasses(Extender)
    results = []

    for ext_class in all_extenders:
        ext_name = ext_class.__name__
        description = (ext_class.__doc__ or "").strip() or ext_class.__name__
        module = ext_class.__module__

        if ext_name in ("Extender", "_CompositeExtender"):
            continue

        wraps_list: List[str] = []
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
