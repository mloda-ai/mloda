"""Failing tests for the registry-backed enumeration/docs rebuild (issue #526, work item 4).

Contract: the registry becomes a first-class enumeration source next to the
__subclasses__() walk. mloda.user exports PluginRegistry and register. A new
list_registered(plugin_type) lives in mloda.core.api.plugin_docs and is
re-exported from mloda.user and mloda.steward; it returns the default-registry
classes for a plugin base type, sorted by registry key. The docs functions gain
a registered_only keyword (default False) that restricts output to registered
classes. Default-mode docs remain the union of registry and subclass walk, with
no duplicate entries.

The autouse conftest fixture restores the default registry around every test,
so clearing it inside a test is safe.
"""

from typing import Any

import mloda.steward
import mloda.user
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import register as registry_register
from mloda.core.api.plugin_docs import (
    get_compute_framework_docs,
    get_extender_docs,
    get_feature_group_docs,
)


# Fails today: PluginRegistry and register are not exported from mloda.user.
class TestPublicRegistryExports:
    def test_user_exports_plugin_registry_and_register(self) -> None:
        from mloda.user import PluginRegistry as user_plugin_registry
        from mloda.user import register as user_register

        assert user_plugin_registry is PluginRegistry
        assert user_register is registry_register
        assert "PluginRegistry" in mloda.user.__all__
        assert "register" in mloda.user.__all__


# Fails today: list_registered does not exist in mloda.core.api.plugin_docs.
class TestListRegisteredDocsApi:
    def test_list_registered_returns_registry_contents_for_base_type(self) -> None:
        from mloda.core.api.plugin_docs import list_registered

        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumListedFG(FeatureGroup):
            """Local feature group registered for enumeration."""

        registry_register(_DocsEnumListedFG)

        assert list_registered(FeatureGroup) == [_DocsEnumListedFG]
        assert list_registered(ComputeFramework) == []
        assert list_registered(Extender) == []

    def test_list_registered_sorted_by_registry_key(self) -> None:
        from mloda.core.api.plugin_docs import list_registered

        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumZzFG(FeatureGroup):
            """Sorts after the Aa double by registry key."""

        class _DocsEnumAaFG(FeatureGroup):
            """Sorts before the Zz double by registry key."""

        registry_register(_DocsEnumZzFG)
        registry_register(_DocsEnumAaFG)

        assert list_registered(FeatureGroup) == [_DocsEnumAaFG, _DocsEnumZzFG], (
            "list_registered must sort by registry key, not by registration order"
        )

    def test_list_registered_reexported_from_user_and_steward(self) -> None:
        from mloda.core.api.plugin_docs import list_registered

        assert getattr(mloda.user, "list_registered") is list_registered
        assert getattr(mloda.steward, "list_registered") is list_registered
        assert "list_registered" in mloda.user.__all__
        assert "list_registered" in mloda.steward.__all__


# Fails today: the docs functions do not accept a registered_only keyword.
class TestRegisteredOnlyDocsFiltering:
    def test_feature_group_docs_registered_only_filters_to_registry(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumRegOnlyFG(FeatureGroup):
            """Registered-only filtering test double for feature group docs."""

        before = [fg.name for fg in get_feature_group_docs(registered_only=True)]
        assert "_DocsEnumRegOnlyFG" not in before, (
            "registered_only=True must hide subclass-walk classes that are not in the registry"
        )

        registry_register(_DocsEnumRegOnlyFG)
        after = [fg.name for fg in get_feature_group_docs(registered_only=True)]
        assert "_DocsEnumRegOnlyFG" in after, "registered_only=True must document registered classes"

    def test_feature_group_docs_registered_only_defaults_to_false(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        default_names = [fg.name for fg in get_feature_group_docs()]
        explicit_names = [fg.name for fg in get_feature_group_docs(registered_only=False)]
        assert default_names == explicit_names

    def test_compute_framework_docs_registered_only_filters_to_registry(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumRegOnlyCFW(ComputeFramework):
            """Registered-only filtering test double for compute framework docs."""

        before = [cfw.name for cfw in get_compute_framework_docs(registered_only=True, available_only=False)]
        assert "_DocsEnumRegOnlyCFW" not in before

        registry_register(_DocsEnumRegOnlyCFW)
        after = [cfw.name for cfw in get_compute_framework_docs(registered_only=True, available_only=False)]
        assert "_DocsEnumRegOnlyCFW" in after

    def test_extender_docs_registered_only_filters_to_registry(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumRegOnlyExtender(Extender):
            """Registered-only filtering test double for extender docs."""

            def wraps(self) -> set[ExtenderHook]:
                return set()

            def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        before = [ext.name for ext in get_extender_docs(registered_only=True)]
        assert "_DocsEnumRegOnlyExtender" not in before

        registry_register(_DocsEnumRegOnlyExtender)
        after = [ext.name for ext in get_extender_docs(registered_only=True)]
        assert "_DocsEnumRegOnlyExtender" in after


# Non-regression guard: default-mode docs keep the subclass-walk fallback.
class TestDocsFallbackNonRegression:
    def test_unregistered_local_fg_still_documented(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumFallbackFG(FeatureGroup):
            """Unregistered local feature group; the subclass walk must still find it."""

        names = [fg.name for fg in get_feature_group_docs()]
        assert "_DocsEnumFallbackFG" in names

    def test_default_docs_are_union_of_registry_and_subclass_walk(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumUnionRegisteredFG(FeatureGroup):
            """Registered half of the union test pair."""

        class _DocsEnumUnionUnregisteredFG(FeatureGroup):
            """Unregistered half of the union test pair."""

        registry_register(_DocsEnumUnionRegisteredFG)

        names = [fg.name for fg in get_feature_group_docs()]
        assert "_DocsEnumUnionRegisteredFG" in names
        assert "_DocsEnumUnionUnregisteredFG" in names

    def test_registered_and_walkable_fg_listed_exactly_once(self) -> None:
        registry = PluginRegistry.default()
        registry.clear()

        class _DocsEnumOnceFG(FeatureGroup):
            """Both registered and subclass-walk reachable; must be documented once."""

        registry_register(_DocsEnumOnceFG)

        names = [fg.name for fg in get_feature_group_docs()]
        assert names.count("_DocsEnumOnceFG") == 1, (
            "a class that is both registered and subclass-reachable must not be documented twice"
        )
