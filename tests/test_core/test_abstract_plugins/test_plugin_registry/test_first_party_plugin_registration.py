"""Every first-party plugin class must stay in the plugin registry, whatever package it lives in (issue #735).

Contract, pinned as a regression guard: after PluginLoader.all(), every concrete public FeatureGroup,
ComputeFramework and Extender shipped in the mloda wheel (package root ``mloda`` or ``mloda_plugins``, both
ship together) holds a default-registry entry, and a first-party ComputeFramework is treated as bundled
regardless of which of the two roots it lives in. Registry membership rests on an unwritten coupling
(loader base package + CORE_PLUGIN_MODULES + the module that DEFINES a class), and #726 showed a class can
leave the registry silently: ApiInputDataFeature moved to mloda.core, dropped out, and tox stayed green.

Exclusions:
- Abstract classes: infrastructure, never registered, never registry-gated.
- Private (leading-underscore) classes: internal machinery, not plugins. The only one today is
  _CompositeExtender, built inside ComputeFramework.get_function_extender from an already-filtered extender
  set, so it never passes through the strict-mode gate (pinned below). The doubles in this module are
  underscore-prefixed for the same reason: the guard must not flag them.

Parallel-safety: the import sweep is cached per worker, assertions are membership-based on this module's own
doubles, and the autouse conftest fixture restores the default registry and the warn-mode dedup set around
every test, so clearing the registry here is safe.
"""

import gc
import importlib
import inspect
import logging
import pkgutil
import sys
import types
from functools import cache
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook, _CompositeExtender
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import (
    CORE_PLUGIN_MODULES,
    OPTIONAL_PLUGIN_DEPENDENCIES,
)
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.prepare.accessible_plugins import (
    PreFilterPlugins,
    _is_bundled_plugin,
    filter_extenders_by_strict_mode,
)
from mloda.user import PluginLoader
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

FIRST_PARTY_ROOTS: tuple[str, ...] = ("mloda", "mloda_plugins")

LOGGER_NAME = "mloda.core.prepare.accessible_plugins"

CORE_DRIFT_MODULE = "mloda.core.first_party_guard_drift_probe"


class _GuardObserverFeatureGroup(FeatureGroup):
    """Observer with no compute_framework_rule: its accessible mapping value is the surviving framework set."""


class _CoreRootedFramework(ComputeFramework):
    """Bundled ComputeFramework that drifted into mloda.core, the ComputeFramework sibling of #726."""

    __module__ = "mloda.core.compute_framework.core_rooted_framework"


class _ExtenderHostFramework(ComputeFramework):
    """Host used to build a _CompositeExtender through the real get_function_extender path."""


class _GuardExtenderA(Extender):
    """Registered extender double."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _GuardExtenderB(Extender):
    """Second registered extender double, so get_function_extender composes instead of returning one."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _qualid(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


@cache
def _sweep_first_party_packages() -> None:
    """Import every first-party module once per worker, so the class set does not depend on import history.

    Missing optional dependencies are skipped by the loader's own OPTIONAL_PLUGIN_DEPENDENCIES policy, so the
    guard keeps passing under tox -e nopyarrow; any other ModuleNotFoundError is a genuine error.
    """
    for root in FIRST_PARTY_ROOTS:
        package = importlib.import_module(root)
        for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{root}."):
            try:
                importlib.import_module(module_info.name)
            except ModuleNotFoundError as error:
                missing_root = error.name.split(".")[0] if error.name else ""
                if missing_root in OPTIONAL_PLUGIN_DEPENDENCIES:
                    continue
                raise


def _loaded_registry() -> PluginRegistry:
    """Default registry holding exactly what one full PluginLoader run discovers."""
    registry = PluginRegistry.default()
    registry.clear()
    PluginLoader.all()
    return registry


def _is_first_party(cls: type[Any]) -> bool:
    return cls.__module__.split(".", 1)[0] in FIRST_PARTY_ROOTS


def _first_party_plugin_classes(base: type[Any], *, include_private: bool = False) -> set[type[Any]]:
    """Concrete first-party subclasses of a plugin base type; private classes are excluded by default."""
    _sweep_first_party_packages()
    return {
        cls
        for cls in get_all_subclasses(base)
        if _is_first_party(cls)
        and not inspect.isabstract(cls)
        and (include_private or not cls.__qualname__.startswith("_"))
    }


def _unregistered_first_party(base: type[Any], *, include_private: bool = False) -> list[type[Any]]:
    """First-party plugin classes with no entry in the registry after a full loader run."""
    candidates = _first_party_plugin_classes(base, include_private=include_private)
    registered = _loaded_registry().registered_classes()
    return sorted((cls for cls in candidates if cls not in registered), key=_qualid)


def _fix_hint(base: type[Any], unregistered: list[type[Any]]) -> str:
    names = ", ".join(_qualid(cls) for cls in unregistered)
    return (
        f"PluginLoader.all() must register every first-party {base.__name__}, but these have no registry "
        f"entry: {names}. Strict mode drops them, warn mode names them, and registered-only listings lose "
        "them, with nothing failing today. Fix: a class defined under mloda/core/ must be listed in "
        "CORE_PLUGIN_MODULES (mloda/core/abstract_plugins/plugin_loader/plugin_loader.py); a class under "
        "mloda_plugins must live in a module the loader scans (a real module, not an __init__.py, and not a "
        "re-export: register_module_plugins only registers the module that DEFINES the class)."
    )


class TestEveryFirstPartyPluginIsRegistered:
    def test_every_first_party_feature_group_is_registered(self) -> None:
        unregistered = _unregistered_first_party(FeatureGroup)
        assert unregistered == [], _fix_hint(FeatureGroup, unregistered)

    def test_every_first_party_compute_framework_is_registered(self) -> None:
        unregistered = _unregistered_first_party(ComputeFramework)
        assert unregistered == [], _fix_hint(ComputeFramework, unregistered)

    def test_every_first_party_extender_is_registered(self) -> None:
        unregistered = _unregistered_first_party(Extender)
        assert unregistered == [], _fix_hint(Extender, unregistered)


class TestGuardDetectsCoreDrift:
    """The guard must have teeth: it fails on exactly the drift that happened in #726."""

    def test_guard_reports_a_core_feature_group_missing_from_core_plugin_modules(self) -> None:
        module = types.ModuleType(CORE_DRIFT_MODULE)

        class DriftedCoreFeatureGroup(FeatureGroup):
            """Public concrete FeatureGroup defined under mloda.core, unknown to CORE_PLUGIN_MODULES."""

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                # Inert while it lives: the default matcher instantiates candidates, and this class
                # has no importable source file behind its synthetic module.
                return False

        DriftedCoreFeatureGroup.__module__ = CORE_DRIFT_MODULE
        setattr(module, "DriftedCoreFeatureGroup", DriftedCoreFeatureGroup)
        sys.modules[CORE_DRIFT_MODULE] = module

        try:
            assert CORE_DRIFT_MODULE not in CORE_PLUGIN_MODULES, "sanity: the loader does not know the drift probe"
            probe_qualid = _qualid(DriftedCoreFeatureGroup)
            reported = [_qualid(cls) for cls in _unregistered_first_party(FeatureGroup)]
        finally:
            # Drop every strong ref so the probe cannot leak into later FeatureGroup.__subclasses__()
            # walks in this worker, which would poison unrelated tests and this file's own guard.
            del sys.modules[CORE_DRIFT_MODULE]
            delattr(module, "DriftedCoreFeatureGroup")
            del DriftedCoreFeatureGroup, module
            gc.collect()

        assert probe_qualid in reported, (
            "the guard must report a concrete public FeatureGroup defined under mloda.core that no loader path "
            f"registers; it reported {reported} instead. Without this, a plugin class moving out of "
            "mloda_plugins leaves the registry silently, exactly as ApiInputDataFeature did in #726."
        )


class TestCompositeExtenderExclusion:
    """_CompositeExtender is private internal machinery, not a plugin: excluded by rule, not by luck."""

    def test_composite_extender_is_first_party_concrete_and_unregistered(self) -> None:
        assert _is_first_party(_CompositeExtender), "sanity: it ships with mloda"
        assert not inspect.isabstract(_CompositeExtender), "sanity: it is concrete"
        assert _CompositeExtender.__qualname__.startswith("_"), "sanity: it is private"

        flagged = _unregistered_first_party(Extender, include_private=True)
        assert _CompositeExtender in flagged, (
            "sanity: no loader path registers _CompositeExtender, so the guard would flag it if private classes "
            "were included; the exclusion must be an explicit rule"
        )
        assert _CompositeExtender not in _unregistered_first_party(Extender), (
            "the private-class exclusion must keep _CompositeExtender out of the guard: it is internal "
            "machinery, never a registry-gated plugin"
        )

    def test_composite_extender_is_built_after_the_strict_mode_filter(self) -> None:
        """It is composed from an already-filtered extender set, so being unregistered is harmless."""
        register_plugin(_GuardExtenderA)
        register_plugin(_GuardExtenderB)
        collector = PluginCollector().set_strict_mode("strict")

        extenders: set[Extender] = {_GuardExtenderA(), _GuardExtenderB()}
        surviving = filter_extenders_by_strict_mode(extenders, collector)
        assert surviving == extenders, "sanity: registered extenders survive strict mode"

        framework = _ExtenderHostFramework(function_extender=surviving)
        composite = framework.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        assert isinstance(composite, _CompositeExtender), "sanity: two matching extenders compose"

        assert filter_extenders_by_strict_mode({composite}, collector) == set(), (
            "strict mode would drop a _CompositeExtender, since it is unregistered; it is safe only because "
            "ComputeFramework.get_function_extender builds it downstream of the filter, which is why the "
            "registration guard excludes private classes instead of demanding an entry for it"
        )


class TestFirstPartyComputeFrameworkIsBundledRegardlessOfPackageRoot:
    """mloda ships mloda.* and mloda_plugins.* in one wheel: bundled means either root, not just mloda_plugins."""

    def test_is_bundled_plugin_accepts_a_core_rooted_framework(self) -> None:
        assert _is_bundled_plugin(PythonDictFramework), "sanity: an mloda_plugins framework is bundled"
        assert _is_bundled_plugin(_CoreRootedFramework), (
            "a bundled ComputeFramework that moves into mloda.core is still bundled: mloda ships mloda.* and "
            "mloda_plugins.* in the same wheel. _is_bundled_plugin roots only at mloda_plugins, so the class "
            "silently loses its bundled exemption the moment it moves, the ComputeFramework sibling of #726."
        )

    def test_strict_mode_keeps_a_core_rooted_framework(self) -> None:
        _loaded_registry()
        register_plugin(_GuardObserverFeatureGroup)
        collector = PluginCollector().set_strict_mode("strict")

        frameworks: set[type[ComputeFramework]] = {_CoreRootedFramework, PythonDictFramework}
        surviving = PreFilterPlugins(frameworks, collector).get_accessible_plugins()[_GuardObserverFeatureGroup]

        assert PythonDictFramework in surviving, "sanity: a bundled mloda_plugins framework survives strict mode"
        assert _CoreRootedFramework in surviving, (
            "strict mode must not drop a first-party ComputeFramework defined under mloda.core: users cannot "
            "register a class that ships with mloda, so the drop is permanent and unfixable for them"
        )

    def test_warn_mode_does_not_flag_a_core_rooted_framework(self, caplog: pytest.LogCaptureFixture) -> None:
        _loaded_registry()
        collector = PluginCollector().set_strict_mode("warn")

        frameworks: set[type[ComputeFramework]] = {_CoreRootedFramework, PythonDictFramework}
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(frameworks, collector)

        message = " ".join(
            record.getMessage()
            for record in caplog.records
            if record.name == LOGGER_NAME and "ComputeFrameworks not registered" in record.getMessage()
        )
        assert _qualid(_CoreRootedFramework) not in message, (
            "warn mode must not report a first-party ComputeFramework as unregistered: like every bundled "
            "framework, the warning is permanent and unfixable for users"
        )
