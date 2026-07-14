"""Every wheel-owned plugin class must stay in the plugin registry, whatever package it lives in (issue #735).

Contract, pinned as a regression guard: after PluginLoader.all(), every concrete FeatureGroup, ComputeFramework
and Extender shipped in the mloda wheel holds a default-registry entry, and a wheel-owned ComputeFramework is
bundled (strict-mode exempt) regardless of which wheel-owned package it lives in. Registry membership rests on
an unwritten coupling (loader base package + CORE_PLUGIN_MODULES + the module that DEFINES a class), and #726
showed a class can leave the registry silently: ApiInputDataFeature moved to mloda.core, dropped out, tox stayed
green.

Wheel-owned is not "starts with mloda": ``mloda`` is a PEP 420 namespace shared with other distributions.
pyproject excludes ``mloda.community*``, ``mloda.enterprise*``, ``mloda.registry*`` and ``mloda.testing*`` from
the wheel and mloda-registry ships into them, so a class rooted there is third party and must stay subject to
strict mode and to the plugin policy gate. ``_is_bundled_plugin`` is the single source of truth for that
question, both in production and for this guard's own scope.

Exclusions from the guard:
- Abstract classes: infrastructure, never registered, never registry-gated (mirrors the strict-mode filter).
- ``_NOT_REGISTRY_GATED``: one explicit waiver set. Nothing else escapes, private or not: register_module_plugins
  registers underscore-prefixed classes and the strict filter exempts only abstract ones, so a private concrete
  class is registry-gated exactly like a public one.

Parallel-safety: the import sweep is cached per worker, doubles with a spoofed first-party ``__module__`` are
built in fixtures and gc-collected on teardown, and the autouse conftest fixture restores the default registry
and the warn-mode dedup set around every test, so clearing the registry here is safe. The sweep costs a few
seconds cold, hence the raised timeouts; the optional-dependency skip policy it relies on is exercised by
``tox -e core`` (tox.ini), which collects tests/test_core/ with only the test and pandas extras installed.
"""

import gc
import importlib
import inspect
import logging
import pkgutil
import sys
import types
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from mloda.core.abstract_plugins.components.input_data.api.api_input_data_feature import ApiInputDataFeature
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook, _CompositeExtender
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import (
    CORE_PLUGIN_MODULES,
    OPTIONAL_PLUGIN_DEPENDENCIES,
)
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import _module_matches_prefix
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry, register_plugin
from mloda.core.prepare.accessible_plugins import (
    _THIRD_PARTY_NAMESPACES,
    PreFilterPlugins,
    _is_bundled_plugin,
    filter_extenders_by_strict_mode,
)
from mloda.user import PluginLoader
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

PROJECT_ROOT = Path(__file__).resolve().parents[4]

LOGGER_NAME = "mloda.core.prepare.accessible_plugins"

# Package roots the mloda wheel owns; the sweep walks exactly these.
WHEEL_ROOTS: tuple[str, ...] = ("mloda", "mloda_plugins")

# Namespaces reserved for third-party distributions inside the shared ``mloda`` namespace. Kept in sync with
# the pyproject wheel excludes by test_reserved_namespaces_stay_in_sync_with_pyproject below.
RESERVED_NAMESPACES: tuple[str, ...] = ("mloda.community", "mloda.enterprise", "mloda.registry", "mloda.testing")

# The one waiver: _CompositeExtender is composed inside ComputeFramework.get_function_extender from an
# already-filtered extender set, so it never passes through the strict-mode gate (pinned below).
_NOT_REGISTRY_GATED: frozenset[type[Any]] = frozenset({_CompositeExtender})

CORE_DRIFT_MODULE = "mloda.core.first_party_guard_drift_probe"


class _GuardObserverFeatureGroup(FeatureGroup):
    """Observer with no compute_framework_rule: its accessible mapping value is the surviving framework set."""


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
    """The key PluginRegistry stores a class under."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _in_reserved_namespace(module: str) -> bool:
    """Boundary-aware: mloda.community.x is reserved, mloda.communityfoo.x is not."""
    return any(_module_matches_prefix(module, namespace) for namespace in RESERVED_NAMESPACES)


def _import_or_skip_optional(module_name: str) -> types.ModuleType | None:
    """Import a wheel-owned module. Missing optional dependencies skip it, every other error is raised."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing_root = error.name.split(".")[0] if error.name else ""
        if missing_root in OPTIONAL_PLUGIN_DEPENDENCIES:
            return None
        raise


def _import_wheel_owned_tree(package_name: str) -> None:
    """Import every wheel-owned module under a package, depth first.

    Hand-rolled instead of pkgutil.walk_packages: walk_packages imports a subpackage before we get a chance to
    skip it (so reserved third-party namespaces installed alongside mloda would be imported, with their deps),
    and with onerror=None it silently swallows an ImportError from a package __init__, which would shrink the
    candidate set to nothing and make the whole guard pass vacuously. Here a broken package raises.
    """
    package = _import_or_skip_optional(package_name)
    if package is None:
        return
    path = getattr(package, "__path__", None)
    if path is None:
        return

    for module_info in pkgutil.iter_modules(list(path), prefix=f"{package_name}."):
        if _in_reserved_namespace(module_info.name):
            continue
        if module_info.ispkg:
            _import_wheel_owned_tree(module_info.name)
        else:
            _import_or_skip_optional(module_info.name)


@cache
def _sweep_wheel_owned_packages() -> None:
    """Import every wheel-owned module once per worker, so the class set does not depend on import history."""
    for root in WHEEL_ROOTS:
        _import_wheel_owned_tree(root)


def _loaded_registry() -> PluginRegistry:
    """Default registry holding exactly what one full PluginLoader run discovers."""
    registry = PluginRegistry.default()
    registry.clear()
    PluginLoader.all()
    return registry


def _wheel_owned_plugin_classes(base: type[Any]) -> set[type[Any]]:
    """Concrete wheel-owned subclasses of a plugin base type.

    Scope comes from _is_bundled_plugin, the production definition of wheel-owned, so there is one source of
    truth; tests.* doubles and third-party namespaces fall out of scope automatically.
    """
    _sweep_wheel_owned_packages()
    return {cls for cls in get_all_subclasses(base) if _is_bundled_plugin(cls) and not inspect.isabstract(cls)}


def _unregistered_wheel_owned(base: type[Any]) -> list[str]:
    """Registry keys of wheel-owned plugin classes with no entry after a full loader run.

    Compared on the registry key, not on class identity: stale duplicate class objects accumulate in
    __subclasses__() (see dedup_feature_group_subclasses), and a duplicate must not fail this guard.
    """
    candidates = {_qualid(cls) for cls in _wheel_owned_plugin_classes(base)}
    registered = {_qualid(cls) for cls in _loaded_registry().registered_classes()}
    waived = {_qualid(cls) for cls in _NOT_REGISTRY_GATED}
    return sorted(candidates - registered - waived)


def _fix_hint(base: type[Any], unregistered: list[str]) -> str:
    return (
        f"PluginLoader.all() must register every wheel-owned {base.__name__}, but these have no registry "
        f"entry: {', '.join(unregistered)}. Strict mode drops them, warn mode names them, and registered-only "
        "listings lose them, with nothing failing today. Fix: a class defined under mloda/core/ must be listed "
        "in CORE_PLUGIN_MODULES (mloda/core/abstract_plugins/plugin_loader/plugin_loader.py); a class under "
        "mloda_plugins must live in a module the loader scans (a real module, not an __init__.py, and not a "
        "re-export: register_module_plugins only registers the module that DEFINES the class). If the class is "
        "genuinely not registry-gated, waive it explicitly in _NOT_REGISTRY_GATED with a reason."
    )


def _pyproject_reserved_namespaces() -> set[str]:
    """Namespaces pyproject keeps out of the mloda wheel, i.e. the ones reserved for other distributions."""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as file:
        data: dict[str, Any] = tomllib.load(file)
    excludes: list[str] = data["tool"]["setuptools"]["packages"]["find"]["exclude"]
    return {pattern.rstrip("*").rstrip(".") for pattern in excludes}


@pytest.fixture
def make_framework_double() -> Iterator[Callable[[str, str], type[ComputeFramework]]]:
    """Build ComputeFrameworks with a spoofed __module__, then drop them from __subclasses__ on teardown.

    They claim a first-party module with no importable source behind it, so they must not outlive the test:
    other tests in this worker walk ComputeFramework.__subclasses__().
    """
    created: list[type[ComputeFramework]] = []

    def make(module: str, name: str) -> type[ComputeFramework]:
        class _FrameworkDouble(ComputeFramework):
            """Concrete framework double."""

        _FrameworkDouble.__module__ = module
        _FrameworkDouble.__name__ = name
        _FrameworkDouble.__qualname__ = name
        created.append(_FrameworkDouble)
        return _FrameworkDouble

    yield make
    created.clear()
    gc.collect()


@pytest.fixture
def make_drift_probe() -> Iterator[Callable[[str, str], str]]:
    """Publish a concrete FeatureGroup in a synthetic wheel-owned module and return its registry key.

    Returns the key, not the class, so the test holds no strong reference; teardown drops the module and
    gc-collects, keeping the probe out of FeatureGroup.__subclasses__() for the rest of the worker's life.
    """
    created: list[type[FeatureGroup]] = []
    modules: list[str] = []

    def make(module_name: str, class_name: str) -> str:
        module = types.ModuleType(module_name)

        class _DriftProbe(FeatureGroup):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: Any,
                options: Any,
                data_access_collection: Any = None,
            ) -> bool:
                # Inert while it lives: the default matcher instantiates candidates, and this class has no
                # importable source file behind its synthetic module.
                return False

        _DriftProbe.__module__ = module_name
        _DriftProbe.__name__ = class_name
        _DriftProbe.__qualname__ = class_name
        setattr(module, class_name, _DriftProbe)
        sys.modules[module_name] = module
        created.append(_DriftProbe)
        modules.append(module_name)
        return f"{module_name}:{class_name}"

    yield make
    for module_name in modules:
        sys.modules.pop(module_name, None)
    created.clear()
    gc.collect()


@pytest.mark.timeout(30)
class TestEveryWheelOwnedPluginIsRegistered:
    """The guard itself: the import sweep costs a few seconds cold, hence the raised timeout."""

    def test_the_sweep_finds_the_known_wheel_owned_plugins(self) -> None:
        """Positive control: `unregistered == []` is vacuously true on an empty candidate set."""
        assert ApiInputDataFeature in _wheel_owned_plugin_classes(FeatureGroup), (
            "the sweep must reach FeatureGroups defined under mloda.core; it found "
            f"{len(_wheel_owned_plugin_classes(FeatureGroup))} FeatureGroups"
        )
        assert PythonDictFramework in _wheel_owned_plugin_classes(ComputeFramework), (
            "the sweep must reach dependency-free ComputeFrameworks under mloda_plugins"
        )
        # OtelExtender, the only concrete public bundled Extender, needs the otel extra, so the public Extender
        # set is legitimately empty in minimal envs (tox -e core). _CompositeExtender is dependency-free and
        # proves the Extender walk reaches classes at all.
        assert _CompositeExtender in _wheel_owned_plugin_classes(Extender), (
            "the sweep must reach Extenders; an empty Extender candidate set would make the guard vacuous"
        )

    def test_every_wheel_owned_feature_group_is_registered(self) -> None:
        unregistered = _unregistered_wheel_owned(FeatureGroup)
        assert unregistered == [], _fix_hint(FeatureGroup, unregistered)

    def test_every_wheel_owned_compute_framework_is_registered(self) -> None:
        unregistered = _unregistered_wheel_owned(ComputeFramework)
        assert unregistered == [], _fix_hint(ComputeFramework, unregistered)

    def test_every_wheel_owned_extender_is_registered(self) -> None:
        unregistered = _unregistered_wheel_owned(Extender)
        assert unregistered == [], _fix_hint(Extender, unregistered)


@pytest.mark.timeout(30)
class TestGuardDetectsCoreDrift:
    """The guard must have teeth: it fails on exactly the drift that happened in #726."""

    @pytest.mark.parametrize("class_name", ["DriftedCoreFeatureGroup", "_PrivateDriftedCoreFeatureGroup"])
    def test_guard_reports_a_core_feature_group_missing_from_core_plugin_modules(
        self,
        class_name: str,
        make_drift_probe: Callable[[str, str], str],
    ) -> None:
        """Public or private: nothing but the explicit waiver escapes the guard.

        The private case is the #735 blind spot itself: register_module_plugins registers underscore-prefixed
        classes and the strict filter exempts only abstract ones, so a private concrete FeatureGroup that drifts
        into mloda/core/ is dropped by strict mode exactly like ApiInputDataFeature was.
        """
        assert CORE_DRIFT_MODULE not in CORE_PLUGIN_MODULES, "sanity: the loader does not know the drift probe"

        probe_key = make_drift_probe(CORE_DRIFT_MODULE, class_name)
        reported = _unregistered_wheel_owned(FeatureGroup)

        assert probe_key in reported, (
            "the guard must report a concrete FeatureGroup defined under mloda.core that no loader path "
            f"registers; it reported {reported} instead. Without this, a plugin class moving out of "
            "mloda_plugins leaves the registry silently, exactly as ApiInputDataFeature did in #726."
        )


class TestCompositeExtenderExclusion:
    """_CompositeExtender is waived by name in _NOT_REGISTRY_GATED, not by a private-name rule."""

    @pytest.mark.timeout(30)
    def test_composite_extender_is_waived_and_would_otherwise_be_flagged(self) -> None:
        assert _is_bundled_plugin(_CompositeExtender), "sanity: it ships in the wheel"
        assert not inspect.isabstract(_CompositeExtender), "sanity: it is concrete"
        assert _CompositeExtender in _NOT_REGISTRY_GATED, "the exclusion must be an explicit waiver"

        registered = {_qualid(cls) for cls in _loaded_registry().registered_classes()}
        assert _qualid(_CompositeExtender) not in registered, (
            "sanity: no loader path registers _CompositeExtender, so only the waiver keeps it out of the guard"
        )
        assert _qualid(_CompositeExtender) not in _unregistered_wheel_owned(Extender), (
            "the waiver must keep _CompositeExtender out of the guard: it is internal machinery, never a "
            "registry-gated plugin"
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
            "registration guard waives it instead of demanding a registry entry for it"
        )


class TestWheelOwnedNamespaceDefinition:
    """Bundled means wheel-owned. ``mloda`` is a shared PEP 420 namespace, so the root alone does not say so."""

    def test_is_bundled_plugin_accepts_wheel_owned_roots(
        self, make_framework_double: Callable[[str, str], type[ComputeFramework]]
    ) -> None:
        core_rooted = make_framework_double("mloda.core.compute_framework.core_rooted", "_CoreRootedFramework")
        assert _is_bundled_plugin(PythonDictFramework), "sanity: an mloda_plugins framework is bundled"
        assert _is_bundled_plugin(core_rooted), (
            "mloda ships mloda.* and mloda_plugins.* in one wheel, so a framework that moves into mloda.core "
            "stays bundled"
        )

    @pytest.mark.parametrize("namespace", RESERVED_NAMESPACES)
    def test_is_bundled_plugin_rejects_reserved_third_party_namespaces(
        self, namespace: str, make_framework_double: Callable[[str, str], type[ComputeFramework]]
    ) -> None:
        """pyproject excludes these from the wheel and mloda-registry ships into them: they are third party.

        Rooting the bundled exemption at the whole ``mloda`` namespace lets a third-party ComputeFramework
        bypass strict mode and the plugin-policy gate that the registry exists to provide.
        """
        nested = make_framework_double(f"{namespace}.compute_framework.some_framework", "_ThirdPartyFramework")
        exact = make_framework_double(namespace, "_ThirdPartyRootFramework")

        assert not _is_bundled_plugin(nested), f"{namespace}.* is reserved for third-party distributions"
        assert not _is_bundled_plugin(exact), f"{namespace} itself is reserved for third-party distributions"

    def test_is_bundled_plugin_matches_namespaces_at_package_boundaries(
        self, make_framework_double: Callable[[str, str], type[ComputeFramework]]
    ) -> None:
        """Reserved namespaces match on package boundaries, like PluginPolicy's module prefixes."""
        sibling = make_framework_double("mloda.communityfoo.framework", "_SiblingFramework")
        assert _is_bundled_plugin(sibling), (
            "mloda.communityfoo is a distinct package from the reserved mloda.community, so it is wheel-owned; "
            "a raw string prefix would wrongly reserve it"
        )

        foreign_root = make_framework_double("mloda_registry.plugins.framework", "_ForeignRootFramework")
        assert not _is_bundled_plugin(foreign_root), (
            "mloda_registry is a separate distribution root, not a wheel-owned package root"
        )

        lookalike_root = make_framework_double("mloda_pluginsfoo.framework", "_LookalikeRootFramework")
        assert not _is_bundled_plugin(lookalike_root), "mloda_pluginsfoo is not the mloda_plugins package root"

    def test_reserved_namespaces_stay_in_sync_with_pyproject(self) -> None:
        """Drift guard: reserving a namespace in pyproject without teaching the filter reopens the bypass."""
        excluded = _pyproject_reserved_namespaces()

        assert set(RESERVED_NAMESPACES) == excluded, (
            f"this test module's RESERVED_NAMESPACES {sorted(RESERVED_NAMESPACES)} must equal the namespaces "
            f"pyproject keeps out of the wheel {sorted(excluded)} "
            "([tool.setuptools.packages.find] exclude)"
        )
        assert set(_THIRD_PARTY_NAMESPACES) == excluded, (
            f"_THIRD_PARTY_NAMESPACES {sorted(_THIRD_PARTY_NAMESPACES)} must equal the namespaces pyproject "
            f"keeps out of the wheel {sorted(excluded)} ([tool.setuptools.packages.find] exclude). A namespace "
            "reserved for another distribution but unknown to _is_bundled_plugin gets the bundled exemption: "
            "strict mode stops filtering it and the plugin-policy gate stops applying to it."
        )


class TestWheelOwnedComputeFrameworkIsBundledRegardlessOfPackageRoot:
    """mloda ships mloda.* and mloda_plugins.* in one wheel: bundled means either root, not just mloda_plugins."""

    def test_strict_mode_keeps_a_core_rooted_framework(
        self, make_framework_double: Callable[[str, str], type[ComputeFramework]]
    ) -> None:
        core_rooted = make_framework_double("mloda.core.compute_framework.core_rooted", "_CoreRootedFramework")
        _loaded_registry()
        register_plugin(_GuardObserverFeatureGroup)
        collector = PluginCollector().set_strict_mode("strict")

        frameworks: set[type[ComputeFramework]] = {core_rooted, PythonDictFramework}
        surviving = PreFilterPlugins(frameworks, collector).get_accessible_plugins()[_GuardObserverFeatureGroup]

        assert PythonDictFramework in surviving, "sanity: a bundled mloda_plugins framework survives strict mode"
        assert core_rooted in surviving, (
            "strict mode must not drop a wheel-owned ComputeFramework defined under mloda.core: users cannot "
            "register a class that ships with mloda, so the drop is permanent and unfixable for them"
        )

    def test_warn_mode_does_not_flag_a_core_rooted_framework(
        self,
        caplog: pytest.LogCaptureFixture,
        make_framework_double: Callable[[str, str], type[ComputeFramework]],
    ) -> None:
        core_rooted = make_framework_double("mloda.core.compute_framework.core_rooted", "_CoreRootedFramework")
        _loaded_registry()
        collector = PluginCollector().set_strict_mode("warn")

        frameworks: set[type[ComputeFramework]] = {core_rooted, PythonDictFramework}
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(frameworks, collector)

        assert _qualid(core_rooted) not in _unregistered_framework_warning(caplog), (
            "warn mode must not report a wheel-owned ComputeFramework as unregistered: like every bundled "
            "framework, the warning is permanent and unfixable for users"
        )


class TestThirdPartyComputeFrameworkIsNotBundled:
    """The inverse: a framework in a reserved namespace is third party and stays registry-gated."""

    def test_strict_mode_drops_an_unregistered_third_party_framework(
        self, make_framework_double: Callable[[str, str], type[ComputeFramework]]
    ) -> None:
        community = make_framework_double("mloda.community.compute_framework.community", "_CommunityFramework")
        _loaded_registry()
        register_plugin(_GuardObserverFeatureGroup)
        collector = PluginCollector().set_strict_mode("strict")

        frameworks: set[type[ComputeFramework]] = {community, PythonDictFramework}
        surviving = PreFilterPlugins(frameworks, collector).get_accessible_plugins()[_GuardObserverFeatureGroup]

        assert PythonDictFramework in surviving, "sanity: a bundled mloda_plugins framework survives strict mode"
        assert community not in surviving, (
            "strict mode must drop an unregistered ComputeFramework rooted in a reserved third-party namespace: "
            "mloda.community ships in another distribution, so the bundled exemption would let it bypass both "
            "strict mode and the plugin-policy gate"
        )

    def test_warn_mode_flags_an_unregistered_third_party_framework(
        self,
        caplog: pytest.LogCaptureFixture,
        make_framework_double: Callable[[str, str], type[ComputeFramework]],
    ) -> None:
        community = make_framework_double("mloda.community.compute_framework.community", "_CommunityFramework")
        _loaded_registry()
        collector = PluginCollector().set_strict_mode("warn")

        frameworks: set[type[ComputeFramework]] = {community, PythonDictFramework}
        with caplog.at_level(logging.WARNING):
            PreFilterPlugins(frameworks, collector)

        assert _qualid(community) in _unregistered_framework_warning(caplog), (
            "warn mode must report an unregistered ComputeFramework rooted in a reserved third-party namespace: "
            "its owner can and must register it, so the warning is actionable"
        )


def _unregistered_framework_warning(caplog: pytest.LogCaptureFixture) -> str:
    """The strict-mode warn-level message naming unregistered ComputeFrameworks."""
    return " ".join(
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and "ComputeFrameworks not registered" in record.getMessage()
    )
