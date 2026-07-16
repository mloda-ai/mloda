from __future__ import annotations

import inspect
import logging
import sys
from copy import deepcopy
from typing import Any, Optional, cast

from mloda.core.abstract_plugins.components.base_feature_group_version import (
    SOURCE_INTROSPECTION_ERRORS,
    BaseFeatureGroupVersion,
)
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import (
    PluginCollector,
    strict_mode_from_env,
)
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import _module_matches_prefix
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.components.utils import get_all_subclasses, safe_field


logger = logging.getLogger(__name__)

_warned_unregistered: set[str] = set()


FeatureGroupEnvironmentMapping = dict[type[FeatureGroup], set[type[ComputeFramework]]]


def registry_for(plugin_collector: PluginCollector | None) -> PluginRegistry:
    """Return the collector's injected registry when set, else the default registry."""
    if plugin_collector is not None and plugin_collector.registry is not None:
        return plugin_collector.registry
    return PluginRegistry.default()


# The wheel owns both roots (pyproject packages: mloda*, mloda_plugins*), minus the namespaces below that
# are reserved for third-party distributions.
_FIRST_PARTY_ROOTS = frozenset({"mloda", "mloda_plugins"})

# ``mloda`` is a shared PEP 420 namespace: pyproject keeps these out of the wheel and other distributions
# (mloda-registry) ship into them ([tool.setuptools.packages.find] exclude).
_THIRD_PARTY_NAMESPACES = frozenset({"mloda.community", "mloda.enterprise", "mloda.registry", "mloda.testing"})


def _is_bundled_plugin(cls: type[Any]) -> bool:
    """True for classes shipped in the mloda wheel (package root ``mloda`` or ``mloda_plugins``).

    ComputeFrameworks are the only kind with this bypass. FeatureGroups and Extenders cannot get one:
    registry membership is their sole strict-mode gate, and bundled ones already earn an entry by being
    loaded (``PluginLoader`` scans ``mloda_plugins``) or listed in ``CORE_PLUGIN_MODULES``. Exempting them
    by package root instead would gut the gate the registry exists to provide; the first-party registration
    guard test is what keeps that list honest.

    ``_THIRD_PARTY_NAMESPACES`` is excluded: those are shared-namespace third-party packages, not in the
    wheel, so they stay registry-gated.
    """
    module = cls.__module__
    if module.split(".", 1)[0] not in _FIRST_PARTY_ROOTS:
        return False
    return not any(_module_matches_prefix(module, namespace) for namespace in _THIRD_PARTY_NAMESPACES)


def filter_extenders_by_strict_mode(
    extenders: set[Extender] | None,
    plugin_collector: PluginCollector | None,
) -> set[Extender] | None:
    """Filter function-extender instances by the registry strict mode."""
    if extenders is None:
        return extenders

    strict_mode = plugin_collector.strict_mode if plugin_collector is not None else strict_mode_from_env()
    if strict_mode == "off":
        return extenders

    registered = registry_for(plugin_collector).registered_classes()
    unregistered = {extender for extender in extenders if type(extender) not in registered}
    unregistered_names = sorted({f"{type(e).__module__}:{type(e).__qualname__}" for e in unregistered})

    if strict_mode == "warn":
        new_names = [name for name in unregistered_names if name not in _warned_unregistered]
        if new_names:
            _warned_unregistered.update(new_names)
            logger.warning("Extenders not registered in the plugin registry: %s.", ", ".join(new_names))
        return extenders

    if unregistered:
        logger.warning(
            "Strict mode dropped Extenders not registered in the plugin registry: %s.",
            ", ".join(unregistered_names),
        )
    return extenders - unregistered


class RedefinitionConflictError(ValueError):
    """Raised by dedup when same-key FG classes differ in source.

    Carries the full list of conflicting classes via ``.conflicts`` so callers
    (e.g. ``resolve_feature``) can populate ``ResolvedFeature.candidates``
    instead of leaving it empty. Subclasses ``ValueError`` so existing
    ``except ValueError:`` callers stay compatible.
    """

    def __init__(self, message: str, conflicts: list[type[FeatureGroup]]) -> None:
        super().__init__(message)
        self.conflicts = conflicts


def _running_in_zmq_shell() -> bool:
    """Detect IPython kernel (Jupyter) environment for the kernel-restart hint."""
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]
    except ImportError:
        return False

    ipython_instance = get_ipython()  # type: ignore[no-untyped-call]
    if ipython_instance is None:
        return False
    return bool(ipython_instance.__class__.__name__ == "ZMQInteractiveShell")


def _safe_class_source_hash(cls: type[FeatureGroup]) -> Optional[str]:
    """Return source hash for a FeatureGroup subclass or None if unavailable.

    ``inspect.getsource`` raises ``OSError`` (no source backing) or ``TypeError``
    (built-in class) for classes built dynamically via ``type()``. Both leave the
    class without a stable source hash, so we return ``None``.
    """
    return safe_field(
        lambda: BaseFeatureGroupVersion.class_source_hash(cls), None, catching=SOURCE_INTROSPECTION_ERRORS
    )


def dedup_feature_group_subclasses(
    classes: set[type[FeatureGroup]],
    allow_redefinition: bool = False,
) -> set[type[FeatureGroup]]:
    """Deduplicate FeatureGroup subclasses sharing the same ``(module, qualname)``.

    Long-lived namespaces (Jupyter notebooks, ``importlib.reload``) accumulate stale
    class objects in ``FeatureGroup.__subclasses__()``. This helper collapses
    identical-content duplicates silently and raises a clear ``ValueError`` when
    duplicates differ in source (override via ``allow_redefinition=True``).
    """
    grouped: dict[tuple[str, str], list[type[FeatureGroup]]] = {}
    for cls in classes:
        key = (cls.__module__, cls.__qualname__)
        grouped.setdefault(key, []).append(cls)

    survivors: set[type[FeatureGroup]] = set()
    conflicts: list[tuple[tuple[str, str], list[tuple[type[FeatureGroup], str]]]] = []

    for key, members in grouped.items():
        if len(members) == 1:
            survivors.add(members[0])
            continue

        # Factory pattern: groups whose members are not bound under their name in
        # any module namespace are produced by factories or held only via
        # closures. They are intentionally distinct objects, so preserve all.
        if not _any_live_in_module(members):
            logger.debug(
                "dedup_feature_group_subclasses: preserving group %s (no member bound in its module)",
                key,
            )
            survivors.update(members)
            continue

        # Live-descendants check: preserve the whole group if any member has
        # subclasses already alive in the FG tree.
        has_live_descendants = any(len(get_all_subclasses(member)) > 0 for member in members)
        if has_live_descendants:
            logger.debug("dedup_feature_group_subclasses: preserving group %s due to live descendants", key)
            survivors.update(members)
            continue

        hashes = [_safe_class_source_hash(m) for m in members]
        if any(h is None for h in hashes):
            logger.debug("dedup_feature_group_subclasses: preserving group %s (source unavailable)", key)
            survivors.update(members)
            continue

        hashes_str: list[str] = cast(list[str], hashes)
        unique_hashes = set(hashes_str)
        if len(unique_hashes) == 1:
            survivor = _pick_survivor(members)
            logger.debug(
                "dedup_feature_group_subclasses: collapsed %d identical members of %s",
                len(members),
                key,
            )
            survivors.add(survivor)
            continue

        if allow_redefinition:
            survivor = _pick_survivor(members)
            logger.debug(
                "dedup_feature_group_subclasses: kept newest of %d differing members of %s (allow_redefinition=True)",
                len(members),
                key,
            )
            survivors.add(survivor)
            continue

        conflicts.append((key, list(zip(members, hashes_str))))

    if conflicts:
        all_conflicting_classes = [_cls for _, hashed in conflicts for _cls, _ in hashed]
        raise RedefinitionConflictError(_build_conflict_error(conflicts), all_conflicting_classes)

    return survivors


def _pick_survivor(members: list[type[FeatureGroup]]) -> type[FeatureGroup]:
    """Pick the live-in-module class.

    Invariant: ``_pick_survivor`` is only called after ``_any_live_in_module(members)``
    returned True at the dedup call site, so at least one member satisfies
    ``_is_live_in_module``. And ``_is_live_in_module`` checks
    ``getattr(module, cls.__name__, None) is cls``, which is single-valued: at most
    one member can match because the module attribute resolves to exactly one
    class object. The for-loop is therefore fully deterministic — set-iteration
    order does not affect which class is returned.

    The ``return members[-1]`` fallback is only theoretically reachable in
    pathological multi-thread states where ``sys.modules`` is mutated between
    ``_any_live_in_module`` and this call.
    """
    for cls in members:
        if _is_live_in_module(cls):
            return cls
    return members[-1]


def _is_live_in_module(cls: type[FeatureGroup]) -> bool:
    """True if ``cls`` is currently bound under its name in its own module."""
    module = sys.modules.get(cls.__module__)
    if module is None:
        return False
    return getattr(module, cls.__name__, None) is cls


def _any_live_in_module(members: list[type[FeatureGroup]]) -> bool:
    """True if any class is currently bound under its name in its module."""
    return any(_is_live_in_module(cls) for cls in members)


def _cell_label(cls: type[FeatureGroup]) -> Optional[str]:
    """Return the first synthetic ``<...>`` filename a class's methods live in.

    Returns ``None`` for classes whose methods all live in real source files.
    Mirrors the synthetic-filename detection in ``_linecache_source_for_class``
    but stops at the first match — sufficient for "where was this class defined"
    in the redef-conflict error message.
    """
    for value in cls.__dict__.values():
        func = value.__func__ if hasattr(value, "__func__") else value
        code = getattr(func, "__code__", None)
        if code is None:
            continue
        co_filename = code.co_filename
        if co_filename.startswith("<") and co_filename.endswith(">"):
            return str(co_filename)
    return None


def _build_conflict_error(
    conflicts: list[tuple[tuple[str, str], list[tuple[type[FeatureGroup], str]]]],
) -> str:
    lines = ["FeatureGroup redefined with different source code:"]
    for (module, qualname), hashed in conflicts:
        for _cls, source_hash in hashed:
            cell = _cell_label(_cls)
            location = f"{module} {cell}" if cell else module
            lines.append(f"  - {qualname} ({location}) source hash {source_hash[:8]}")
    lines.append(
        "Set PluginCollector(...).set_allow_redefinition() to keep only the most "
        "recently defined version of each class."
    )
    if _running_in_zmq_shell():
        lines.append("If you are running this in a notebook, restart the kernel to clear stale class definitions.")
    return "\n".join(lines)


class PreFilterPlugins:
    def __init__(
        self,
        compute_frameworks: set[type[ComputeFramework]],
        plugin_collector: Optional[PluginCollector] = None,
    ) -> None:
        feature_groups = self._set_feature_groups(plugin_collector)
        compute_frameworks = self._set_compute_frameworks(compute_frameworks, plugin_collector)

        self.accessible_plugins = self.resolve_feature_group_compute_framework_limitations(
            feature_groups, compute_frameworks
        )

    def get_accessible_plugins(self) -> FeatureGroupEnvironmentMapping:
        return self.accessible_plugins

    def _set_feature_groups(self, plugin_collector: Optional[PluginCollector] = None) -> set[type[FeatureGroup]]:
        accessible_feature_groups = self.get_featuregroup_subclasses()

        if plugin_collector:
            for accessible_fg in deepcopy(accessible_feature_groups):
                if not plugin_collector.applicable_feature_group_class(accessible_fg):
                    accessible_feature_groups.remove(accessible_fg)

        allow_redefinition = plugin_collector.allow_redefinition if plugin_collector is not None else False
        strict_mode = plugin_collector.strict_mode if plugin_collector is not None else strict_mode_from_env()

        registered: set[type[Any]] = set()
        if strict_mode != "off":
            registered = registry_for(plugin_collector).registered_classes()

        if strict_mode == "strict":
            before_strict = accessible_feature_groups
            accessible_feature_groups = {
                fg for fg in accessible_feature_groups if fg in registered or inspect.isabstract(fg)
            }
            if plugin_collector is not None:
                dropped_enabled = sorted(
                    f"{fg.__module__}:{fg.__qualname__}"
                    for fg in before_strict - accessible_feature_groups
                    if fg in plugin_collector.enabled_feature_group_classes
                )
                if dropped_enabled:
                    logger.warning(
                        "Explicitly enabled FeatureGroups were dropped by strict mode because they are "
                        "not registered in the plugin registry: %s.",
                        ", ".join(dropped_enabled),
                    )
            had_concrete = any(not inspect.isabstract(fg) for fg in before_strict)
            has_concrete = any(not inspect.isabstract(fg) for fg in accessible_feature_groups)
            if had_concrete and not has_concrete:
                raise ValueError(
                    "Strict mode filtered out all FeatureGroups: none of the accessible FeatureGroups "
                    "are registered in the plugin registry. Register them via mloda.user.register_plugin() or "
                    "relax MLODA_PLUGIN_REGISTRY_STRICT to disable strict mode."
                )

        accessible_feature_groups = dedup_feature_group_subclasses(
            accessible_feature_groups, allow_redefinition=allow_redefinition
        )

        if strict_mode == "warn":
            unregistered = sorted(
                f"{fg.__module__}:{fg.__qualname__}"
                for fg in accessible_feature_groups
                if fg not in registered
                and not inspect.isabstract(fg)
                and f"{fg.__module__}:{fg.__qualname__}" not in _warned_unregistered
            )
            if unregistered:
                _warned_unregistered.update(unregistered)
                logger.warning(
                    "FeatureGroups not registered in the plugin registry: %s.",
                    ", ".join(unregistered),
                )

        if len(accessible_feature_groups) == 0:
            raise ValueError("No feature groups are loaded. Did you call PluginLoader.all()?")
        return accessible_feature_groups

    def _set_compute_frameworks(
        self,
        compute_frameworks: set[type[ComputeFramework]],
        plugin_collector: Optional[PluginCollector] = None,
    ) -> set[type[ComputeFramework]]:
        compute_frameworks = compute_frameworks.intersection(self.get_cfw_subclasses())

        strict_mode = plugin_collector.strict_mode if plugin_collector is not None else strict_mode_from_env()
        if strict_mode == "off":
            return compute_frameworks

        registered = registry_for(plugin_collector).registered_classes()
        # Bundled plugin frameworks ship with mloda, like abstract classes: never filtered or flagged.
        unregistered = {
            cfw
            for cfw in compute_frameworks
            if cfw not in registered and not inspect.isabstract(cfw) and not _is_bundled_plugin(cfw)
        }
        unregistered_names = sorted(f"{cfw.__module__}:{cfw.__qualname__}" for cfw in unregistered)

        if strict_mode == "warn":
            new_names = [name for name in unregistered_names if name not in _warned_unregistered]
            if new_names:
                _warned_unregistered.update(new_names)
                logger.warning("ComputeFrameworks not registered in the plugin registry: %s.", ", ".join(new_names))
            return compute_frameworks

        surviving = compute_frameworks - unregistered
        had_concrete = any(not inspect.isabstract(cfw) for cfw in compute_frameworks)
        has_concrete = any(not inspect.isabstract(cfw) for cfw in surviving)
        if had_concrete and not has_concrete:
            raise ValueError(
                "Strict mode filtered out all ComputeFrameworks: none of the requested ComputeFrameworks "
                "are registered in the plugin registry. Register them via mloda.user.register_plugin() or "
                "relax MLODA_PLUGIN_REGISTRY_STRICT to disable strict mode."
            )
        return surviving

    def resolve_feature_group_compute_framework_limitations(
        self,
        feature_groups: set[type[FeatureGroup]],
        compute_frameworks: set[type[ComputeFramework]],
    ) -> FeatureGroupEnvironmentMapping:
        # Fail closed: a provider that raises while declaring its frameworks aborts the build, raw and unwrapped.
        accessible_plugins: FeatureGroupEnvironmentMapping = {}
        for feature_group in feature_groups:
            definition = feature_group.compute_framework_definition()
            new_set_of_compute_frameworks = {cp_fg for cp_fg in definition if cp_fg in compute_frameworks}
            accessible_plugins[feature_group] = new_set_of_compute_frameworks

        return accessible_plugins

    @staticmethod
    def get_cfw_subclasses() -> set[type[ComputeFramework]]:
        all_subclasses = get_all_subclasses(ComputeFramework)
        available_subclasses = {cls for cls in all_subclasses if cls.is_available()}
        return available_subclasses

    @staticmethod
    def get_featuregroup_subclasses() -> set[type[FeatureGroup]]:
        return get_all_subclasses(FeatureGroup)
