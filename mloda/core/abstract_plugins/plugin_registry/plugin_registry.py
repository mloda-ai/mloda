"""Explicit plugin registry for FeatureGroup, ComputeFramework, and Extender classes."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import (
    ApprovalStatus,
    PluginPolicy,
    PluginPolicyViolationError,
)

logger = logging.getLogger(__name__)

_PLUGIN_BASE_TYPES: tuple[type[Any], ...] = (FeatureGroup, ComputeFramework, Extender)


class PluginRegistryCollisionError(ValueError):
    """Raised when a different class is registered under an already taken key."""


class PluginSource(str, Enum):
    """Provenance of a registry entry."""

    MANUAL = "manual"
    LOADER = "loader"
    ENTRY_POINT = "entry_point"


def _normalize_source(source: PluginSource | str) -> PluginSource:
    if isinstance(source, PluginSource):
        return source
    members = {member.value: member for member in PluginSource}
    if source not in members:
        valid = ", ".join(member.value for member in PluginSource)
        raise ValueError(f"Invalid plugin source {source!r}. Valid values are: {valid}.")
    return members[source]


def _normalize_approval(approval: ApprovalStatus | str) -> ApprovalStatus:
    if isinstance(approval, ApprovalStatus):
        return approval
    members = {member.value: member for member in ApprovalStatus}
    if approval not in members:
        valid = ", ".join(member.value for member in ApprovalStatus)
        raise ValueError(f"Invalid approval status {approval!r}. Valid values are: {valid}.")
    return members[approval]


@dataclass(frozen=True)
class PluginRegistryEntry:
    cls: type[Any]
    name: str
    plugin_type: type[Any]
    source_module: str
    source: PluginSource
    owner: str | None = None
    approval: ApprovalStatus = ApprovalStatus.UNREVIEWED


PluginRegistrySnapshot = dict[str, PluginRegistryEntry]


def _resolve_plugin_type(cls: type[Any]) -> type[Any]:
    for base in _PLUGIN_BASE_TYPES:
        if isinstance(cls, type) and issubclass(cls, base):
            return base
    raise ValueError(f"{cls!r} is not a subclass of FeatureGroup, ComputeFramework, or Extender.")


class PluginRegistry:
    def __init__(self) -> None:
        # Only default() lazy init is locked; registration is expected single-threaded at import/startup time.
        self._entries: dict[str, PluginRegistryEntry] = {}
        self._policy: PluginPolicy | None = None
        self._policy_warned_keys: set[str] = set()

    @classmethod
    def default(cls) -> PluginRegistry:
        global _default
        with _default_lock:
            if _default is None:
                _default = cls()
        return _default

    @property
    def policy(self) -> PluginPolicy | None:
        return self._policy

    def set_policy(self, policy: PluginPolicy | None) -> None:
        self._policy = policy

    def register(
        self,
        cls: type[Any],
        *,
        name: str | None = None,
        source: PluginSource | str = PluginSource.MANUAL,
        replace: bool = False,
        owner: str | None = None,
        approval: ApprovalStatus | str = ApprovalStatus.UNREVIEWED,
    ) -> str | None:
        normalized_source = _normalize_source(source)
        normalized_approval = _normalize_approval(approval)
        plugin_type = _resolve_plugin_type(cls)
        key = name if name is not None else f"{cls.__module__}:{cls.__qualname__}"
        if self._policy is not None and not self._policy.allows(key, cls.__module__, normalized_approval):
            if normalized_source is PluginSource.MANUAL:
                raise PluginPolicyViolationError(f"Plugin '{key}' is denied by the installed plugin policy.")
            if key not in self._policy_warned_keys:
                self._policy_warned_keys.add(key)
                logger.warning("Plugin '%s' was denied by the installed plugin policy and not registered.", key)
            return None
        existing_key = next((k for k, entry in self._entries.items() if entry.cls is cls), None)
        if existing_key is not None and existing_key != key:
            raise PluginRegistryCollisionError(
                f"{cls!r} is already registered under key '{existing_key}'; a class holds exactly one key. "
                f"To rename, unregister('{existing_key}') first, then register under the new key."
            )
        existing = self._entries.get(key)
        if existing is not None and not replace:
            if existing.cls is cls:
                return key
            raise PluginRegistryCollisionError(
                f"Key '{key}' is already registered to {existing.cls!r}; cannot register {cls!r}. "
                "Pass replace=True to replace the existing entry."
            )
        self._entries[key] = PluginRegistryEntry(
            cls=cls,
            name=key,
            plugin_type=plugin_type,
            source_module=cls.__module__,
            source=normalized_source,
            owner=owner,
            approval=normalized_approval,
        )
        return key

    def set_approval(self, name: str, approval: ApprovalStatus | str, owner: str | None = None) -> None:
        normalized_approval = _normalize_approval(approval)
        entry = self.get_entry(name)
        new_owner = owner if owner is not None else entry.owner
        self._entries[name] = dataclasses.replace(entry, approval=normalized_approval, owner=new_owner)

    def unregister(self, name: str) -> None:
        if name not in self._entries:
            raise ValueError(f"No plugin registered under key '{name}'.")
        del self._entries[name]

    def get(self, name: str) -> type[Any] | None:
        entry = self._entries.get(name)
        return entry.cls if entry is not None else None

    def get_entry(self, name: str) -> PluginRegistryEntry:
        entry = self._entries.get(name)
        if entry is None:
            raise ValueError(f"No plugin registered under key '{name}'.")
        return entry

    def is_registered(self, cls: type[Any]) -> bool:
        return any(entry.cls is cls for entry in self._entries.values())

    def list_registered(self, plugin_type: type[Any]) -> list[type[Any]]:
        if plugin_type not in _PLUGIN_BASE_TYPES:
            raise ValueError(
                f"{plugin_type!r} is not a valid plugin base type. "
                "Valid types are: FeatureGroup, ComputeFramework, Extender."
            )
        result: list[type[Any]] = []
        seen: set[type[Any]] = set()
        for _key, entry in sorted(self._entries.items()):
            if entry.plugin_type is not plugin_type or entry.cls in seen:
                continue
            seen.add(entry.cls)
            result.append(entry.cls)
        return result

    def registered_classes(self) -> set[type[Any]]:
        return {entry.cls for entry in self._entries.values()}

    def snapshot(self) -> PluginRegistrySnapshot:
        return dict(self._entries)

    def restore(self, snapshot: PluginRegistrySnapshot) -> None:
        self._entries = dict(snapshot)

    def clear(self) -> None:
        self._entries.clear()


_default: PluginRegistry | None = None
_default_lock = threading.Lock()


def register_plugin(
    cls: type[Any],
    *,
    name: str | None = None,
    source: PluginSource | str = PluginSource.MANUAL,
    replace: bool = False,
    owner: str | None = None,
    approval: ApprovalStatus | str = ApprovalStatus.UNREVIEWED,
) -> str | None:
    return PluginRegistry.default().register(
        cls, name=name, source=source, replace=replace, owner=owner, approval=approval
    )


def register_module_plugins(module: ModuleType, *, source: PluginSource | str = PluginSource.LOADER) -> list[str]:
    """Register all plugin classes defined in a module into the default registry.

    Last-write-wins (replace=True) so importlib.reload updates entries; provenance may flip to "loader".
    """
    registry = PluginRegistry.default()
    keys: list[str] = []
    for obj in vars(module).values():
        if not isinstance(obj, type) or obj in _PLUGIN_BASE_TYPES:
            continue
        if not issubclass(obj, _PLUGIN_BASE_TYPES) or obj.__module__ != module.__name__:
            continue
        if inspect.isabstract(obj):
            continue
        key = registry.register(obj, source=source, replace=True)
        if key is not None:
            keys.append(key)
    return keys
