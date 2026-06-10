"""Explicit plugin registry for FeatureGroup, ComputeFramework, and Extender classes."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender

_PLUGIN_BASE_TYPES: tuple[type[Any], ...] = (FeatureGroup, ComputeFramework, Extender)


class PluginRegistryCollisionError(ValueError):
    """Raised when a different class is registered under an already taken key."""


@dataclass(frozen=True)
class PluginRegistryEntry:
    cls: type[Any]
    name: str
    plugin_type: type[Any]
    source_module: str
    source: str


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

    @classmethod
    def default(cls) -> PluginRegistry:
        global _default
        with _default_lock:
            if _default is None:
                _default = cls()
        return _default

    def register(
        self, cls: type[Any], *, name: str | None = None, source: str = "manual", replace: bool = False
    ) -> str:
        plugin_type = _resolve_plugin_type(cls)
        key = name if name is not None else f"{cls.__module__}:{cls.__qualname__}"
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
            source=source,
        )
        return key

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

    def snapshot(self) -> PluginRegistrySnapshot:
        return dict(self._entries)

    def restore(self, snapshot: PluginRegistrySnapshot) -> None:
        self._entries = dict(snapshot)

    def clear(self) -> None:
        self._entries.clear()


_default: PluginRegistry | None = None
_default_lock = threading.Lock()


def register(cls: type[Any], *, name: str | None = None, source: str = "manual", replace: bool = False) -> str:
    return PluginRegistry.default().register(cls, name=name, source=source, replace=replace)


def register_module_plugins(module: ModuleType, *, source: str = "loader") -> list[str]:
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
        keys.append(registry.register(obj, source=source, replace=True))
    return keys
