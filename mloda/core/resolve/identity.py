from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class PluginIdentity:
    """Stable plugin identity: defining module plus qualified class name."""

    module: str
    qualname: str

    @classmethod
    def from_class(cls, plugin_class: type[Any]) -> PluginIdentity:
        """Build the identity of a plugin class."""
        return cls(module=plugin_class.__module__, qualname=plugin_class.__qualname__)

    def render(self) -> str:
        """Render the canonical "module:qualname" string."""
        return f"{self.module}:{self.qualname}"
