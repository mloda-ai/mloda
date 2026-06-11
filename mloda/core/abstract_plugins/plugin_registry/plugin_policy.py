"""Deployment-scoped governance policy for plugin registration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ApprovalStatus(str, Enum):
    """Review state of a registered plugin."""

    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class PluginPolicyViolationError(Exception):
    """Raised when a manual registration violates the installed plugin policy."""


@dataclass(frozen=True)
class PluginPolicy:
    """Deny-first registration policy evaluated per (key, module, approval)."""

    allowed_module_prefixes: tuple[str, ...] | None = None
    denied_module_prefixes: tuple[str, ...] = ()
    allowed_keys: tuple[str, ...] = ()
    denied_keys: tuple[str, ...] = ()
    require_approval: bool = False

    def allows(self, key: str, module: str, approval: ApprovalStatus) -> bool:
        """Return True if the plugin may register under this policy."""
        if key in self.denied_keys:
            return False
        if any(module.startswith(prefix) for prefix in self.denied_module_prefixes):
            return False
        if key not in self.allowed_keys:
            if self.allowed_module_prefixes is not None and not any(
                module.startswith(prefix) for prefix in self.allowed_module_prefixes
            ):
                return False
        if self.require_approval and approval is not ApprovalStatus.APPROVED:
            return False
        return True
