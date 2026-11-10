"""Server-verified tenant/project/principal context seam.

A platform (server, portal, job runner) sets this once per run; it is never
overridable through a feature's Options.
"""

from contextvars import ContextVar
from dataclasses import dataclass

from mloda.core.abstract_plugins.components.var_scope import VarScope


@dataclass(frozen=True)
class VerifiedContext:
    """Server-verified tenant/project/principal values for the scope of a run."""

    tenant_id: str | None = None
    project_id: str | None = None
    principal: str | None = None


_current_verified_context: ContextVar[VerifiedContext | None] = ContextVar("_current_verified_context", default=None)


def current_verified_context() -> VerifiedContext | None:
    """Return the VerifiedContext active in the current verified_context() scope, else None."""
    return _current_verified_context.get()


def verified_context(
    *,
    tenant_id: str | None = None,
    project_id: str | None = None,
    principal: str | None = None,
) -> VarScope[VerifiedContext | None]:
    """Make tenant_id/project_id/principal the current_verified_context() for the scope, restoring the previous value on exit."""
    return VarScope(
        _current_verified_context,
        VerifiedContext(tenant_id=tenant_id, project_id=project_id, principal=principal),
    )
