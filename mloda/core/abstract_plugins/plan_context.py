from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PlanContext:
    """Per-plan identity, minted once when a session is planned; the verified identity is read at that moment."""

    plan_id: str
    tenant_id: str | None
    project_id: str | None
    principal: str | None
    created_at: datetime
