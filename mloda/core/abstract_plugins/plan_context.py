from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

PlanOrigin = Literal["run_all", "stream_all", "prepare", "explain", "diagnose"]


@dataclass(frozen=True)
class PlanContext:
    """Per-plan identity, minted once when a session is planned; the verified identity is read at that moment."""

    plan_id: str
    tenant_id: str | None
    project_id: str | None
    principal: str | None
    created_at: datetime
    structure_hash: str | None = field(default=None, compare=False)
    content_hash: str | None = field(default=None, compare=False)
    origin: PlanOrigin | None = None
