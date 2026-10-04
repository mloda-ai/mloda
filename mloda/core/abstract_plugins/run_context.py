from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RunContext:
    """Internal, not part of the public API.

    Per-run values every ComputeFramework carries into hooks and spawn workers; keep it picklable.
    """

    run_id: str | None = None
    plan_id: str | None = None
    started_at: datetime | None = None
    carrier: dict[str, str] | None = field(default=None, hash=False)  # a dict cannot hash; equality still compares it
    child_bootstrap: Callable[[], None] | None = None
    graceful_shutdown_timeout: float = 2.0
    tenant_id: str | None = None
    project_id: str | None = None
    principal: str | None = None
    # Plan-time owning-distribution version per module. None field: nothing resolved; None value: no owner.
    plugin_versions: Mapping[str, str | None] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        # Copy on ingest so a hook mutating the carrier never reaches the caller's dict.
        if self.carrier is not None:
            object.__setattr__(self, "carrier", dict(self.carrier))
        if self.plugin_versions is not None:
            object.__setattr__(self, "plugin_versions", dict(self.plugin_versions))
