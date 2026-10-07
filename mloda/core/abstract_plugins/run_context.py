from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mloda.core.abstract_plugins.components.read_only_dict import _frozen_dict


@dataclass(frozen=True)
class RunContext:
    """Per-run values every ComputeFramework carries into hooks and worker processes; keep it picklable.

    Exported from mloda.steward as the argument of on_run_start and on_run_complete; the other facades do not export it.
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
    structure_hash: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Copy on ingest so a hook mutating the carrier never reaches the caller's dict.
        if self.carrier is not None:
            object.__setattr__(self, "carrier", _frozen_dict(self.carrier))
        if self.plugin_versions is not None:
            object.__setattr__(self, "plugin_versions", _frozen_dict(dict(self.plugin_versions)))

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        if self.carrier is not None:
            object.__setattr__(self, "carrier", _frozen_dict(self.carrier))
        if self.plugin_versions is not None:
            object.__setattr__(self, "plugin_versions", _frozen_dict(dict(self.plugin_versions)))
