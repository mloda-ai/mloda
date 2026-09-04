from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunContext:
    """Per-run values every ComputeFramework carries into hooks and spawn workers; keep it picklable."""

    run_id: str | None = None
    carrier: dict[str, str] | None = field(default=None, hash=False)  # a dict cannot hash; equality still compares it
    child_bootstrap: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        # Copy on ingest so a hook mutating the carrier never reaches the caller's dict.
        if self.carrier is not None:
            object.__setattr__(self, "carrier", dict(self.carrier))
