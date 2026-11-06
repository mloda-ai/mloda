from collections.abc import Mapping
from dataclasses import dataclass, field

from mloda.core.abstract_plugins.components.read_only_dict import _frozen_dict


@dataclass(frozen=True)
class RunSurface:
    """What a prepared session's run() accepts: api_data columns, artifact names, parameters, credentials."""

    api_data: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _frozen_dict({}))
    artifacts: frozenset[str] = frozenset()
    parameters: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: _frozen_dict({}))
    credentials: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_data", _frozen_dict(dict(self.api_data)))
        object.__setattr__(self, "parameters", _frozen_dict(dict(self.parameters)))
