from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FileSource:
    """Immutable, hashable descriptor for a file-backed input.

    Decouples file input from any compute framework: ``load_data`` resolves a file into
    this lightweight value object, and a per-framework transformer materializes it into
    that framework's native type.

    A caller may pass any ``Sequence[str]`` for ``columns``; it is normalized to an
    immutable ``tuple`` in ``__post_init__`` so the descriptor stays hashable.
    """

    path: str
    format: str
    columns: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
