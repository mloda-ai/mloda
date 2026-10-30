"""Spy that counts walks of a class's subclass tree starting at a given root."""

from typing import Any

import pytest


def count_root_walks(monkeypatch: pytest.MonkeyPatch, root: type[Any]) -> list[int]:
    """Install a `__subclasses__` spy on `root`; the returned list gains one entry per call made on `root` itself."""
    walks: list[int] = []

    def spy(cls: type) -> list[type]:
        if cls is root:
            walks.append(1)
        return type.__subclasses__(cls)

    monkeypatch.setattr(root, "__subclasses__", classmethod(spy))
    return walks
