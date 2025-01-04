from __future__ import annotations
from typing import Any


class FeatureName:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FeatureName):
            raise Exception(f"Cannot compare FeatureName with {type(other)}.")
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def replace(self, old: str, new: str = "") -> None:
        self.name = self.name.replace(old, new)
