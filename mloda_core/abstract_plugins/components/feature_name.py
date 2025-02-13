from __future__ import annotations
from typing import Any


class FeatureName:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FeatureName):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        raise TypeError(f"Cannot compare FeatureName with {type(other)}.")

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def replace(self, old: str, new: str = "") -> None:
        self.name = self.name.replace(old, new)

    def __contains__(self, item: str) -> bool:
        return item in self.name

    def get_name(self) -> str:
        return self.name
