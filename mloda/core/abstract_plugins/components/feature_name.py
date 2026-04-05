from __future__ import annotations

from enum import Enum


class FeatureName(str):
    def __new__(cls, name: str) -> FeatureName:
        if isinstance(name, FeatureName):
            return name
        if isinstance(name, Enum):
            name = name.value
        return super().__new__(cls, name)

    def __repr__(self) -> str:
        return f"FeatureName({super().__repr__()})"
