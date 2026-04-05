from __future__ import annotations


class FeatureName(str):
    def __new__(cls, name: str) -> FeatureName:
        if isinstance(name, FeatureName):
            return name
        return super().__new__(cls, name)

    def __repr__(self) -> str:
        return f"FeatureName({super().__repr__()})"
