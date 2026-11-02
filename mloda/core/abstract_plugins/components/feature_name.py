from __future__ import annotations


class FeatureName(str):
    def __new__(cls, name: str) -> FeatureName:
        if isinstance(name, FeatureName):
            return name
        return super().__new__(cls, name)
