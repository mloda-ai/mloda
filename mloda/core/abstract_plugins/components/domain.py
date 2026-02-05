from __future__ import annotations
from typing import Any


class Domain:
    """Represents a domain for isolating features across business contexts.

    Domains enable data isolation between different contexts (Sales, Finance, Test, etc.).
    The framework matches feature domains to feature group domains for resolution.

    Definition:
        - Feature: via `domain` parameter or `options={"domain": "..."}`
        - FeatureGroup: via `get_domain()` classmethod (default: "default_domain")

    Propagation:
        When a parent feature has a domain, child features inherit it automatically.
        You can override this by setting an explicit domain on each dependent feature.

        +------------------------------------------+---------------+------------------+
        | Child Definition                         | Parent Domain | Result           |
        +------------------------------------------+---------------+------------------+
        | "child" (string)                         | "Sales"       | Inherits "Sales" |
        | Feature("child")                         | "Sales"       | Inherits "Sales" |
        | Feature("child", domain="Finance")       | "Sales"       | Keeps "Finance"  |
        | Any                                      | None          | No domain        |
        +------------------------------------------+---------------+------------------+

    Validation:
        IdentifyFeatureGroupClass ensures at least one feature group matches the feature's domain.
        If a feature has no domain and multiple groups match, an error is raised.
    """

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def get_default_domain(cls) -> Domain:
        """
        No specified domain leads to default domain.
        """
        return Domain("default_domain")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Domain):
            raise ValueError(f"Cannot compare Domain with {type(other)}")
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
