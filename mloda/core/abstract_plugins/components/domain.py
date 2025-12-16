from __future__ import annotations
from typing import Any


class Domain:
    """
    Documentation domain:

    default value is default_domain. The purpose of this is to allow for a default domain to be used if no domain is given.
    Usecase: Testing, POCS etc. This is a tool to make life easier for the user.

    can be defined by:
    feature.domain: either by options or by domain
    feature_group.domain: returns the domain name rule for the feature group

    We validate in IdentifyFeatureGroupClass that there is atleast one feature group with the same domain as the feature.
    If the feature does not have a domain, and we have not exactly one matching feature group to the feature, we raise an error.
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
