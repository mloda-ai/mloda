"""Helper for the registry-isolation tests (#845): defines a FeatureGroup subclass OUTSIDE a test module.

A class built here carries THIS module's ``__module__``, which is what the reclaim gate filters on.
"""

from __future__ import annotations

from mloda.core.abstract_plugins.feature_group import FeatureGroup


def define_helper_subclass() -> str:
    """Define a FeatureGroup subclass here and return only its name; returning the class would pin it."""

    class HelperMadeRegistryProbe845rFeatureGroup(FeatureGroup):
        pass

    return HelperMadeRegistryProbe845rFeatureGroup.__name__
