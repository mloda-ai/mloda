"""Shared builder of a name-path pattern FeatureGroup, kept under tests/mixins/ so leaks are reported."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import FeatureChainParserMixin


def make_name_path_group(default: str | None = None, with_level: bool = False) -> type[FeatureGroup]:
    """Pattern group binding `mode` from `<src>__tt_<mode>`; with_level adds an unbound required key."""
    mapping: dict[str, PropertySpec] = {}
    if default is None:
        mapping["mode"] = PropertySpec("Name-bound mode.", context=True, allowed_values=("fast", "slow"))
    else:
        mapping["mode"] = PropertySpec(
            "Name-bound mode.", context=True, allowed_values=("fast", "slow"), default=default
        )
    if with_level:
        mapping["level"] = PropertySpec("Unbound level.", context=True, allowed_values=("lo", "hi"))
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "PREFIX_PATTERN": r".*__tt_(?P<mode>fast|slow)$",
        "MIN_IN_FEATURES": 1,
        "MAX_IN_FEATURES": 1,
        "PROPERTY_MAPPING": mapping,
    }
    return type("NamePathToyGroup", (FeatureChainParserMixin, FeatureGroup), namespace)
