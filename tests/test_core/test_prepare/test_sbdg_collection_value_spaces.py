"""Failing tests pinning declared_option_values against non-dict Mapping value spaces
(sbdg defect 2).

property_spec accepts any Mapping | Iterable for allowed_values. Non-Mapping iterables
(e.g. range) are normalized to a tuple, so they already work. A Mapping that is not a
plain dict (e.g. types.MappingProxyType) is kept as-is, but
FeatureGroup.declared_option_values only recognizes
isinstance(extracted, (dict, list, tuple, set, frozenset)) and silently returns an
empty value space for it. Consequences pinned here:

- SUBTYPES = SubtypeDeclaration(key=<that key>) fails class creation with
  "without an enumerable value space" although the spec is legal and enumerable.
- declared_option_values on a plain (non-SUBTYPES) class returns frozenset() instead
  of the stringified keys.

A bare str allowed_values is rejected at property_spec construction time (substring
membership trap), so no legal spec can carry a str value space; no string pin needed.
"""

from types import MappingProxyType
from typing import Any

from mloda.provider import FeatureGroup, PropertySpec, SubtypeDeclaration, property_spec


SBDG_SIZE_KEY = "sbdg_size"
SBDG_PROXY_VALUES = MappingProxyType({2: "two", 3: "three", 4: "four"})


def _build(*args: Any, **kwargs: Any) -> PropertySpec:
    """Call ``property_spec`` through an untyped seam.

    The builder's declared ``allowed_values`` type lists the container shapes; a ``range`` is one
    of the iterables its runtime materializes but its type deliberately does not name.
    """
    return property_spec(*args, **kwargs)


class SbdgProxyValuesFG(FeatureGroup):
    """Legal spec whose allowed_values is an immutable Mapping (no SUBTYPES declared)."""

    PROPERTY_MAPPING = {
        SBDG_SIZE_KEY: property_spec(
            "Size with an immutable mapping value space.",
            strict=True,
            allowed_values=SBDG_PROXY_VALUES,
        ),
    }


class TestSbdgDeclaredOptionValuesAcceptsMappingProxy:
    def test_declared_option_values_enumerates_mapping_proxy_keys(self) -> None:
        assert SbdgProxyValuesFG.declared_option_values(SBDG_SIZE_KEY) == frozenset({"2", "3", "4"})

    def test_range_value_space_is_normalized_and_enumerable(self) -> None:
        # Regression guard: property_spec materializes non-Mapping iterables to a tuple,
        # so a range-backed spec is already enumerable.
        class SbdgRangeValuesFG(FeatureGroup):
            PROPERTY_MAPPING = {
                "sbdg_range_size": _build(
                    "Size from a range.",
                    strict=True,
                    allowed_values=range(2, 5),
                ),
            }

        assert SbdgRangeValuesFG.declared_option_values("sbdg_range_size") == frozenset({"2", "3", "4"})


class TestSbdgSubtypeDeclarationOnMappingProxyKey:
    def test_class_defines_and_universe_enumerates(self) -> None:
        # Today this raises ValueError "... without an enumerable value space" at class creation.
        class SbdgProxySubtypesFG(FeatureGroup):
            PROPERTY_MAPPING = {
                SBDG_SIZE_KEY: property_spec(
                    "Size with an immutable mapping value space and SUBTYPES.",
                    strict=True,
                    allowed_values=MappingProxyType({2: "two", 3: "three", 4: "four"}),
                ),
            }
            SUBTYPES = SubtypeDeclaration(key=SBDG_SIZE_KEY)

        assert SbdgProxySubtypesFG.subtype_universe() == frozenset({"2", "3", "4"})
