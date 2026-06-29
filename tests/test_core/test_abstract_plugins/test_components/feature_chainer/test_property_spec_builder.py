"""Tests for the ``property_spec`` authoring helper (issue #543).

``property_spec`` builds a conventional PROPERTY_MAPPING entry from explicit
arguments, validating the invariants that previously had to be enforced by hand:

* ``strict=True`` requires a non-empty ``allowed_values`` (a strict enum with no
  value space rejects everything),
* ``allowed_values`` without ``strict`` is a silent no-op and is rejected,
* a strict, non-``None`` ``default`` must be within the allowed set,
* a one-shot iterable for ``allowed_values`` is materialized so it survives reuse.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import property_spec


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks throwaway FeatureGroup subclasses.

    Copied from ``test_property_mapping_default_invariant.py``. The round-trip test
    defines a FeatureGroup subclass; this fixture forces a collection afterwards
    and asserts none of this module's classes linger in the registry.
    """
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


class TestPropertySpecImport:
    """The helper is exported from the public provider surface."""

    def test_property_spec_importable_from_provider(self) -> None:
        """``from mloda.provider import property_spec`` works."""
        from mloda.provider import property_spec as imported

        assert callable(imported)


class TestPropertySpecEmission:
    """A valid call emits a conventional spec dict."""

    def test_emits_conventional_spec_dict(self) -> None:
        """All conventional keys are present with the expected values."""
        spec = property_spec("desc", strict=True, allowed_values={"add": "Addition"}, default="add")

        assert spec["explanation"] == "desc"
        assert spec[DefaultOptionKeys.allowed_values] == {"add": "Addition"}
        assert spec[DefaultOptionKeys.strict_validation] is True
        assert spec[DefaultOptionKeys.context] is True
        assert spec[DefaultOptionKeys.default] == "add"


class TestPropertySpecInvariants:
    """Invalid combinations raise ``ValueError``."""

    def test_strict_without_allowed_values_raises(self) -> None:
        """``strict=True`` with no value space rejects everything, so it is illegal."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True)

    def test_allowed_values_without_strict_raises(self) -> None:
        """``allowed_values`` is never enforced without ``strict`` (silent no-op), so it is rejected."""
        with pytest.raises(ValueError):
            property_spec("d", allowed_values={"a": "A"})

    def test_strict_default_outside_allowed_set_raises(self) -> None:
        """A strict, non-``None`` default must be within the allowed set."""
        with pytest.raises(ValueError):
            property_spec("d", strict=True, allowed_values={"add": "A"}, default="mul")

    def test_strict_default_none_is_exempt(self) -> None:
        """An omitted/``None`` default is always legal under strict validation."""
        spec = property_spec("d", strict=True, allowed_values={"add": "A"})

        assert spec[DefaultOptionKeys.allowed_values] == {"add": "A"}


class TestPropertySpecIterableAllowedValues:
    """``allowed_values`` may be a plain iterable, and one-shot iterables are materialized."""

    def test_plain_iterable_accepted(self) -> None:
        """A tuple of allowed values is accepted with strict validation."""
        spec = property_spec("d", strict=True, allowed_values=("add", "sub"), default="add")

        emitted = spec[DefaultOptionKeys.allowed_values]
        assert set(emitted) == {"add", "sub"}

    def test_one_shot_generator_is_materialized(self) -> None:
        """A generator must be materialized so the emitted allowed_values is re-iterable."""
        spec = property_spec("d", strict=True, allowed_values=(x for x in ("add", "sub")), default="add")

        emitted = spec[DefaultOptionKeys.allowed_values]
        first = list(emitted)
        second = list(emitted)
        assert first == second
        assert set(first) == {"add", "sub"}


class TestPropertySpecRoundTrip:
    """A spec built by the helper drives membership validation end to end."""

    def test_round_trip_through_property_mapping(self) -> None:
        """The built entry defines without error and accepts/rejects via the parser."""

        class RoundTripFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\w]+)_op$"
            PROPERTY_MAPPING = {
                "operation_type": property_spec(
                    "op",
                    strict=True,
                    allowed_values={"add": "Addition", "sub": "Subtraction"},
                    default="add",
                )
            }

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return data

        property_mapping = RoundTripFeatureGroup.PROPERTY_MAPPING

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "add"}), property_mapping
            )
            is True
        )
        with pytest.raises(ValueError, match="mul"):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"operation_type": "mul"}), property_mapping
            )
