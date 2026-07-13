"""Failing tests hardening the SubtypeDeclaration surface (issue #639 follow-up).

Pins: a 'supported' key naming no declared framework is a loud ValueError from
subtype_support_matrix(); supported keys are stringified on ingest; and the
declaration is hashable even with mappings set.
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureGroup, SubtypeDeclaration, property_spec


SBFIX_KEY = "sbfix_kind"
SBFIX_UNIVERSE = frozenset({"sum", "median"})
SBFIX_BOGUS_FRAMEWORK = "SbfixNoSuchFramework"


def _sbfix_noop_resolver(feature_name: str, options: Options) -> Optional[str]:
    return None


class SbfixFwAlpha(ComputeFramework):
    """Dummy compute framework for subtype-declaration hardening tests."""


class SbfixBogusSupportedFG(FeatureGroup):
    """Declaration whose 'supported' names a framework the family never declares."""

    SUBTYPES = SubtypeDeclaration(
        key=SBFIX_KEY,
        supported={SBFIX_BOGUS_FRAMEWORK: {"sum"}},
    )
    PROPERTY_MAPPING = {
        SBFIX_KEY: property_spec(
            "Operation kind with a bogus supported framework.",
            strict=True,
            allowed_values={"sum": "Sum", "median": "Median"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SbfixFwAlpha}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SbfixHealthySupportedFG(FeatureGroup):
    """Correct declaration for contrast; its matrix must stay clean."""

    SUBTYPES = SubtypeDeclaration(
        key=SBFIX_KEY,
        supported={SbfixFwAlpha.get_class_name(): {"sum"}},
    )
    PROPERTY_MAPPING = {
        SBFIX_KEY: property_spec(
            "Operation kind with a correct supported framework.",
            strict=True,
            allowed_values={"sum": "Sum", "median": "Median"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SbfixFwAlpha}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestSbfixMatrixRejectsUndeclaredSupportedFramework:
    """subtype_support_matrix() raises on a 'supported' key naming no declared framework."""

    def test_class_definition_succeeds_since_values_are_in_universe(self) -> None:
        assert SbfixBogusSupportedFG.subtype_universe() == SBFIX_UNIVERSE

    def test_matrix_raises_value_error_naming_framework_and_class(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SbfixBogusSupportedFG.subtype_support_matrix()
        message = str(exc_info.value)
        assert SBFIX_BOGUS_FRAMEWORK in message
        assert "SbfixBogusSupportedFG" in message

    def test_correct_declaration_still_produces_clean_matrix(self) -> None:
        assert SbfixHealthySupportedFG.subtype_support_matrix() == {
            SbfixFwAlpha.get_class_name(): frozenset({"sum"}),
        }


class TestSbfixSupportedKeysStringifiedOnIngest:
    """Supported keys are stringified when the declaration is constructed."""

    def test_numeric_supported_key_is_stringified(self) -> None:
        raw_supported: dict[Any, set[str]] = {123: {"2"}}
        decl = SubtypeDeclaration(universe={"2"}, resolver=_sbfix_noop_resolver, supported=raw_supported)
        assert decl.supported is not None
        assert set(decl.supported) == {"123"}
        assert decl.supported["123"] == frozenset({"2"})


class TestSbfixDeclarationHashable:
    """hash(SubtypeDeclaration(...)) works with parametric_families and supported set."""

    def _make(self, supported_values: set[str]) -> SubtypeDeclaration:
        return SubtypeDeclaration(
            universe={"a", "b"},
            resolver=_sbfix_noop_resolver,
            parametric_families={"fam": "A family"},
            supported={"SbfixFwAlpha": supported_values},
        )

    def test_hash_succeeds_with_mappings_set(self) -> None:
        decl = self._make({"a", "fam"})
        assert isinstance(hash(decl), int)

    def test_equal_inputs_are_equal_with_equal_hashes(self) -> None:
        first = self._make({"a", "fam"})
        second = self._make({"a", "fam"})
        assert first == second
        assert hash(first) == hash(second)

    def test_differing_declaration_is_not_equal(self) -> None:
        first = self._make({"a", "fam"})
        third = self._make({"b"})
        assert first != third
