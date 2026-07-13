"""Red-phase tests: derived subtype accessor overrides must be allowed at runtime (issue #639 cleanup).

Enforcement of the derived-accessor contract moves to @final + mypy --strict only.
The runtime guard _reject_derived_accessor_overrides in FeatureGroup.__init_subclass__
is removed; defining an overriding subclass must succeed and the override takes effect.
The matrix-voiding check (subtype_support_matrix raising for a hand-written
supports_compute_framework override) stays.
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureGroup, SubtypeDeclaration, property_spec


SBRM_KEY = "sbrm_window_function"


class SbrmFwAlpha(ComputeFramework):
    """Dummy compute framework for the override-allowed tests."""


class SbrmDeclaredBaseFG(FeatureGroup):
    """Declared family used as a base for override subclasses."""

    SUBTYPES = SubtypeDeclaration(key=SBRM_KEY)
    PROPERTY_MAPPING = {
        SBRM_KEY: property_spec(
            "Window function subtype.",
            strict=True,
            allowed_values={"median": "Median", "sum": "Sum"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SbrmFwAlpha}


class TestSbrmDerivedAccessorOverrideAllowed:
    """Overriding a derived subtype accessor is a mypy-only concern; runtime must not raise."""

    def test_sbrm_canonical_subtype_override_defines_and_takes_effect(self) -> None:
        def sbrm_canonical(cls: type[FeatureGroup], subtype: str) -> str:
            return f"sbrm_canon_{subtype}"

        overriding = type(
            "SbrmCanonicalOverrideFG",
            (FeatureGroup,),
            {"canonical_subtype": classmethod(sbrm_canonical)},
        )
        assert issubclass(overriding, FeatureGroup)
        assert overriding.canonical_subtype("median") == "sbrm_canon_median"

    def test_sbrm_resolve_subtype_override_on_declared_family_takes_effect(self) -> None:
        def sbrm_resolve(cls: type[FeatureGroup], feature_name: FeatureName | str, options: Options) -> Optional[str]:
            return "sbrm_resolved"

        overriding = type(
            "SbrmResolveOverrideFG",
            (SbrmDeclaredBaseFG,),
            {"resolve_subtype": classmethod(sbrm_resolve)},
        )
        assert issubclass(overriding, SbrmDeclaredBaseFG)
        assert overriding.resolve_subtype("sbrm_any_feature", Options()) == "sbrm_resolved"


class TestSbrmMatrixVoidingCheckSurvives:
    """Contrast pin: hand-written supports_compute_framework still voids the matrix."""

    def test_sbrm_supports_hook_override_still_raises_from_matrix(self) -> None:
        def sbrm_supports(
            cls: type[FeatureGroup],
            feature_name: FeatureName | str,
            options: Options,
            compute_framework: type[ComputeFramework],
        ) -> bool:
            return True

        overriding: Any = type(
            "SbrmSupportsOverrideFG",
            (SbrmDeclaredBaseFG,),
            {"supports_compute_framework": classmethod(sbrm_supports)},
        )
        with pytest.raises(ValueError) as exc_info:
            overriding.subtype_support_matrix()
        message = str(exc_info.value)
        assert "supports_compute_framework" in message
        assert "SbrmSupportsOverrideFG" in message
