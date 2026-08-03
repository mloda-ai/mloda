"""Issue #884: an in_features value the matcher cannot resolve is a plain non-match at the resolution seam.

A falsy value (``""``, ``0``, ``False``, ``{}``) makes ``Options.get_in_features`` raise a ValueError, which the mixin
now catches and answers False to, so nothing reaches the engine and no ``matcher_error`` near-miss is recorded. Before
the fix it escaped as a raise. A truthy value the matcher cannot count (``{"a": 1}``) was always a plain non-match, so
the two shapes must resolve identically.

Assertions read structured facts off the EvaluationResult, per the seam's own docstring. The probe class lives inside
a factory and is dropped before any assert runs, so a failing assert never pins a throwaway FeatureGroup into its
traceback and trips the no-leak fixture in tests/conftest.py.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Optional

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.core.prepare.resolution_types import EvaluationResult
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


IN_FEATURES_FEATURE_884 = "src__op1_infeat884"
IN_FEATURES_CLASS_NAME_884 = "InFeaturesMixinFG884"
RESOLUTION_ERROR_NAME = "FeatureResolutionError"
MATCHER_ERROR_STAGE = "matcher_error"


class InFeaturesFw884(ComputeFramework):
    """Dummy compute framework for the unresolvable-in_features tests."""


def _make_in_features_mixin_fg() -> type[FeatureGroup]:
    """A mixin candidate matched by its own name, so only the in_features gate can still reject it."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class InFeaturesMixinFG884(FeatureChainParserMixin, FeatureGroup):
        """Matches its own name pattern and declares one compute framework."""

        PREFIX_PATTERN = r".*__(?P<operation>\w+)_infeat884$"
        PROPERTY_MAPPING = {
            "operation": property_spec("operation carried by the name", allowed_values=("op1",), context=True),
        }

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {InFeaturesFw884}

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return InFeaturesMixinFG884


@dataclass(frozen=True)
class _ResolutionSnapshot:
    """Plain-data readout of one evaluation. Holds no class and no exception object."""

    error_type: Optional[str]
    identified_names: tuple[str, ...]
    eliminations: tuple[tuple[str, str, str], ...]  # (class name, stage, reason), sorted


def _resolve(in_features: Any) -> _ResolutionSnapshot:
    """Evaluate one feature carrying that in_features value and read the outcome out as plain data."""
    feature_group = _make_in_features_mixin_fg()
    try:
        feature = Feature(IN_FEATURES_FEATURE_884, Options(context={DefaultOptionKeys.in_features: in_features}))
        plugins: FeatureGroupEnvironmentMapping = {feature_group: {InFeaturesFw884}}
        error_type: Optional[str] = None
        result: Optional[EvaluationResult] = None
        try:
            result = evaluate_or_raise(feature, plugins, None)
        except FeatureResolutionError as exc:
            error_type = type(exc).__name__
            result = exc.result
        except Exception as exc:  # noqa: BLE001  (an untyped escape is a fact this test wants to report)
            error_type = type(exc).__name__
        snapshot = _ResolutionSnapshot(
            error_type=error_type,
            identified_names=()
            if result is None
            else tuple(sorted(group.get_class_name() for group in result.identified)),
            eliminations=()
            if result is None
            else tuple(
                sorted(
                    (group.get_class_name(), str(elimination.stage), str(elimination.reason))
                    for group, elimination in result.eliminations.items()
                )
            ),
        )
        del result, plugins, feature
        return snapshot
    finally:
        del feature_group
        gc.collect()


class TestUnresolvableInFeaturesIsAPlainNonMatch:
    """A candidate that cannot resolve the requested in_features simply loses; it is not a broken matcher."""

    def test_falsy_in_features_records_no_matcher_error(self) -> None:
        """An empty string is a value the matcher cannot resolve, so it is a non-match, not a raise to contain."""
        snapshot = _resolve("")

        assert snapshot.error_type == RESOLUTION_ERROR_NAME, (
            f"the failure must stay the typed resolution error, got: {snapshot.error_type}"
        )
        assert snapshot.identified_names == (), f"nothing may win this resolution, got: {snapshot.identified_names}"
        matcher_errors = [entry for entry in snapshot.eliminations if entry[1] == MATCHER_ERROR_STAGE]
        assert matcher_errors == [], f"a non-match must not be recorded as a matcher defect, got: {matcher_errors}"

    def test_truthy_uncountable_in_features_records_nothing(self) -> None:
        """The contrast case, passing today: a value the matcher cannot count records no elimination at all."""
        snapshot = _resolve({"a": 1})

        assert snapshot.error_type == RESOLUTION_ERROR_NAME
        assert snapshot.identified_names == ()
        assert snapshot.eliminations == (), f"an uncountable value is a silent non-match, got: {snapshot.eliminations}"

    def test_both_unresolvable_shapes_resolve_identically(self) -> None:
        """Falsy and truthy unresolvable values are the same kind of non-match, so their outcomes must agree."""
        falsy = _resolve("")
        truthy = _resolve({"a": 1})

        assert falsy == truthy, f"the two shapes must be one behavior: {falsy} vs {truthy}"

    def test_resolvable_in_features_still_identifies_the_candidate(self) -> None:
        """Sanity pin, passes today: the candidate is really reachable, so the assertions above are not vacuous."""
        snapshot = _resolve(["a", "b"])

        assert snapshot.error_type is None, f"a resolvable value must still resolve, got: {snapshot.error_type}"
        assert snapshot.identified_names == (IN_FEATURES_CLASS_NAME_884,)
        assert snapshot.eliminations == ()
