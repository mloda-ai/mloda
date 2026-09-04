"""A name whose source count violates MIN_IN_FEATURES is a near-miss, not a bare "no feature groups found".

The gate rejects at match time (#944), so the actionable in_feature-count message must reach the
resolution-failure report as an elimination. The probe class is dropped under gc.collect() before any
assert, so no failing assert pins it (tests/conftest.py).
"""

from __future__ import annotations

import gc

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.core.prepare.resolution_types import EvaluationResult
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


MIN_COUNT_CLASS_NAME = "MinInFeaturesMixinFG944"
BELOW_MIN_FEATURE = "src1__op1_mincount944"
INSIDE_RANGE_FEATURE = "src1&src2__op1_mincount944"
RESOLUTION_ERROR_NAME = "FeatureResolutionError"
VALUE_REJECTION_STAGE = "value_rejection"
BELOW_MIN_REASON = f"Feature '{BELOW_MIN_FEATURE}' requires at least 2 in_feature(s), but found 1"


class MinCountFw944(ComputeFramework):
    """Dummy compute framework for the name-source-count elimination tests."""


def _make_min_count_mixin_fg() -> type[FeatureGroup]:
    """A mixin candidate whose name identifies it, so only the in_feature count gate can reject it."""
    gc.collect()  # class objects are cyclic: collect leftovers before defining a twin

    class MinInFeaturesMixinFG944(FeatureChainParserMixin, FeatureGroup):
        PREFIX_PATTERN = r".*__(?P<operation>\w+)_mincount944$"
        MIN_IN_FEATURES = 2
        PROPERTY_MAPPING = {"operation": property_spec("operation", allowed_values=("op1",), context=True)}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MinCountFw944}

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return MinInFeaturesMixinFG944


def _resolve(feature_name: str) -> tuple[str | None, tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    """Evaluate that name: (error type, winner names, (class, stage, reason) per elimination)."""
    feature_group = _make_min_count_mixin_fg()
    feature = Feature(feature_name, Options())
    plugins: FeatureGroupEnvironmentMapping = {feature_group: {MinCountFw944}}
    error_type: str | None = None
    result: EvaluationResult | None = None
    try:
        result = evaluate_or_raise(feature, plugins, None)
    except FeatureResolutionError as exc:
        error_type, result = type(exc).__name__, exc.result
    identified: tuple[str, ...] = ()
    eliminations: tuple[tuple[str, str, str], ...] = ()
    if result is not None:
        identified = tuple(sorted(g.get_class_name() for g in result.identified))
        eliminations = tuple(
            sorted((g.get_class_name(), str(e.stage), str(e.reason)) for g, e in result.eliminations.items())
        )
    del result, plugins, feature, feature_group
    gc.collect()
    return error_type, identified, eliminations


class TestNameSourceCountBelowMinIsANearMiss:
    def test_resolution_still_fails(self) -> None:
        error_type, identified, _ = _resolve(BELOW_MIN_FEATURE)

        assert error_type == RESOLUTION_ERROR_NAME, f"the count gate must still reject, got: {error_type}"
        assert identified == (), f"nothing may win this resolution, got: {identified}"

    def test_the_in_feature_count_reason_is_reported_as_an_elimination(self) -> None:
        """The pre-gate diagnostic survives: the report names the declared MIN and the count the name carries."""
        _, _, eliminations = _resolve(BELOW_MIN_FEATURE)

        expected = ((MIN_COUNT_CLASS_NAME, VALUE_REJECTION_STAGE, BELOW_MIN_REASON),)
        assert eliminations == expected, f"the count reason must surface as a near-miss, got: {eliminations}"

    def test_a_count_inside_the_range_still_identifies_the_candidate(self) -> None:
        """Sanity pin: the candidate is really reachable, so the assertions above are not vacuous."""
        error_type, identified, eliminations = _resolve(INSIDE_RANGE_FEATURE)

        assert error_type is None, f"a count inside the range must still resolve, got: {error_type}"
        assert identified == (MIN_COUNT_CLASS_NAME,)
        assert eliminations == ()
