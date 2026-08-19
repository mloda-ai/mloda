"""An in_features value the matcher cannot resolve is a plain non-match at the resolution seam.

The probe matches by options, not by name: a name that identifies the group carries its own sources and
the option is never consulted (#944). The probe class is dropped under gc.collect() before any assert,
so no failing assert pins it (tests/conftest.py).
"""

from __future__ import annotations

import gc
from typing import Any, Optional

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
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


IN_FEATURES_FEATURE_884 = "in_features_probe_884"
IN_FEATURES_CLASS_NAME_884 = "InFeaturesMixinFG884"
RESOLUTION_ERROR_NAME = "FeatureResolutionError"
MATCHER_ERROR_STAGE = "matcher_error"


class InFeaturesFw884(ComputeFramework):
    """Dummy compute framework for the unresolvable-in_features tests."""


def _make_in_features_mixin_fg() -> type[FeatureGroup]:
    """A mixin candidate matched by its options, so only the in_features gate can still reject it."""
    gc.collect()  # class objects are cyclic: collect leftovers before defining a twin

    class InFeaturesMixinFG884(FeatureChainParserMixin, FeatureGroup):
        PREFIX_PATTERN = r".*__(?P<operation>\w+)_infeat884$"
        PROPERTY_MAPPING = {"operation": property_spec("operation", allowed_values=("op1",), context=True)}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {InFeaturesFw884}

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return InFeaturesMixinFG884


def _resolve(in_features: Any) -> tuple[Optional[str], tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    """Evaluate one feature carrying that value: (error type, winner names, (class, stage, reason) per elimination)."""
    feature_group = _make_in_features_mixin_fg()
    options = Options(context={"operation": "op1", DefaultOptionKeys.in_features: in_features})
    feature = Feature(IN_FEATURES_FEATURE_884, options)
    plugins: FeatureGroupEnvironmentMapping = {feature_group: {InFeaturesFw884}}
    try:
        error_type: Optional[str] = None
        result: Optional[EvaluationResult] = evaluate_or_raise(feature, plugins, None)
    except FeatureResolutionError as exc:
        error_type, result = type(exc).__name__, exc.result
    except Exception as exc:  # noqa: BLE001  (an untyped escape is a fact this test wants to report)
        error_type, result = type(exc).__name__, None
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


class TestUnresolvableInFeaturesIsAPlainNonMatch:
    def test_falsy_in_features_records_no_matcher_error(self) -> None:
        """An empty string is a value the matcher cannot resolve, so it is a non-match, not a raise to contain."""
        error_type, identified, eliminations = _resolve("")

        assert error_type == RESOLUTION_ERROR_NAME, f"the failure must stay the typed resolution error: {error_type}"
        assert identified == (), f"nothing may win this resolution, got: {identified}"
        matcher_errors = [entry for entry in eliminations if entry[1] == MATCHER_ERROR_STAGE]
        assert matcher_errors == [], f"a non-match must not be recorded as a matcher defect, got: {matcher_errors}"

    def test_truthy_uncountable_in_features_records_nothing(self) -> None:
        """Contrast case, passing today: a value the matcher cannot count records no elimination at all."""
        error_type, identified, eliminations = _resolve({"a": 1})

        assert error_type == RESOLUTION_ERROR_NAME
        assert identified == ()
        assert eliminations == (), f"an uncountable value is a silent non-match, got: {eliminations}"

    def test_both_unresolvable_shapes_resolve_identically(self) -> None:
        falsy = _resolve("")
        truthy = _resolve({"a": 1})

        assert falsy == truthy, f"the two shapes must be one behavior: {falsy} vs {truthy}"

    def test_resolvable_in_features_still_identifies_the_candidate(self) -> None:
        """Sanity pin: the candidate is really reachable, so the assertions above are not vacuous."""
        error_type, identified, eliminations = _resolve(["a", "b"])

        assert error_type is None, f"a resolvable value must still resolve, got: {error_type}"
        assert identified == (IN_FEATURES_CLASS_NAME_884,)
        assert eliminations == ()
