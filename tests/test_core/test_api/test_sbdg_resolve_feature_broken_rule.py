"""resolve_feature's never-raises + fail-closed contract against a broken compute_framework_rule.

resolve_feature builds accessible_plugins = {fg: {available declared frameworks}} via PreFilterPlugins,
then delegates matching to IdentifyFeatureGroupClass.evaluate(...). Building that environment is
fail-closed (#790): a FeatureGroup whose compute_framework_definition() raises aborts the build, exactly
as it does for the engine. resolve_feature keeps its never-raising contract by projecting that provider
failure into ResolvedFeature.error. Matching is never reached, so the broken group is not a candidate.

Engine/debug parity for this failure is pinned in
tests/test_core/test_resolution_parity/test_environment_build_failure_parity.py.

The broken class is scoped per test (del + gc.collect(), same pattern as
test_plugin_docs.py) so it does not leak into the session-wide subclass registry.
"""

import gc
from typing import Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.steward import ResolvedFeature, resolve_feature


SBDG_BROKEN_RULE_FEATURE = "sbdg_broken_rule_feature"
SBDG_BROKEN_RULE_MESSAGE = "sbdg rule exploded"


def _make_broken_rule_fg() -> type[FeatureGroup]:
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class SbdgBrokenRuleFG(FeatureGroup):
        """Matches a unique name but its compute_framework_rule raises when consulted."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise RuntimeError(SBDG_BROKEN_RULE_MESSAGE)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return str(feature_name) == SBDG_BROKEN_RULE_FEATURE

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return SbdgBrokenRuleFG


class TestSbdgResolveFeatureBrokenRule:
    """resolve_feature never raises, and a broken-rule group fails the environment build closed."""

    def test_resolve_feature_does_not_propagate_the_rule_exception(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            assert isinstance(result, ResolvedFeature)
            del result
        finally:
            del broken_fg
            gc.collect()

    def test_broken_rule_group_aborts_the_build_and_is_not_a_candidate(self) -> None:
        broken_fg = _make_broken_rule_fg()
        # Read the result into plain strings inside the scope: a retained ResolvedFeature (or a failing
        # assert's frame) would pin the broken class past the del below and break unrelated tests.
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            feature_name = result.feature_name
            winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
            error = result.error
            candidate_names = [candidate.get_class_name() for candidate in result.candidates]
            del result
        finally:
            del broken_fg
            gc.collect()

        assert feature_name == SBDG_BROKEN_RULE_FEATURE
        # Its rule raises while the environment is built, so the build aborts: no winner, no candidates.
        assert winner_name is None
        assert error is not None
        # The error names the real provider failure instead of a generic no-match. The exact format is
        # pinned once, in test_environment_build_failure_parity.py.
        assert SBDG_BROKEN_RULE_MESSAGE in error
        assert candidate_names == []
