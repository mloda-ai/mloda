"""resolve_feature's never-raises + fail-closed contract against a broken compute_framework_rule.

After #755 resolve_feature builds accessible_plugins = {fg: {available declared frameworks}},
guarding a raising compute_framework_definition() so it degrades to an EMPTY framework set, then
delegates matching to IdentifyFeatureGroupClass.evaluate(...). A FeatureGroup whose
compute_framework_rule() raises therefore maps to an empty set: it still matches name+domain+scope
(so it stays in candidates), but the seam never adds it to the winners because it has no supported
framework. resolve_feature never raises and the broken group fails closed (does not win).

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


def _make_broken_rule_fg() -> type[FeatureGroup]:
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class SbdgBrokenRuleFG(FeatureGroup):
        """Matches a unique name but its compute_framework_rule raises when consulted."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise RuntimeError("sbdg rule exploded")

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
    """resolve_feature never raises, and a broken-rule group fails closed instead of winning."""

    def test_resolve_feature_does_not_propagate_the_rule_exception(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            assert isinstance(result, ResolvedFeature)
            del result
        finally:
            del broken_fg
            gc.collect()

    def test_broken_rule_group_is_candidate_but_does_not_win(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            assert result.feature_name == SBDG_BROKEN_RULE_FEATURE
            # Still a criteria match (name+domain+scope), so it stays in candidates.
            assert broken_fg in result.candidates
            # But its rule raises -> empty framework set -> no supported framework -> fails closed.
            assert result.feature_group is None
            assert result.error is not None
            del result
        finally:
            del broken_fg
            gc.collect()
