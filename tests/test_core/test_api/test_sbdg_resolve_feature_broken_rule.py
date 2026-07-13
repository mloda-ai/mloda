"""Failing test pinning resolve_feature's never-raises contract against a broken
compute_framework_definition (sbdg defect 1).

The capability split in mloda/core/api/plugin_docs.py (around lines 411-425) is
safe_field-guarded, but the open-degrade fallback then calls
candidate.compute_framework_definition() UNGUARDED. A FeatureGroup whose
compute_framework_rule() raises therefore fails the split (degrading it to None),
and the fallback re-raises the very exception the guard just swallowed.

Expected failure today: resolve_feature propagates RuntimeError("sbdg rule exploded")
instead of returning a ResolvedFeature.

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
    """resolve_feature never raises, even when the open-degrade fallback path is hit."""

    def test_resolve_feature_does_not_propagate_the_rule_exception(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            assert isinstance(result, ResolvedFeature)
            del result
        finally:
            del broken_fg
            gc.collect()

    def test_degenerate_resolution_still_names_the_matching_group(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            assert result.feature_name == SBDG_BROKEN_RULE_FEATURE
            assert broken_fg in result.candidates
            # Degenerate path: capability info may be empty, but the resolution itself survives.
            assert result.feature_group is broken_fg
            del result
        finally:
            del broken_fg
            gc.collect()
