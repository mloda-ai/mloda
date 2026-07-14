"""Target-contract tests for resolve_feature against a broken compute_framework_rule
(issue #722 Stage 3b).

The rewired resolve_feature shares the engine's standalone environment build
(build_resolution_environment). A FeatureGroup whose compute_framework_rule() raises
fails that build with an invalid-declaration error, exactly like the same declaration
kills every engine run today. resolve_feature stays never-raising: the environment
error is returned as ResolvedFeature.error with feature_group=None instead of the old
open-degraded resolution that named the broken class as the winner.

The broken class is scoped per test (del + gc.collect(), same pattern as
test_plugin_docs.py) so it does not leak into the session-wide subclass registry and
poison other standalone environment builds.
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
    """resolve_feature never raises; a broken declaration fails the environment build instead."""

    def test_resolve_feature_does_not_propagate_the_rule_exception(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            assert isinstance(result, ResolvedFeature)
            del result
        finally:
            del broken_fg
            gc.collect()

    def test_broken_rule_fails_the_environment_build_with_an_error(self) -> None:
        broken_fg = _make_broken_rule_fg()
        try:
            result = resolve_feature(SBDG_BROKEN_RULE_FEATURE)
            assert result.feature_name == SBDG_BROKEN_RULE_FEATURE
            # TARGET CONTRACT: the invalid declaration fails the standalone environment build,
            # so there is no winner and no fabricated candidate list; the error carries the
            # original declaration failure message.
            assert result.feature_group is None
            assert result.error is not None
            assert "sbdg rule exploded" in result.error
            assert result.candidates == []
            del result
        finally:
            del broken_fg
            gc.collect()
