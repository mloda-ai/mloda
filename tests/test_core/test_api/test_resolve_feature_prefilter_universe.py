"""Failing tests (TDD Red, issue #757) for resolve_feature routing its universe through PreFilterPlugins.

Today ``resolve_feature`` hand-builds its candidate universe from every installed FeatureGroup and
only applies the collector's applicability (enabled/disabled) filter. It never applies registry strict
mode, never intersects a caller-supplied compute-framework set, and never labels the environment it
resolved against. That diverges from what a real run's engine sees, which is built by
``PreFilterPlugins`` in ``mloda/core/prepare/accessible_plugins.py`` (see ``Engine.__init__``).

Target contract for the Green phase (these tests encode it and must FAIL today):

  1. DoD-1: a new keyword-only ``compute_frameworks: Optional[set[type[ComputeFramework]]]`` parameter.
     Restricting to a framework a matching group does not declare empties its available set, so the
     group fails to resolve, mirroring how ``PreFilterPlugins`` intersects the caller's framework set.
  2. DoD-2: passing a ``plugin_collector`` yields the same universe the engine builds via
     ``PreFilterPlugins`` - including registry strict mode, which resolve_feature ignores today.
  3. DoD-3: ``ResolvedFeature`` gains ``environment: str = "standalone-default"``, carried on every
     result (success and error paths).
  4. Never-raising is preserved under the new routing.

Right-reason failures today: DoD-1 tests raise ``TypeError`` (no ``compute_frameworks`` parameter);
DoD-2 asserts a strict-mode drop resolve_feature does not yet apply; DoD-3 and DoD-4 touch the
not-yet-existing ``environment`` field (``AttributeError``).
"""

import gc
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.user import PluginCollector, PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


PLAIN_FEATURE_757 = "PlainResolve757Feature"
PYTHON_DICT_ONLY_FEATURE = "PythonDictOnlyResolve757Feature"
STRICT_EXCLUDED_FEATURE = "StrictExcludedResolve757Feature"
BROKEN_RULE_FEATURE_757 = "broken_rule_resolve_757_feature"


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


class PlainResolve757FeatureGroup(FeatureGroup):
    """A plain, unambiguously resolvable fixture matching its own feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PLAIN_FEATURE_757

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class PythonDictOnlyResolve757FeatureGroup(FeatureGroup):
    """Declares ONLY PythonDictFramework, so a Pandas-only restriction empties its available set."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PYTHON_DICT_ONLY_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class StrictExcludedResolve757FeatureGroup(FeatureGroup):
    """Unregistered fixture: registry strict mode drops it from the engine's universe."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == STRICT_EXCLUDED_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _make_broken_rule_fg_757() -> type[FeatureGroup]:
    """Build a per-test broken-rule FeatureGroup, mirroring test_sbdg_resolve_feature_broken_rule.py.

    Scoped per test (del + gc.collect() by the caller) so its raising compute_framework_rule cannot
    leak into the session-wide subclass registry and break unrelated PreFilterPlugins builds.
    """
    gc.collect()

    class BrokenRuleResolve757FG(FeatureGroup):
        """Matches a unique name but its compute_framework_rule raises when consulted."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise RuntimeError("resolve757 rule exploded")

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return str(feature_name) == BROKEN_RULE_FEATURE_757

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return BrokenRuleResolve757FG


class TestResolveFeatureComputeFrameworkRestriction:
    """DoD-1: a keyword-only compute_frameworks set intersects each group's available frameworks (#757)."""

    def test_undeclared_framework_restriction_empties_available_set_and_fails(self) -> None:
        """Restricting to a framework the group does not declare fails resolution, like PreFilterPlugins.

        The fixture declares only PythonDictFramework; restricting to {PandasDataFrame} makes the
        intersection empty, so the group has no supported framework and cannot resolve.
        """
        collector = PluginCollector.enabled_feature_groups({PythonDictOnlyResolve757FeatureGroup})

        # Full default environment (restriction unset): the group resolves.
        default_result = resolve_feature(PYTHON_DICT_ONLY_FEATURE, plugin_collector=collector)
        assert default_result.feature_group is PythonDictOnlyResolve757FeatureGroup
        assert default_result.error is None

        # Restriction to an undeclared-but-available framework empties the group's set: fails closed.
        restricted = resolve_feature(
            PYTHON_DICT_ONLY_FEATURE,
            plugin_collector=collector,
            compute_frameworks={PandasDataFrame},
        )
        assert restricted.feature_group is None
        assert restricted.error is not None

    def test_compute_frameworks_none_matches_the_default_environment(self) -> None:
        """compute_frameworks=None must behave exactly like the unset, full-environment default."""
        collector = PluginCollector.enabled_feature_groups({PythonDictOnlyResolve757FeatureGroup})

        unset = resolve_feature(PYTHON_DICT_ONLY_FEATURE, plugin_collector=collector)
        explicit_none = resolve_feature(PYTHON_DICT_ONLY_FEATURE, plugin_collector=collector, compute_frameworks=None)

        assert explicit_none.feature_group is unset.feature_group
        assert explicit_none.feature_group is PythonDictOnlyResolve757FeatureGroup
        assert explicit_none.error == unset.error


class TestResolveFeatureCollectorUniverseParity:
    """DoD-2: resolve_feature's universe honors the same collector policy the engine's PreFilterPlugins does."""

    def test_strict_mode_drop_matches_engine_prefilter_universe(self) -> None:
        """A strict-mode collector drops the unregistered fixture from both the engine env and resolve_feature.

        The engine builds its universe via ``PreFilterPlugins(compute_frameworks, collector)``. With
        registry strict mode, an unregistered concrete FeatureGroup is dropped. resolve_feature must
        honor that same policy; today it ignores strict mode and still resolves the group.
        """
        collector = PluginCollector().set_strict_mode("strict")
        compute_frameworks_set = PreFilterPlugins.get_cfw_subclasses()

        # This is exactly what mloda/core/core/engine.py builds.
        engine_env = PreFilterPlugins(compute_frameworks_set, collector).get_accessible_plugins()
        assert StrictExcludedResolve757FeatureGroup not in engine_env

        # Without the strict collector the group resolves by its unique name.
        baseline = resolve_feature(STRICT_EXCLUDED_FEATURE)
        assert baseline.feature_group is StrictExcludedResolve757FeatureGroup
        assert baseline.error is None

        # With the strict collector, resolve_feature must apply the same drop the engine applies.
        scoped = resolve_feature(STRICT_EXCLUDED_FEATURE, plugin_collector=collector)
        assert scoped.feature_group is None
        assert StrictExcludedResolve757FeatureGroup not in scoped.candidates
        assert scoped.error is not None


class TestResolvedFeatureEnvironmentLabel:
    """DoD-3: ResolvedFeature gains environment: str = "standalone-default", carried on every result (#757)."""

    def test_dataclass_default_is_standalone_default(self) -> None:
        """The 4-positional-arg constructor still works and defaults environment to the literal label."""
        result = ResolvedFeature("n", None, [], None)
        assert result.environment == "standalone-default"

    def test_resolving_result_carries_environment(self) -> None:
        """A successful resolution reports the standalone-default environment."""
        collector = PluginCollector.enabled_feature_groups({PlainResolve757FeatureGroup})

        result = resolve_feature(PLAIN_FEATURE_757, plugin_collector=collector)

        assert result.feature_group is PlainResolve757FeatureGroup
        assert result.environment == "standalone-default"

    def test_no_match_error_result_carries_environment(self) -> None:
        """A no-match / error result also reports the standalone-default environment."""
        result = resolve_feature("CompletelyMissing757FeatureXYZ")

        assert result.feature_group is None
        assert result.error is not None
        assert result.environment == "standalone-default"


class TestResolveFeatureNeverRaisesUnderNewRouting:
    """DoD-4: the never-raising contract survives a broken-plugin / strict scenario (#757)."""

    def test_broken_rule_under_strict_collector_fails_closed_without_raising(self) -> None:
        """A broken-rule group under a strict collector fails closed and is labelled, never raising."""
        broken_fg = _make_broken_rule_fg_757()
        try:
            collector = PluginCollector().set_strict_mode("strict")

            result = resolve_feature(BROKEN_RULE_FEATURE_757, plugin_collector=collector)

            assert isinstance(result, ResolvedFeature)
            assert result.feature_group is None
            assert result.error is not None
            assert result.environment == "standalone-default"
        finally:
            del broken_fg
            gc.collect()
