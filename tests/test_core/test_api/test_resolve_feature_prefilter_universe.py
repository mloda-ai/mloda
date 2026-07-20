"""resolve_feature routes its candidate universe through PreFilterPlugins (#757).

``PreFilterPlugins`` (``mloda/core/prepare/accessible_plugins.py``) is the same builder ``Engine.__init__``
uses, so the debug API resolves against the universe a real run sees. What that routing pins:

  1. A keyword-only ``compute_frameworks: Optional[set[type[ComputeFramework]]]`` parameter. Restricting to
     a framework a matching group does not declare empties its available set, so the group fails to resolve,
     mirroring how ``PreFilterPlugins`` intersects the caller's framework set.
  2. A ``plugin_collector`` yields the same universe the engine builds, registry strict mode included.
  3. Scoping the universe (strict mode, and equally a collector's enabled/disabled filter) drops a broken
     plugin before its declaration is consulted, so it cannot abort the build for everyone (#790).
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
from mloda.core.prepare.accessible_plugins import PreFilterPlugins
from mloda.user import PluginCollector, PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


PYTHON_DICT_ONLY_FEATURE = "PythonDictOnlyResolve757Feature"
STRICT_EXCLUDED_FEATURE = "StrictExcludedResolve757Feature"
BROKEN_RULE_FEATURE_757 = "broken_rule_resolve_757_feature"


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


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


class TestScopingABrokenPluginOutOfTheUniverse:
    """DoD-4: scoping the universe keeps a broken plugin from ever being consulted (#757, #790)."""

    def test_strict_collector_drops_broken_plugin_before_its_declaration_is_consulted(self) -> None:
        """A broken plugin scoped out of the universe yields an ordinary no-match, not a build failure.

        Environment building is fail-closed (#790): one plugin that raises while declaring its frameworks
        aborts the build for every feature in the process. Scoping the universe is the escape hatch from
        that blast radius. Strict mode drops the unregistered class in _set_feature_groups, BEFORE the
        declaration loop consults it (a PluginCollector's enabled/disabled filter does the same), so the
        build succeeds and the feature simply does not resolve.

        The absent attribution prefix is what makes this test non-vacuous: it is the only thing that tells
        "scoped out, never consulted" apart from "consulted and aborted the build".
        """
        broken_fg = _make_broken_rule_fg_757()
        try:
            collector = PluginCollector().set_strict_mode("strict")

            result = resolve_feature(BROKEN_RULE_FEATURE_757, plugin_collector=collector)
            winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
            error = result.error
            del result
        finally:
            del broken_fg
            gc.collect()

        assert winner_name is None
        assert error is not None
        assert f"No feature groups found for feature name: '{BROKEN_RULE_FEATURE_757}'" in error
        assert "Failed to build the plugin environment" not in error
