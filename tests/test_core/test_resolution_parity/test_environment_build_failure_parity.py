"""Paired engine/debug tests for environment-build failure semantics (#790).

A FeatureGroup whose compute_framework_rule() raises is an ENVIRONMENT-BUILD failure: it breaks
PreFilterPlugins while it maps feature groups to their frameworks, before any matching happens. That build is
fail-closed for both callers: the engine lets the provider's exception propagate raw, and resolve_feature
keeps its never-raising contract by projecting the same failure into ResolvedFeature.error. Since the build
aborts before matching, a broken group is not a listed candidate.

The broken class is built per test by a factory and dropped in a finally (del + gc.collect(), same pattern as
test_sbdg_resolve_feature_broken_rule.py). This matters more than usual here: under fail-closed semantics a
leaked broken class aborts the environment build of every unrelated resolve_feature/engine test in the same
xdist worker. So nothing that could pin it may outlive the scoped block: no exception traceback, and no
ResolvedFeature holding it in candidates. The assertions therefore run on plain strings read out of the
result before the scope closes.
"""

import gc
import sys
from typing import Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import PreFilterPlugins


ENV_BUILD_FAILURE_FEATURE = "probe790_env_build_failure"
ENV_BUILD_FAILURE_MESSAGE = "probe790 compute_framework_rule exploded"


def _make_broken_rule_fg() -> type[FeatureGroup]:
    """Build a fresh probe whose compute_framework_rule raises while the environment is built."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class EnvBuildFailureProbe790(FeatureGroup):
        """Matches a unique feature name, but its compute_framework_rule raises when consulted."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise RuntimeError(ENV_BUILD_FAILURE_MESSAGE)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return str(feature_name) == ENV_BUILD_FAILURE_FEATURE

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return EnvBuildFailureProbe790


def _capture_engine_build_failure() -> Optional[str]:
    """Run the engine's exact environment build and return the raised message, or None if it did not raise.

    Engine.__init__ builds PreFilterPlugins(compute_frameworks, plugin_collector); with the process-wide
    framework set and no collector this is that call.

    An unbound 'except' plus a transient sys.exc_info() read: a bound 'except ... as exc' keeps a traceback
    that pins the broken class through the PreFilterPlugins frame. Only the message string escapes here, so
    no traceback outlives the block.
    """
    try:
        PreFilterPlugins(PreFilterPlugins.get_cfw_subclasses(), None)
    except RuntimeError:
        return str(sys.exc_info()[1])
    return None


def test_engine_environment_build_fails_fast_on_broken_rule() -> None:
    """The engine's environment build propagates the provider's failure raw. Passes today: this side stays."""
    broken_fg = _make_broken_rule_fg()
    try:
        message = _capture_engine_build_failure()
    finally:
        del broken_fg
        gc.collect()

    assert message is not None, "PreFilterPlugins must not swallow a raising compute_framework_rule"
    assert ENV_BUILD_FAILURE_MESSAGE in message


def test_resolve_feature_projects_the_environment_build_failure() -> None:
    """resolve_feature does not raise and reports the real provider failure instead of a generic miss."""
    broken_fg = _make_broken_rule_fg()
    try:
        result = resolve_feature(ENV_BUILD_FAILURE_FEATURE)
        winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
        error = result.error
        candidate_names = [candidate.get_class_name() for candidate in result.candidates]
        environment = result.environment
        del result
    finally:
        del broken_fg
        gc.collect()

    # Never-raising contract: the failure is projected into the result, not thrown.
    assert winner_name is None
    assert error is not None
    assert "Failed to build the plugin environment" in error
    assert "RuntimeError" in error
    assert ENV_BUILD_FAILURE_MESSAGE in error
    # The build aborts before matching, so a broken group is not a listed candidate.
    assert candidate_names == []
    assert environment == "standalone-default"


def test_engine_and_resolve_feature_report_the_same_provider_failure() -> None:
    """Parity: both paths surface the SAME provider failure, not two independently drifting strings."""
    broken_fg = _make_broken_rule_fg()
    try:
        engine_message = _capture_engine_build_failure()
        result = resolve_feature(ENV_BUILD_FAILURE_FEATURE)
        resolve_error = result.error
        del result
    finally:
        del broken_fg
        gc.collect()

    assert engine_message is not None
    assert resolve_error is not None
    assert engine_message in resolve_error
