"""Paired engine/debug tests for environment-build failure semantics (#790, unified in #850).

A FeatureGroup whose compute_framework_rule() raises is an ENVIRONMENT-BUILD failure: it breaks
PreFilterPlugins while it maps feature groups to their frameworks, before any matching happens. That build is
fail-closed for every caller and now attributes the culprit uniformly (#850): the engine RAISES the typed
FrameworkDeclarationError instead of letting the provider's raw exception propagate, and resolve_feature keeps
its never-raising contract by projecting the SAME attributed text into ResolvedFeature.error. For an unscoped
feature the two strings are identical (resolve_feature only appends a scope suffix, empty here).
mlodaAPI.diagnose projects the same failure into a ResolutionDiagnosis instead of raising. Since the build
aborts before matching, a broken group is not a listed candidate.

Attribution is split by who is at fault rather than by which exception type a plugin happens to raise: a
PLUGIN's failure is attributed ("Failed to build the plugin environment: <type>: <msg> (raised by
<module:qualname> while declaring its compute frameworks)"), while mloda's own environment preconditions are
already complete user-facing sentences and stay bare. A PluginCollector that filters out every FeatureGroup is
one such precondition, and its message names the collector rather than the loader (#850).

The broken class is built per test by a factory and dropped in a finally (del + gc.collect(), same pattern as
test_sbdg_resolve_feature_broken_rule.py). This matters more than usual here: under fail-closed semantics a
leaked broken class aborts the environment build of every unrelated resolve_feature/engine test in the same
xdist worker. So nothing that could pin it may outlive the scoped block: no exception traceback (the engine and
prepare failures are caught unbound and read via sys.exc_info(), never pytest.raises/as exc), and no
ResolvedFeature holding it in candidates. The assertions therefore run on plain strings read out of the result
before the scope closes. The collector-filtered probe below is exempt: it declares its frameworks cleanly and
is merely filtered, so plain pytest.raises is safe for it.
"""

import gc
import sys
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.request import mlodaAPI
from mloda.core.core.engine import Engine
from mloda.core.prepare.accessible_plugins import (
    EnvironmentPreconditionError,
    FrameworkDeclarationError,
    PreFilterPlugins,
    RedefinitionConflictError,
)
from mloda.core.prepare.identify_feature_group import ResolutionDiagnosis
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


ENV_BUILD_FAILURE_FEATURE = "probe790_env_build_failure"
ENV_BUILD_FAILURE_MESSAGE = "probe790 compute_framework_rule exploded"
ENV_BUILD_VALUE_ERROR_FEATURE = "probe790_env_build_value_error"
ENV_BUILD_VALUE_ERROR_MESSAGE = "probe790 compute_framework_rule rejected its own declaration"
ENV_BUILD_MLODA_PRECONDITION_ERROR_FEATURE = "probe790_env_build_mloda_precondition_error"
ENV_BUILD_MLODA_PRECONDITION_ERROR_MESSAGE = "probe790 compute_framework_rule raised EnvironmentPreconditionError"
ENV_BUILD_MLODA_REDEFINITION_ERROR_FEATURE = "probe790_env_build_mloda_redefinition_error"
ENV_BUILD_MLODA_REDEFINITION_ERROR_MESSAGE = "probe790 compute_framework_rule raised RedefinitionConflictError"


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


def _make_value_error_rule_fg() -> type[FeatureGroup]:
    """Build a fresh probe whose compute_framework_rule raises ValueError while the environment is built.

    ValueError is not an exotic shape for this failure: FeatureGroup.compute_framework_definition raises it
    itself when compute_framework_rule() returns a non-set, the classic plugin-author mistake. It is also
    the shape mloda uses for its OWN environment preconditions, which is what makes the two easy to confuse.
    """
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class EnvBuildValueErrorProbe790(FeatureGroup):
        """Matches a unique feature name, but its compute_framework_rule rejects itself when consulted."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise ValueError(ENV_BUILD_VALUE_ERROR_MESSAGE)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return str(feature_name) == ENV_BUILD_VALUE_ERROR_FEATURE

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return EnvBuildValueErrorProbe790


def _make_mloda_typed_error_rule_fg(
    exc_type: type[ValueError], expected_feature_name: str, message: str
) -> type[FeatureGroup]:
    """Build a fresh probe whose compute_framework_rule raises a mloda-imported error type.

    EnvironmentPreconditionError and RedefinitionConflictError are public and importable, and subclass
    ValueError, so a plugin author can raise either from their own rule. The failure must still be
    attributed to the PLUGIN, not misread as mloda's OWN precondition just because the plugin borrowed a
    type mloda also uses for its policy.
    """
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class EnvBuildMlodaTypedErrorProbe790(FeatureGroup):
        """Matches a unique feature name, but its compute_framework_rule raises a mloda error type."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            raise exc_type(message)

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            return str(feature_name) == expected_feature_name

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
            return None

    return EnvBuildMlodaTypedErrorProbe790


def _capture_engine_build_failure() -> Optional[str]:
    """Drive a real Engine and return the message its environment build raised, or None if it did not raise.

    The Engine itself is under test here, not a stand-in: asserting in prose that PreFilterPlugins is what
    engine.py builds would keep this file green if the engine ever wrapped that build in a try/except, which
    is the exact drift these paired tests exist to catch. Engine.__init__ reaches its environment build
    before any planning work, so this raises in well under a millisecond and constructs nothing else.

    The engine attributes the provider failure as FrameworkDeclarationError (#850), so that is the type
    caught here. An unbound 'except' plus a transient sys.exc_info() read: a bound 'except ... as exc' (or
    pytest.raises) keeps a traceback that pins the broken class through the Engine frame. Only the message
    string escapes here, so no traceback outlives the block.
    """
    try:
        Engine(Features([Feature(ENV_BUILD_FAILURE_FEATURE)]), PreFilterPlugins.get_cfw_subclasses(), None)
    except FrameworkDeclarationError:
        return str(sys.exc_info()[1])
    return None


def test_engine_environment_build_fails_fast_on_broken_rule() -> None:
    """The engine aborts the run and ATTRIBUTES the culprit as FrameworkDeclarationError, not a raw exception (#850)."""
    broken_fg = _make_broken_rule_fg()
    identity = f"{broken_fg.__module__}:{broken_fg.__qualname__}"
    try:
        message = _capture_engine_build_failure()
    finally:
        del broken_fg
        gc.collect()

    assert message is not None, "the engine must not swallow a raising compute_framework_rule"
    assert "Failed to build the plugin environment" in message
    assert "RuntimeError" in message
    assert ENV_BUILD_FAILURE_MESSAGE in message
    assert identity in message


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


def test_resolve_feature_names_the_broken_plugin_for_an_unrelated_feature() -> None:
    """An unrelated feature still surfaces the broken plugin, named as module:qualname (#794)."""
    broken_fg = _make_broken_rule_fg()
    identity = f"{broken_fg.__module__}:{broken_fg.__qualname__}"
    try:
        result = resolve_feature("unrelated_probe794_feature")
        winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
        error = result.error
        candidate_names = [candidate.get_class_name() for candidate in result.candidates]
        del result
    finally:
        del broken_fg
        gc.collect()

    # The broken plugin declares EVERY group's frameworks, so it aborts the build of an unrelated feature too.
    assert winner_name is None
    assert error is not None
    assert "Failed to build the plugin environment" in error
    assert "RuntimeError" in error
    assert ENV_BUILD_FAILURE_MESSAGE in error
    # #794: the fail-closed error must name the culprit class, not just the raised exception.
    assert identity in error
    assert candidate_names == []


def test_engine_and_resolve_feature_report_the_same_provider_failure() -> None:
    """Parity: for an unscoped feature both paths surface the SAME attributed string, not two drifting ones (#850)."""
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
    # resolve_feature appends only a scope suffix, empty for this unscoped feature, so the strings are equal.
    assert engine_message == resolve_error


def test_diagnose_projects_the_environment_build_failure() -> None:
    """mlodaAPI.diagnose projects the attributed build failure into a ResolutionDiagnosis instead of raising (#850).

    The diagnosis carries the same text the raising prepare() path throws, with no records and no failed
    feature: the build aborts before any feature is resolved. prepare()'s failure is caught unbound and read
    via sys.exc_info() so its traceback never pins the broken class, matching the engine helper's discipline.
    """
    broken_fg = _make_broken_rule_fg()
    identity = f"{broken_fg.__module__}:{broken_fg.__qualname__}"
    try:
        diagnosis = mlodaAPI.diagnose([ENV_BUILD_FAILURE_FEATURE], compute_frameworks={PandasDataFrame})
        complete = diagnosis.complete
        message = diagnosis.message
        records = diagnosis.records
        feature_name = diagnosis.feature_name
        failed_result = diagnosis.failed_result
        del diagnosis
        prepare_error: Optional[str] = None
        try:
            mlodaAPI.prepare([ENV_BUILD_FAILURE_FEATURE], compute_frameworks={PandasDataFrame})
        except FrameworkDeclarationError:
            prepare_error = str(sys.exc_info()[1])
    finally:
        del broken_fg
        gc.collect()

    assert complete is False
    assert records == []
    assert feature_name is None
    assert failed_result is None
    assert message is not None
    assert "Failed to build the plugin environment" in message
    assert identity in message
    # The projected message is exactly what the raising path throws for the same request.
    assert prepare_error is not None
    assert message == prepare_error


def test_resolve_feature_attributes_a_plugin_raised_value_error() -> None:
    """A plugin's ValueError is attributed to the build like any other exception type it may raise.

    The exception type a plugin happens to raise must not decide whether the caller learns that a plugin
    broke. Unattributed, this failure is indistinguishable from an ordinary no-match: the reader sees a
    bare sentence and hunts for a feature-name typo that does not exist.
    """
    broken_fg = _make_value_error_rule_fg()
    try:
        result = resolve_feature(ENV_BUILD_VALUE_ERROR_FEATURE)
        winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
        error = result.error
        candidate_names = [candidate.get_class_name() for candidate in result.candidates]
        del result
    finally:
        del broken_fg
        gc.collect()

    assert winner_name is None
    assert error is not None
    assert "Failed to build the plugin environment" in error
    assert "ValueError" in error
    assert ENV_BUILD_VALUE_ERROR_MESSAGE in error
    assert candidate_names == []


@pytest.mark.parametrize(
    ("exc_type", "feature_name", "message"),
    [
        (
            EnvironmentPreconditionError,
            ENV_BUILD_MLODA_PRECONDITION_ERROR_FEATURE,
            ENV_BUILD_MLODA_PRECONDITION_ERROR_MESSAGE,
        ),
        (
            RedefinitionConflictError,
            ENV_BUILD_MLODA_REDEFINITION_ERROR_FEATURE,
            ENV_BUILD_MLODA_REDEFINITION_ERROR_MESSAGE,
        ),
    ],
)
def test_resolve_feature_attributes_a_plugin_raised_mloda_typed_error(
    exc_type: type[ValueError], feature_name: str, message: str
) -> None:
    """A plugin raising a mloda error type it imported is still attributed to the PLUGIN (#795).

    EnvironmentPreconditionError and RedefinitionConflictError are public and subclass ValueError, so a
    plugin can raise either from its own compute_framework_rule. Attribution must follow who is at fault, not
    which type was raised: this plugin's failure must be prefixed and name the culprit, never projected bare
    as if it were mloda's OWN precondition. Bare projection here would carry only the plugin's raw sentence,
    which does not contain "raised by <identity>", so asserting the identity proves the split held. Both
    types are exercised independently so a future type-specific carve-out cannot regress one while the other
    stays green.
    """
    broken_fg = _make_mloda_typed_error_rule_fg(exc_type, feature_name, message)
    identity = f"{broken_fg.__module__}:{broken_fg.__qualname__}"
    try:
        result = resolve_feature(feature_name)
        winner_name = result.feature_group.get_class_name() if result.feature_group is not None else None
        error = result.error
        candidate_names = [candidate.get_class_name() for candidate in result.candidates]
        del result
    finally:
        del broken_fg
        gc.collect()

    assert winner_name is None
    assert error is not None
    assert "Failed to build the plugin environment" in error
    assert identity in error
    assert exc_type.__name__ in error
    assert message in error
    assert candidate_names == []


def test_own_environment_precondition_is_projected_bare() -> None:
    """mloda's OWN environment preconditions stay unprefixed: no plugin is to blame for them.

    "Strict mode filtered out all FeatureGroups: ..." is already a complete, user-facing sentence naming
    its own fix. Prefixing it with "Failed to build the plugin environment" would blame a plugin for
    mloda's own policy. This is the counterweight to the attribution above: the two must stay distinct
    types, not distinct exception classes-of-the-day, so this test guards the split from either side.

    The precondition fires while the universe is assembled, before any declaration is consulted, so the
    broken probe alive here is never touched. It is present only to guarantee the precondition's premise:
    at least one concrete FeatureGroup existed before strict mode filtered them all out.
    """
    broken_fg = _make_broken_rule_fg()
    try:
        # An injected empty registry, not PluginRegistry.default().clear(): no global state is mutated,
        # so this stays independent of whatever else ran in this worker.
        collector = PluginCollector().set_strict_mode("strict").set_registry(PluginRegistry())
        result = resolve_feature(ENV_BUILD_FAILURE_FEATURE, plugin_collector=collector)
        error = result.error
        candidate_names = [candidate.get_class_name() for candidate in result.candidates]
        del result
    finally:
        del broken_fg
        gc.collect()

    assert error is not None
    assert "Strict mode filtered out all FeatureGroups" in error
    assert "Failed to build the plugin environment" not in error
    assert ENV_BUILD_FAILURE_MESSAGE not in error
    assert candidate_names == []


COLLECTOR_FILTERED_FEATURE_850 = "probe850_collector_filtered_feature"


class CollectorFilteredProbe850(FeatureGroup):
    """Declares its frameworks cleanly; exists only to be enabled AND disabled by one collector (#850).

    A module-scope, harmless global subclass: its rule never raises and it matches nothing, so it does not
    pollute any other test's candidate universe. It is not subject to the leaked-broken-class hazard, so the
    tests below may use plain pytest.raises.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _collector_that_filters_everything() -> PluginCollector:
    """A collector that both enables and disables the probe, emptying the (non-empty) loaded universe.

    applicable_feature_group_class checks disabled first, so the probe is dropped; and because the enabled set
    is non-empty and holds only that probe, every OTHER FeatureGroup is dropped too. The universe was
    non-empty, so this is a collector configuration mistake, not a nothing-loaded state.
    """
    collector = PluginCollector()
    collector.add_enabled_feature_group_classes({CollectorFilteredProbe850})
    collector.add_disabled_feature_group_classes({CollectorFilteredProbe850})
    return collector


def test_collector_that_filters_everything_names_the_collector_not_the_loader() -> None:
    """The env build blames the PluginCollector, with a message distinct from the nothing-loaded hint (#850)."""
    collector = _collector_that_filters_everything()

    with pytest.raises(EnvironmentPreconditionError) as exc_info:
        PreFilterPlugins(PreFilterPlugins.get_cfw_subclasses(), collector)

    message = str(exc_info.value)
    assert "PluginCollector" in message
    # A distinctive fragment: it cannot match an unrelated collector error, unlike the bare substring.
    assert "filtered out every FeatureGroup" in message
    # The loaded universe was non-empty, so the nothing-loaded hint would misdirect the fix.
    assert "Did you call PluginLoader.all()?" not in message


def test_diagnose_projects_the_collector_precondition_failure() -> None:
    """mlodaAPI.diagnose projects the collector's env-build precondition failure instead of raising (#850)."""
    collector = _collector_that_filters_everything()

    diagnosis = mlodaAPI.diagnose(
        [COLLECTOR_FILTERED_FEATURE_850], compute_frameworks={PandasDataFrame}, plugin_collector=collector
    )

    assert isinstance(diagnosis, ResolutionDiagnosis)
    assert diagnosis.complete is False
    assert diagnosis.records == []
    assert diagnosis.feature_name is None
    assert diagnosis.failed_result is None

    with pytest.raises(EnvironmentPreconditionError) as exc_info:
        mlodaAPI.prepare(
            [COLLECTOR_FILTERED_FEATURE_850], compute_frameworks={PandasDataFrame}, plugin_collector=collector
        )

    message = diagnosis.message
    assert message is not None
    assert message == str(exc_info.value)
    assert "PluginCollector" in message
    # A distinctive fragment: it cannot match an unrelated collector error, unlike the bare substring.
    assert "filtered out every FeatureGroup" in message
