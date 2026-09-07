"""Failing tests for the shared resolution helper pair in identify_feature_group (board issue os-016).

The "evaluate -> render -> raise" idiom is hand-copied in the engine, in the blessed os-014 test seam and
(as a projection) in resolve_feature. Two module-level helpers replace those copies:

  * ``evaluate_and_render(feature, accessible_plugins, links=None, data_access_collection=None)`` returns
    ``(EvaluationResult, str | None)``: one ``IdentifyFeatureGroupClass.evaluate`` pass plus
    ``render_resolution_failure`` over it, the message being None exactly when the feature resolved.
  * ``resolve_or_raise(feature, accessible_plugins, links=None, data_access_collection=None, partial_records=())``
    returns that same EvaluationResult on success and raises the typed ``FeatureResolutionError`` on failure,
    carrying the rendered message, the feature name, the evaluation and the snapshotted, capped partial records.

``ComputeFrameworkPinError`` is a misuse validated before matching, so it escapes both helpers unconverted, and
the error ``resolve_or_raise`` raises must be equivalent to the one the os-014 seam raises for the same feature.

The third converged call site, ``resolve_feature``, does not delegate to the helper: it projects the pair's
result into a ``ResolvedFeature`` behind a never-raise guard that now covers rendering as well as evaluation.
Both halves of that are pinned here, against the helper itself.

All fixture names carry an ``016`` suffix: test feature groups become global subclasses and the suite runs in
parallel, so a shared name would leak into another module's candidate universe.
"""

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api import plugin_docs
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, PreFilterPlugins
from mloda.core.prepare.identify_feature_group import (
    ComputeFrameworkPinError,
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    evaluate_and_render,
    resolve_or_raise,
)
from mloda.core.prepare.resolution_failure_renderer import render_resolution_failure, scope_callout
from mloda.core.prepare.resolution_types import (
    PARTIAL_RECORDS_CAP,
    EvaluationResult,
    ResolutionRecord,
)
from tests.helpers.plugin_stubs import make_fg
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


# Unique names so nothing collides with the parallel suite (board issue os-016).
RESOLVE_MATCH_FEATURE_016 = "resolve_or_raise_016_single_match"
RESOLVE_NO_MATCH_FEATURE_016 = "resolve_or_raise_016_no_match_at_all"
RESOLVE_MULTIPLE_FEATURE_016 = "resolve_or_raise_016_multiple_match"
RESOLVE_ABSTRACT_FEATURE_016 = "resolve_or_raise_016_abstract_match"


class ResolveOrRaiseFw016(ComputeFramework):
    """Dummy compute framework for the resolution-helper tests."""


class ResolveOrRaiseFwBeta016(ComputeFramework):
    """Second dummy compute framework, distinct from ResolveOrRaiseFw016."""


ResolveOrRaiseMatchFG016 = make_fg("ResolveOrRaiseMatchFG016", matches=RESOLVE_MATCH_FEATURE_016)

ResolveOrRaiseSiblingAFG016 = make_fg(
    "ResolveOrRaiseSiblingAFG016", matches=RESOLVE_MULTIPLE_FEATURE_016, domain="resolve_or_raise_016_a"
)

ResolveOrRaiseSiblingBFG016 = make_fg(
    "ResolveOrRaiseSiblingBFG016", matches=RESOLVE_MULTIPLE_FEATURE_016, domain="resolve_or_raise_016_b"
)

ResolveOrRaiseAbstractFG016 = make_fg(
    "ResolveOrRaiseAbstractFG016", matches=RESOLVE_ABSTRACT_FEATURE_016, abstract=True
)


def _match_plugins() -> FeatureGroupEnvironmentMapping:
    """One concrete candidate: the match name resolves, every other name fails with kind 'none'."""
    return {ResolveOrRaiseMatchFG016: {ResolveOrRaiseFw016}}


def _multiple_plugins() -> FeatureGroupEnvironmentMapping:
    """Two unrelated siblings matching the same name on distinct frameworks: kind 'multiple'."""
    return {
        ResolveOrRaiseSiblingAFG016: {ResolveOrRaiseFw016},
        ResolveOrRaiseSiblingBFG016: {ResolveOrRaiseFwBeta016},
    }


def _abstract_plugins() -> FeatureGroupEnvironmentMapping:
    """Only an uninstantiable abstract base matches: kind 'abstract_only'."""
    return {ResolveOrRaiseAbstractFG016: {ResolveOrRaiseFw016}}


def _plugins_for(feature_name: str) -> FeatureGroupEnvironmentMapping:
    """The accessible-plugin set that makes the given fixture name fail with its intended kind."""
    if feature_name == RESOLVE_MULTIPLE_FEATURE_016:
        return _multiple_plugins()
    if feature_name == RESOLVE_ABSTRACT_FEATURE_016:
        return _abstract_plugins()
    return _match_plugins()


def _frameworks_016() -> set[type[ComputeFramework]]:
    """A fresh compute-framework restriction, the one resolve_feature accepts as a keyword."""
    return {ResolveOrRaiseFw016}


def _collector_for(feature_name: str) -> PluginCollector:
    """A collector restricting the universe to exactly the fixture groups behind one failure kind."""
    if feature_name == RESOLVE_MULTIPLE_FEATURE_016:
        return PluginCollector.enabled_feature_groups({ResolveOrRaiseSiblingAFG016, ResolveOrRaiseSiblingBFG016})
    if feature_name == RESOLVE_ABSTRACT_FEATURE_016:
        return PluginCollector.enabled_feature_groups({ResolveOrRaiseAbstractFG016})
    return PluginCollector.enabled_feature_groups({ResolveOrRaiseMatchFG016})


def _built_plugins_for(feature_name: str) -> FeatureGroupEnvironmentMapping:
    """The environment resolve_feature builds for that fixture, via the same PreFilterPlugins path."""
    return PreFilterPlugins(_frameworks_016(), _collector_for(feature_name)).get_accessible_plugins()


def _resolve_feature_016(feature_name: str, scope: type[FeatureGroup] | None = None) -> ResolvedFeature:
    """Run resolve_feature over the fixture universe of one failure kind."""
    return resolve_feature(
        feature_name,
        feature_group=scope,
        plugin_collector=_collector_for(feature_name),
        compute_frameworks=_frameworks_016(),
    )


def _empty_result() -> EvaluationResult:
    """A minimal EvaluationResult for cheap ResolutionRecord construction."""
    return EvaluationResult(identified={})


def _record(name: str, result: EvaluationResult) -> ResolutionRecord:
    """A cheap ResolutionRecord with the given feature name."""
    return ResolutionRecord(feature_name=name, requested=True, result=result)


def _pinned_feature(name: str) -> Feature:
    """A feature pinned to two compute frameworks, the misuse evaluate() rejects before matching."""
    feature = Feature(name)
    feature.compute_frameworks = {ResolveOrRaiseFw016, ResolveOrRaiseFwBeta016}
    return feature


def _raised_with_records(records: list[ResolutionRecord]) -> FeatureResolutionError:
    """The error resolve_or_raise raises for the no-match fixture, carrying the given partial records."""
    with pytest.raises(FeatureResolutionError) as exc_info:
        resolve_or_raise(
            feature=Feature(RESOLVE_NO_MATCH_FEATURE_016),
            accessible_plugins=_match_plugins(),
            links=None,
            data_access_collection=None,
            partial_records=records,
        )
    return exc_info.value


FAILURE_CASES = [
    ("none", RESOLVE_NO_MATCH_FEATURE_016),
    ("multiple", RESOLVE_MULTIPLE_FEATURE_016),
    ("abstract_only", RESOLVE_ABSTRACT_FEATURE_016),
]


RENDER_EXPLOSION_016 = "resolve_or_raise_016 render step exploded"


def _exploding_evaluate_and_render(
    feature: Feature,
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: set[Link] | None = None,
    data_access_collection: DataAccessCollection | None = None,
) -> tuple[EvaluationResult, str | None]:
    """Stand-in for the helper pair that always raises, standing for a renderer that blows up."""
    raise RuntimeError(RENDER_EXPLOSION_016)


class TestEvaluateAndRenderSuccess:
    """On a resolvable feature the pair returns evaluate()'s result and no message."""

    def test_returns_the_evaluate_result_and_a_none_message(self) -> None:
        """The message is None exactly when the feature resolved, and the result equals evaluate()'s."""
        feature = Feature(RESOLVE_MATCH_FEATURE_016)

        result, message = evaluate_and_render(
            feature=feature,
            accessible_plugins=_match_plugins(),
            links=None,
            data_access_collection=None,
        )
        direct = IdentifyFeatureGroupClass.evaluate(feature, _match_plugins(), links=None)

        assert message is None
        assert isinstance(result, EvaluationResult)
        assert result == direct
        assert result.identified == {ResolveOrRaiseMatchFG016: {ResolveOrRaiseFw016}}

    def test_links_and_data_access_collection_are_optional(self) -> None:
        """Both trailing arguments default to None, so the two-argument call resolves the same way."""
        result, message = evaluate_and_render(
            feature=Feature(RESOLVE_MATCH_FEATURE_016),
            accessible_plugins=_match_plugins(),
        )

        assert message is None
        assert result.failure_kind is None


class TestEvaluateAndRenderFailure:
    """On a failing feature the pair returns the evaluation and the renderer's message verbatim."""

    @pytest.mark.parametrize(("failure_kind", "feature_name"), FAILURE_CASES)
    def test_message_is_exactly_the_renderer_projection(self, failure_kind: str, feature_name: str) -> None:
        """The returned message is byte-identical to render_resolution_failure over the same evaluation."""
        feature = Feature(feature_name)
        accessible_plugins = _plugins_for(feature_name)

        result, message = evaluate_and_render(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
        direct = IdentifyFeatureGroupClass.evaluate(feature, _plugins_for(feature_name), links=None)

        assert result.failure_kind == failure_kind
        assert message is not None
        assert message == render_resolution_failure(direct, feature)

    def test_failed_result_matches_evaluate_structurally(self) -> None:
        """The returned result carries the same identified / criteria / abstract / elimination facts."""
        feature = Feature(RESOLVE_NO_MATCH_FEATURE_016)

        result, _ = evaluate_and_render(
            feature=feature,
            accessible_plugins=_match_plugins(),
            links=None,
            data_access_collection=None,
        )
        direct = IdentifyFeatureGroupClass.evaluate(feature, _match_plugins(), links=None)

        assert result.identified == direct.identified
        assert result.criteria_matched == direct.criteria_matched
        assert result.abstract_matched == direct.abstract_matched
        assert result.eliminations == direct.eliminations


class TestResolveOrRaiseSuccess:
    """On a resolvable feature resolve_or_raise returns evaluate()'s result unchanged."""

    def test_returns_the_same_result_as_evaluate(self) -> None:
        """No raise, and the returned EvaluationResult equals the direct evaluate() call's."""
        feature = Feature(RESOLVE_MATCH_FEATURE_016)

        result = resolve_or_raise(
            feature=feature,
            accessible_plugins=_match_plugins(),
            links=None,
            data_access_collection=None,
        )
        direct = IdentifyFeatureGroupClass.evaluate(feature, _match_plugins(), links=None)

        assert isinstance(result, EvaluationResult)
        assert result == direct
        assert result.failure_kind is None

    def test_optional_arguments_default(self) -> None:
        """links, data_access_collection and partial_records all default, so two arguments suffice."""
        result = resolve_or_raise(
            feature=Feature(RESOLVE_MATCH_FEATURE_016),
            accessible_plugins=_match_plugins(),
        )

        assert result.identified == {ResolveOrRaiseMatchFG016: {ResolveOrRaiseFw016}}


class TestResolveOrRaiseFailure:
    """On a failing feature resolve_or_raise raises the typed error carrying message, name and evaluation."""

    def test_raises_the_typed_error_with_the_renderer_message(self) -> None:
        """str(error) is exactly render_resolution_failure over the same evaluation."""
        feature = Feature(RESOLVE_NO_MATCH_FEATURE_016)

        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=feature,
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )
        direct = IdentifyFeatureGroupClass.evaluate(feature, _match_plugins(), links=None)
        expected = render_resolution_failure(direct, feature)

        assert isinstance(exc_info.value, ValueError)
        assert expected is not None
        assert str(exc_info.value) == expected

    def test_error_carries_the_feature_name(self) -> None:
        """feature_name is the string form of the failing feature's name."""
        feature = Feature(RESOLVE_NO_MATCH_FEATURE_016)

        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=feature,
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )

        assert exc_info.value.feature_name == str(feature.name)

    def test_error_result_carries_the_same_evaluation(self) -> None:
        """The attached result holds the structured facts of the single pass it just ran."""
        feature = Feature(RESOLVE_NO_MATCH_FEATURE_016)

        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=feature,
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )
        direct = IdentifyFeatureGroupClass.evaluate(feature, _match_plugins(), links=None)
        result = exc_info.value.result

        assert result.failure_kind == "none"
        assert result.identified == direct.identified
        assert result.criteria_matched == direct.criteria_matched
        assert result.abstract_matched == direct.abstract_matched
        assert result.eliminations == direct.eliminations

    @pytest.mark.parametrize(("failure_kind", "feature_name"), FAILURE_CASES)
    def test_raises_for_every_failure_kind(self, failure_kind: str, feature_name: str) -> None:
        """Each of the three failure kinds raises with the renderer's own message over the same evaluation."""
        feature = Feature(feature_name)
        accessible_plugins = _plugins_for(feature_name)

        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )
        direct = IdentifyFeatureGroupClass.evaluate(feature, _plugins_for(feature_name), links=None)
        expected = render_resolution_failure(direct, feature)

        assert exc_info.value.result.failure_kind == failure_kind
        assert expected is not None
        assert str(exc_info.value) == expected


class TestResolveOrRaisePartialRecords:
    """partial_records default to empty, reach the error, are snapshotted, and honor the cap."""

    def test_partial_records_default_to_the_empty_tuple(self) -> None:
        """A call that passes no records raises an error carrying no records at all."""
        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=Feature(RESOLVE_NO_MATCH_FEATURE_016),
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )

        assert exc_info.value.partial_records == ()

    def test_passed_partial_records_reach_the_error(self) -> None:
        """The records the caller passes arrive on the raised error, in order, as a tuple."""
        result = _empty_result()
        records = [_record("passed_016_a", result), _record("passed_016_b", result)]

        error = _raised_with_records(records)

        assert isinstance(error.partial_records, tuple)
        assert [record.feature_name for record in error.partial_records] == ["passed_016_a", "passed_016_b"]

    def test_partial_records_are_snapshotted(self) -> None:
        """Mutating the caller's list or a record's payload afterwards does not change the error."""
        payload = EvaluationResult(identified={}, criteria_matched={ResolveOrRaiseMatchFG016})
        records = [_record("snapshot_016", payload)]

        error = _raised_with_records(records)
        records.append(_record("appended_016", _empty_result()))
        payload.criteria_matched.add(ResolveOrRaiseSiblingAFG016)

        assert len(error.partial_records) == 1
        assert error.partial_records[0].feature_name == "snapshot_016"
        assert error.partial_records[0].result.criteria_matched == {ResolveOrRaiseMatchFG016}

    def test_partial_records_are_truncated_to_the_last_cap_entries(self) -> None:
        """More records than PARTIAL_RECORDS_CAP keeps the tail and drops the head."""
        result = _empty_result()
        records = [_record(f"capped_016_{index}", result) for index in range(PARTIAL_RECORDS_CAP + 3)]

        error = _raised_with_records(records)

        assert len(error.partial_records) == PARTIAL_RECORDS_CAP
        assert error.partial_records[0].feature_name == "capped_016_3"
        assert error.partial_records[-1].feature_name == f"capped_016_{PARTIAL_RECORDS_CAP + 2}"


class TestComputeFrameworkPinErrorEscapesBothHelpers:
    """A >1 compute-framework pin is a misuse, never converted into a resolution failure."""

    def test_evaluate_and_render_propagates_the_pin_error(self) -> None:
        """evaluate_and_render lets ComputeFrameworkPinError out instead of rendering a message."""
        with pytest.raises(ComputeFrameworkPinError):
            evaluate_and_render(
                feature=_pinned_feature(RESOLVE_NO_MATCH_FEATURE_016),
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )

    def test_resolve_or_raise_propagates_the_pin_error(self) -> None:
        """resolve_or_raise lets the same pin error out, unconverted, with its own wording."""
        feature = _pinned_feature(RESOLVE_NO_MATCH_FEATURE_016)

        with pytest.raises(ComputeFrameworkPinError) as exc_info:
            resolve_or_raise(
                feature=feature,
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )

        assert ResolveOrRaiseFw016.get_class_name() in str(exc_info.value)
        assert ResolveOrRaiseFwBeta016.get_class_name() in str(exc_info.value)

    def test_pin_error_escapes_even_on_a_resolvable_name(self) -> None:
        """The pin check runs before matching, so a name that would resolve still raises the pin error."""
        with pytest.raises(ComputeFrameworkPinError):
            resolve_or_raise(
                feature=_pinned_feature(RESOLVE_MATCH_FEATURE_016),
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )


class TestSeamParity:
    """resolve_or_raise and the blessed os-014 seam raise equivalent errors for the same failing feature."""

    @pytest.mark.parametrize(("failure_kind", "feature_name"), FAILURE_CASES)
    def test_helper_and_seam_errors_are_equivalent(self, failure_kind: str, feature_name: str) -> None:
        """Same message, same feature_name and same failure_kind across all three failure kinds."""
        feature = Feature(feature_name)

        with pytest.raises(FeatureResolutionError) as helper_info:
            resolve_or_raise(
                feature=feature,
                accessible_plugins=_plugins_for(feature_name),
                links=None,
                data_access_collection=None,
            )
        with pytest.raises(FeatureResolutionError) as seam_info:
            evaluate_or_raise(
                feature=feature,
                accessible_plugins=_plugins_for(feature_name),
                links=None,
                data_access_collection=None,
            )

        assert str(helper_info.value) == str(seam_info.value)
        assert helper_info.value.feature_name == seam_info.value.feature_name
        assert helper_info.value.result.failure_kind == failure_kind
        assert seam_info.value.result.failure_kind == failure_kind


class TestResolveFeatureCallSiteParity:
    """resolve_feature projects exactly what resolve_or_raise raises for the same environment.

    The seam above delegates to resolve_or_raise, so it re-enters the same function; resolve_feature does
    not, which makes it the one call site whose agreement can actually break. It takes no accessible-plugins
    mapping, only a collector and a compute-framework restriction, so the environment is rebuilt through the
    PreFilterPlugins path it uses internally instead of the hand-built fixture mappings. All three failure
    kinds are expressible that way, which the environment test below pins.

    Complements test_single_pass_exactly_once.py::TestEngineAndResolveFeatureAgree, whose reference is the
    engine seam and whose subject is the per-attempt hook budget; here the reference is the helper itself.
    """

    def test_the_rebuilt_environment_is_the_fixture_universe(self) -> None:
        """The collector projection yields exactly the fixture groups, so both call sites see one universe."""
        assert _built_plugins_for(RESOLVE_NO_MATCH_FEATURE_016) == _match_plugins()
        assert _built_plugins_for(RESOLVE_ABSTRACT_FEATURE_016) == _abstract_plugins()
        assert set(_built_plugins_for(RESOLVE_MULTIPLE_FEATURE_016)) == {
            ResolveOrRaiseSiblingAFG016,
            ResolveOrRaiseSiblingBFG016,
        }

    @pytest.mark.parametrize(("failure_kind", "feature_name"), FAILURE_CASES)
    def test_resolve_feature_error_is_the_error_the_helper_raises(self, failure_kind: str, feature_name: str) -> None:
        """Same string on both call sites, for all three failure kinds."""
        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=Feature(feature_name),
                accessible_plugins=_built_plugins_for(feature_name),
                links=None,
                data_access_collection=None,
            )
        resolved = _resolve_feature_016(feature_name)

        assert exc_info.value.result.failure_kind == failure_kind
        assert resolved.feature_group is None
        assert resolved.error == str(exc_info.value)

    @pytest.mark.parametrize(("failure_kind", "feature_name"), FAILURE_CASES)
    def test_resolve_feature_candidates_are_that_error_criteria_matched(
        self, failure_kind: str, feature_name: str
    ) -> None:
        """The projected candidates are the failing evaluation's criteria-matched groups, sorted by name."""
        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=Feature(feature_name),
                accessible_plugins=_built_plugins_for(feature_name),
                links=None,
                data_access_collection=None,
            )
        resolved = _resolve_feature_016(feature_name)

        expected = sorted(candidate.get_class_name() for candidate in exc_info.value.result.criteria_matched)
        assert [candidate.get_class_name() for candidate in resolved.candidates] == expected

    def test_a_resolvable_feature_agrees_on_the_winner(self) -> None:
        """Success side of the same parity: the helper's identified group is the ResolvedFeature's winner."""
        result = resolve_or_raise(
            feature=Feature(RESOLVE_MATCH_FEATURE_016),
            accessible_plugins=_built_plugins_for(RESOLVE_MATCH_FEATURE_016),
            links=None,
            data_access_collection=None,
        )
        resolved = _resolve_feature_016(RESOLVE_MATCH_FEATURE_016)

        assert resolved.error is None
        assert resolved.feature_group is next(iter(result.identified))
        assert resolved.feature_group is ResolveOrRaiseMatchFG016


class TestResolveFeatureDegradesARaisingHelper:
    """resolve_feature guards evaluation AND rendering, so a raising helper becomes an error result.

    The one intentional behaviour delta of the convergence: the never-raise guard now wraps the whole
    evaluate-plus-render pair, so a renderer that blows up can no longer escape the debug API. The
    degraded result is fail-closed exactly like the pre-existing raising-evaluate path: no candidates, and
    the scope callout appended by resolve_feature because no rendered message carried one.
    """

    def test_a_raising_helper_becomes_the_error_instead_of_escaping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The call returns a ResolvedFeature carrying the raised message rather than propagating it."""
        monkeypatch.setattr(plugin_docs, "evaluate_and_render", _exploding_evaluate_and_render)

        resolved = _resolve_feature_016(RESOLVE_MATCH_FEATURE_016)

        assert isinstance(resolved, ResolvedFeature)
        assert resolved.feature_name == RESOLVE_MATCH_FEATURE_016
        assert resolved.feature_group is None
        assert resolved.error is not None
        assert RENDER_EXPLOSION_016 in resolved.error

    def test_the_degraded_path_reports_no_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail-closed: nothing is re-matched to fill in candidates behind the failure."""
        monkeypatch.setattr(plugin_docs, "evaluate_and_render", _exploding_evaluate_and_render)

        resolved = _resolve_feature_016(RESOLVE_MATCH_FEATURE_016)

        assert resolved.candidates == []

    def test_the_degraded_path_appends_the_scope_callout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A scoped request still names its scope once, as on the raising-evaluate path."""
        monkeypatch.setattr(plugin_docs, "evaluate_and_render", _exploding_evaluate_and_render)

        resolved = _resolve_feature_016(RESOLVE_MATCH_FEATURE_016, scope=ResolveOrRaiseMatchFG016)

        callout = scope_callout(ResolveOrRaiseMatchFG016)
        assert callout is not None
        assert resolved.error is not None
        assert RENDER_EXPLOSION_016 in resolved.error
        assert resolved.error.endswith(callout)
        assert resolved.error.count(callout) == 1

    def test_the_same_call_resolves_without_the_raising_helper(self) -> None:
        """Control: the degradation above is caused by the raise, not by the fixture environment."""
        resolved = _resolve_feature_016(RESOLVE_MATCH_FEATURE_016)

        assert resolved.error is None
        assert resolved.feature_group is ResolveOrRaiseMatchFG016
