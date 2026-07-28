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

All fixture names carry an ``016`` suffix: test feature groups become global subclasses and the suite runs in
parallel, so a shared name would leak into another module's candidate universe.
"""

from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    PARTIAL_RECORDS_CAP,
    ComputeFrameworkPinError,
    EvaluationResult,
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    ResolutionRecord,
    evaluate_and_render,
    render_resolution_failure,
    resolve_or_raise,
)
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


class ResolveOrRaiseMatchFG016(FeatureGroup):
    """Concrete feature group matching exactly one unique helper-test feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == RESOLVE_MATCH_FEATURE_016

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ResolveOrRaiseSiblingAFG016(FeatureGroup):
    """First of two unrelated siblings matching the same name (distinct domain 'resolve_or_raise_016_a')."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("resolve_or_raise_016_a")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == RESOLVE_MULTIPLE_FEATURE_016

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ResolveOrRaiseSiblingBFG016(FeatureGroup):
    """Second unrelated sibling matching the same name (distinct domain 'resolve_or_raise_016_b')."""

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain("resolve_or_raise_016_b")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == RESOLVE_MULTIPLE_FEATURE_016

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ResolveOrRaiseAbstractFG016(FeatureGroup):
    """Abstract base matching a unique name; uninstantiable via an unimplemented abstract hook.

    No concrete subclass is registered, so this base is the only name-match and can never win.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == RESOLVE_ABSTRACT_FEATURE_016

    @classmethod
    @abstractmethod
    def _resolve_or_raise_016_abstract_hook(cls, data: Any) -> Any:
        """Abstract hook that makes this base uninstantiable."""
        ...

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


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
        """Each of the three failure kinds raises, and the message matches the pair's rendered one."""
        feature = Feature(feature_name)
        accessible_plugins = _plugins_for(feature_name)

        with pytest.raises(FeatureResolutionError) as exc_info:
            resolve_or_raise(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )
        _, message = evaluate_and_render(
            feature=feature,
            accessible_plugins=_plugins_for(feature_name),
            links=None,
            data_access_collection=None,
        )

        assert exc_info.value.result.failure_kind == failure_kind
        assert str(exc_info.value) == message


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
        with pytest.raises(ComputeFrameworkPinError) as exc_info:
            evaluate_and_render(
                feature=_pinned_feature(RESOLVE_NO_MATCH_FEATURE_016),
                accessible_plugins=_match_plugins(),
                links=None,
                data_access_collection=None,
            )

        assert not isinstance(exc_info.value, FeatureResolutionError)

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

        assert not isinstance(exc_info.value, FeatureResolutionError)
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
