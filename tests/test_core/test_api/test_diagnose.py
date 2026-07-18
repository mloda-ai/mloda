"""Failing tests for the non-raising resolution preflight ``mlodaAPI.diagnose()`` (issue #812).

Contract under test:
  * ``mloda.core.prepare.identify_feature_group.ResolutionDiagnosis`` is a frozen dataclass beside
    ``ResolutionRecord`` and ``EvaluationResult``, with fields ``records``, ``complete``, ``feature_name``,
    ``failed_result``, ``message`` in that order. It is re-exported from ``mloda.user`` and ``mloda.steward``.
  * ``mlodaAPI.diagnose(...)`` runs the SAME eager planning as ``prepare(...)`` but NEVER raises for a
    resolution or configuration failure; it projects the outcome into a ``ResolutionDiagnosis``.
      - success: ``complete is True``, ``records == prepare(...).resolution_report()``, the three failure fields None.
      - resolution failure: ``complete is False``, ``records`` holds the records captured before the failing
        feature, and ``feature_name``/``failed_result``/``message`` describe it and equal the raising path's error.
      - configuration error: ``complete is False``, ``records == []``, ``message`` equals the raised ValueError text.

All fixture feature-group names carry an ``_812`` suffix and root feature names a ``diagnose_`` prefix: test
feature groups become global subclasses and a ``DataCreator`` claim is registry-wide, so a shared name would
leak into another module's candidate universe in the parallel suite.
"""

import dataclasses
from typing import Any, Optional

import pandas as pd
import pytest

# Aliased: a bare ``import mloda.user`` would bind the name ``mloda`` to the package and collide with the
# ``mloda`` mlodaAPI alias imported below.
import mloda.steward as mloda_steward
import mloda.user as mloda_user
from mloda.core.prepare.identify_feature_group import (
    EvaluationResult,
    FeatureResolutionError,
    ResolutionDiagnosis,
    ResolutionRecord,
)
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


# ---------------------------------------------------------------------------
# Test feature groups and fixtures
# ---------------------------------------------------------------------------

SOURCE_FEATURE_812 = "diagnose_sales_812"
UNKNOWN_FEATURE_812 = "diagnose_unknown_812"
BOGUS_ORDERING = "bogus"


class DiagnoseSource_812(FeatureGroup):
    """Pandas root source providing the feature the consumer chains off."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({SOURCE_FEATURE_812})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({SOURCE_FEATURE_812: [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class DiagnoseConsumer_812(FeatureGroup):
    """Consumes the source feature, so the source becomes a derived input during recursion."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_FEATURE_812)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["DiagnoseConsumer_812"] = data[SOURCE_FEATURE_812] * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class DiagnoseRaisingInputs_812(FeatureGroup):
    """Resolves, but raises a plain ValueError during the planning recursion (not a resolution failure)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        raise ValueError("diagnose planning boom 812")

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_DIAGNOSE_PLUGINS = PluginCollector.enabled_feature_groups({DiagnoseSource_812, DiagnoseConsumer_812})


def _success_features() -> list[Feature | str]:
    """A single requested consumer chaining off one derived source feature."""
    return ["DiagnoseConsumer_812"]


def _failure_features() -> list[Feature | str]:
    """A resolvable consumer FIRST, then an unknown name that no group can resolve."""
    return ["DiagnoseConsumer_812", UNKNOWN_FEATURE_812]


def _diagnose_success() -> ResolutionDiagnosis:
    return mloda.diagnose(_success_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_DIAGNOSE_PLUGINS)


def _prepare_success() -> mlodaAPI:
    return mloda.prepare(_success_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_DIAGNOSE_PLUGINS)


def _diagnose_failure() -> ResolutionDiagnosis:
    return mloda.diagnose(_failure_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_DIAGNOSE_PLUGINS)


def _raised_failure_error() -> FeatureResolutionError:
    """Catch the typed error the raising path produces for the same failing request."""
    with pytest.raises(FeatureResolutionError) as exc_info:
        mloda.prepare(_failure_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_DIAGNOSE_PLUGINS)
    return exc_info.value


def _diagnose_config_error() -> ResolutionDiagnosis:
    return mloda.diagnose(
        _success_features(),
        compute_frameworks={PandasDataFrame},
        plugin_collector=_DIAGNOSE_PLUGINS,
        column_ordering=BOGUS_ORDERING,
    )


def _raised_config_error() -> ValueError:
    """Catch the plain configuration ValueError the raising path produces for a bogus column_ordering."""
    with pytest.raises(ValueError) as exc_info:
        mloda.prepare(
            _success_features(),
            compute_frameworks={PandasDataFrame},
            plugin_collector=_DIAGNOSE_PLUGINS,
            column_ordering=BOGUS_ORDERING,
        )
    # The column_ordering guard runs first in __init__, before resolution, so this is not a resolution error.
    assert not isinstance(exc_info.value, FeatureResolutionError)
    return exc_info.value


# ---------------------------------------------------------------------------
# ResolutionDiagnosis dataclass
# ---------------------------------------------------------------------------


class TestResolutionDiagnosisDataclass:
    """ResolutionDiagnosis is a frozen dataclass with the documented field order and defaults."""

    def test_resolution_diagnosis_is_a_dataclass(self) -> None:
        assert dataclasses.is_dataclass(ResolutionDiagnosis)

    def test_resolution_diagnosis_is_frozen(self) -> None:
        diagnosis = ResolutionDiagnosis(records=[], complete=True)

        with pytest.raises(dataclasses.FrozenInstanceError):
            diagnosis.complete = False  # type: ignore[misc]

    def test_resolution_diagnosis_field_order(self) -> None:
        field_names = [field.name for field in dataclasses.fields(ResolutionDiagnosis)]

        assert field_names == ["records", "complete", "feature_name", "failed_result", "message"]

    def test_optional_fields_default_to_none(self) -> None:
        diagnosis = ResolutionDiagnosis(records=[], complete=True)

        assert diagnosis.feature_name is None
        assert diagnosis.failed_result is None
        assert diagnosis.message is None


# ---------------------------------------------------------------------------
# diagnose() success outcome
# ---------------------------------------------------------------------------


class TestDiagnoseSuccess:
    """A fully resolvable request yields a complete diagnosis whose records mirror prepare()."""

    def test_diagnose_is_available_on_both_aliases(self) -> None:
        # mloda is an alias of the mlodaAPI class, so diagnose resolves to the same method under both names.
        assert mloda is mlodaAPI
        assert callable(mlodaAPI.diagnose)

    def test_success_returns_a_resolution_diagnosis(self) -> None:
        assert isinstance(_diagnose_success(), ResolutionDiagnosis)

    def test_success_is_complete_with_no_failure_fields(self) -> None:
        diagnosis = _diagnose_success()

        assert diagnosis.complete is True
        assert diagnosis.feature_name is None
        assert diagnosis.failed_result is None
        assert diagnosis.message is None

    def test_success_records_are_non_empty_resolution_records(self) -> None:
        diagnosis = _diagnose_success()

        assert isinstance(diagnosis.records, list)
        assert diagnosis.records, "a resolvable request must report at least one resolution record"
        assert all(isinstance(record, ResolutionRecord) for record in diagnosis.records)

    def test_success_records_equal_prepare_resolution_report(self) -> None:
        """diagnose() runs the same eager planning as prepare(): the records match exactly."""
        diagnosis = _diagnose_success()
        report = _prepare_success().resolution_report()

        assert diagnosis.records == report


# ---------------------------------------------------------------------------
# diagnose() success records are detached
# ---------------------------------------------------------------------------


class TestDiagnoseSuccessRecordsAreDetached:
    """The returned records are a detached copy; mutating them does not corrupt a fresh diagnosis."""

    def test_clearing_records_does_not_corrupt_a_fresh_diagnosis(self) -> None:
        first = _diagnose_success()
        count = len(first.records)
        assert count

        first.records.clear()

        assert len(_diagnose_success().records) == count

    def test_mutating_a_records_nested_result_does_not_corrupt_a_fresh_diagnosis(self) -> None:
        """Clearing a returned record's mutable EvaluationResult must not corrupt a fresh diagnosis."""
        first = _diagnose_success()
        first.records[0].result.identified.clear()
        first.records[0].result.criteria_matched.clear()

        fresh = _diagnose_success()

        assert fresh.records[0].result.identified, "a later diagnosis must not reflect the caller's mutation"
        assert fresh.records[0].result.failure_kind is None

    def test_records_still_equal_a_fresh_prepare_report_after_mutation(self) -> None:
        first = _diagnose_success()
        first.records.clear()

        assert _diagnose_success().records == _prepare_success().resolution_report()


# ---------------------------------------------------------------------------
# diagnose() resolution-failure outcome
# ---------------------------------------------------------------------------


class TestDiagnoseResolutionFailure:
    """An unresolvable feature name projects the raising path's error instead of raising."""

    def test_resolution_failure_is_incomplete(self) -> None:
        assert _diagnose_failure().complete is False

    def test_records_hold_the_feature_resolved_before_the_failure(self) -> None:
        names = [record.feature_name for record in _diagnose_failure().records]

        assert "DiagnoseConsumer_812" in names

    def test_failing_feature_record_is_not_in_records(self) -> None:
        names = [record.feature_name for record in _diagnose_failure().records]

        assert UNKNOWN_FEATURE_812 not in names

    def test_feature_name_is_the_failing_feature(self) -> None:
        assert _diagnose_failure().feature_name == UNKNOWN_FEATURE_812

    def test_failed_result_is_an_evaluation_result_with_a_failure_kind(self) -> None:
        failed = _diagnose_failure().failed_result

        assert isinstance(failed, EvaluationResult)
        assert failed.failure_kind is not None

    def test_message_equals_the_raising_paths_error_text(self) -> None:
        diagnosis = _diagnose_failure()
        caught = _raised_failure_error()

        assert isinstance(diagnosis.message, str)
        assert diagnosis.message
        assert diagnosis.message == str(caught)

    def test_feature_name_and_failure_kind_match_the_raising_path(self) -> None:
        diagnosis = _diagnose_failure()
        caught = _raised_failure_error()

        assert diagnosis.feature_name == caught.feature_name
        assert diagnosis.failed_result is not None
        assert diagnosis.failed_result.failure_kind == caught.result.failure_kind


# ---------------------------------------------------------------------------
# diagnose() configuration-error outcome
# ---------------------------------------------------------------------------


class TestDiagnoseConfigurationError:
    """A non-resolution ValueError (bogus column_ordering) is projected, not raised."""

    def test_configuration_error_is_incomplete(self) -> None:
        assert _diagnose_config_error().complete is False

    def test_configuration_error_has_empty_records(self) -> None:
        assert _diagnose_config_error().records == []

    def test_configuration_error_has_no_failing_feature_fields(self) -> None:
        diagnosis = _diagnose_config_error()

        assert diagnosis.feature_name is None
        assert diagnosis.failed_result is None

    def test_message_equals_the_raised_value_error_text(self) -> None:
        diagnosis = _diagnose_config_error()
        caught = _raised_config_error()

        assert isinstance(diagnosis.message, str)
        assert diagnosis.message
        assert diagnosis.message == str(caught)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestResolutionDiagnosisIsPubliclyExported:
    """ResolutionDiagnosis is re-exported from mloda.user and mloda.steward, mirroring ResolutionRecord."""

    def test_resolution_diagnosis_exported_from_user_and_steward(self) -> None:
        assert "ResolutionDiagnosis" in mloda_user.__all__
        assert "ResolutionDiagnosis" in mloda_steward.__all__

    def test_user_and_steward_reexport_the_same_object(self) -> None:
        assert mloda_user.ResolutionDiagnosis is ResolutionDiagnosis
        assert mloda_steward.ResolutionDiagnosis is ResolutionDiagnosis


# ---------------------------------------------------------------------------
# diagnose() only swallows resolution and setup-configuration errors
# ---------------------------------------------------------------------------


class TestDiagnoseDoesNotSwallowPlanningErrors:
    """A non-resolution ValueError raised during the planning pass must propagate, not be projected."""

    def test_non_resolution_planning_error_propagates(self) -> None:
        plugins = PluginCollector.enabled_feature_groups({DiagnoseRaisingInputs_812})
        with pytest.raises(ValueError) as exc_info:
            mloda.diagnose(
                ["DiagnoseRaisingInputs_812"], compute_frameworks={PandasDataFrame}, plugin_collector=plugins
            )
        # It is the planning-time error, not a resolution failure projected into a diagnosis.
        assert not isinstance(exc_info.value, FeatureResolutionError)
        assert "diagnose planning boom 812" in str(exc_info.value)


# ---------------------------------------------------------------------------
# diagnose() empty request
# ---------------------------------------------------------------------------


class TestDiagnoseEmptyRequest:
    """An empty request resolves to a trivially complete diagnosis with no records."""

    def test_empty_request_is_trivially_complete(self) -> None:
        diagnosis = mloda.diagnose([], compute_frameworks={PandasDataFrame}, plugin_collector=_DIAGNOSE_PLUGINS)
        assert diagnosis.complete is True
        assert diagnosis.records == []
        assert diagnosis.feature_name is None
        assert diagnosis.failed_result is None
        assert diagnosis.message is None
