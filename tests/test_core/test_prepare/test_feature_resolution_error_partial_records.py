"""Failing tests for the diagnose refactor: FeatureResolutionError carries the partial resolution records.

Contract under test (PR #836):
  * ``PARTIAL_RECORDS_CAP`` is a module-level int constant (1000) in ``identify_feature_group``.
  * ``FeatureResolutionError.__init__`` accepts ``partial_records: Sequence[ResolutionRecord] = ()`` and stores
    it as a tuple truncated to the LAST ``PARTIAL_RECORDS_CAP`` entries; ``__reduce__`` round-trips it via pickle.
  * The raising ``mloda.prepare`` path attaches the records resolved before the failure to the error, and
    ``mloda.diagnose`` sources its failure projection records from that payload instead of a retained engine.
  * ``SetupConfigurationError`` (a ValueError, not a FeatureResolutionError) replaces the plain ValueError for
    setup-phase configuration errors, and the ``_defer_planning`` / ``auto_plan`` / ``Engine.plan`` hack is gone.

All fixture feature-group names carry an ``_836pr`` suffix and root feature names a ``partial_records_`` prefix:
test feature groups become global subclasses and a ``DataCreator`` claim is registry-wide, so a shared name would
leak into another module's candidate universe in the parallel suite.
"""

import inspect
import pickle  # nosec B403
from typing import Any, Optional

import pandas as pd
import pytest

from mloda.core.api.request import SetupConfigurationError
from mloda.core.core.engine import Engine
from mloda.core.prepare.identify_feature_group import (
    EvaluationResult,
    FeatureResolutionError,
    ResolutionRecord,
)
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


# ---------------------------------------------------------------------------
# Test feature groups and fixtures
# ---------------------------------------------------------------------------

SOURCE_FEATURE_836PR = "partial_records_sales_836pr"
UNKNOWN_FEATURE_836PR = "partial_records_unknown_836pr"
CONSUMER_FEATURE_836PR = "PartialRecordsConsumer_836pr"


class PartialRecordsSource_836pr(FeatureGroup):
    """Pandas root source providing the feature the consumer chains off."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({SOURCE_FEATURE_836PR})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({SOURCE_FEATURE_836PR: [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class PartialRecordsConsumer_836pr(FeatureGroup):
    """Consumes the source feature, so the source resolves as a derived input before the failing request."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature(SOURCE_FEATURE_836PR)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data[CONSUMER_FEATURE_836PR] = data[SOURCE_FEATURE_836PR] * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_PLUGINS_836PR = PluginCollector.enabled_feature_groups({PartialRecordsSource_836pr, PartialRecordsConsumer_836pr})


def _empty_result() -> EvaluationResult:
    """A minimal EvaluationResult for cheap ResolutionRecord construction."""
    return EvaluationResult(identified={})


def _record(name: str, result: EvaluationResult) -> ResolutionRecord:
    """A cheap ResolutionRecord with the given feature name."""
    return ResolutionRecord(feature_name=name, requested=True, result=result)


def _failing_request_features() -> list[Feature | str]:
    """A resolvable consumer FIRST, then an unknown name that no group can resolve."""
    return [CONSUMER_FEATURE_836PR, UNKNOWN_FEATURE_836PR]


def _raised_failure_error() -> FeatureResolutionError:
    """Catch the typed error the raising prepare path produces for the failing request."""
    with pytest.raises(FeatureResolutionError) as exc_info:
        mloda.prepare(
            _failing_request_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_PLUGINS_836PR
        )
    return exc_info.value


# ---------------------------------------------------------------------------
# PARTIAL_RECORDS_CAP constant
# ---------------------------------------------------------------------------


class TestPartialRecordsCapConstant:
    """PARTIAL_RECORDS_CAP is a module-level int constant with value 1000."""

    def test_partial_records_cap_is_1000(self) -> None:
        from mloda.core.prepare.identify_feature_group import PARTIAL_RECORDS_CAP

        assert isinstance(PARTIAL_RECORDS_CAP, int)
        assert PARTIAL_RECORDS_CAP == 1000


# ---------------------------------------------------------------------------
# FeatureResolutionError.partial_records constructor behavior
# ---------------------------------------------------------------------------


class TestFeatureResolutionErrorPartialRecords:
    """The error stores partial_records as a tuple, defaulting to empty and capped to the tail."""

    def test_partial_records_defaults_to_the_empty_tuple(self) -> None:
        error = FeatureResolutionError("boom 836pr", UNKNOWN_FEATURE_836PR, _empty_result())

        assert error.partial_records == ()

    def test_partial_records_is_stored_as_a_tuple(self) -> None:
        result = _empty_result()
        records = [_record("kept_836pr_a", result), _record("kept_836pr_b", result)]

        error = FeatureResolutionError("boom 836pr", UNKNOWN_FEATURE_836PR, result, partial_records=records)

        assert isinstance(error.partial_records, tuple)
        assert error.partial_records == tuple(records)

    def test_partial_records_is_truncated_to_the_last_cap_entries(self) -> None:
        from mloda.core.prepare.identify_feature_group import PARTIAL_RECORDS_CAP

        result = _empty_result()
        records = [_record(f"capped_836pr_{index}", result) for index in range(PARTIAL_RECORDS_CAP + 5)]

        error = FeatureResolutionError("boom 836pr", UNKNOWN_FEATURE_836PR, result, partial_records=records)

        assert len(error.partial_records) == PARTIAL_RECORDS_CAP
        # Tail kept, head dropped: the first 5 records are gone and the last one survives.
        assert error.partial_records[0].feature_name == "capped_836pr_5"
        assert error.partial_records[-1].feature_name == f"capped_836pr_{PARTIAL_RECORDS_CAP + 4}"


# ---------------------------------------------------------------------------
# FeatureResolutionError pickle round-trip
# ---------------------------------------------------------------------------


class TestFeatureResolutionErrorPickleRoundTrip:
    """__reduce__ carries partial_records through pickle alongside message, feature_name and result."""

    def test_pickle_round_trips_partial_records(self) -> None:
        result = _empty_result()
        records = [_record("pickled_836pr_a", result), _record("pickled_836pr_b", result)]
        error = FeatureResolutionError("pickle boom 836pr", UNKNOWN_FEATURE_836PR, result, partial_records=records)

        restored = pickle.loads(pickle.dumps(error))  # nosec B301

        assert str(restored) == "pickle boom 836pr"
        assert restored.feature_name == UNKNOWN_FEATURE_836PR
        assert len(restored.partial_records) == 2


# ---------------------------------------------------------------------------
# Raising path carries the payload
# ---------------------------------------------------------------------------


class TestRaisingPathCarriesPartialRecords:
    """mloda.prepare attaches the records resolved before the failure to the raised error."""

    def test_error_partial_records_is_a_tuple(self) -> None:
        error = _raised_failure_error()

        assert isinstance(error.partial_records, tuple)

    def test_partial_records_hold_the_features_resolved_before_the_failure(self) -> None:
        error = _raised_failure_error()

        names = [record.feature_name for record in error.partial_records]

        assert CONSUMER_FEATURE_836PR in names
        assert SOURCE_FEATURE_836PR in names

    def test_partial_records_hold_no_record_for_the_failing_feature(self) -> None:
        error = _raised_failure_error()

        names = [record.feature_name for record in error.partial_records]

        assert UNKNOWN_FEATURE_836PR not in names


# ---------------------------------------------------------------------------
# SetupConfigurationError
# ---------------------------------------------------------------------------


class TestSetupConfigurationError:
    """Setup-phase configuration errors raise a dedicated ValueError subclass."""

    def test_is_a_value_error_but_not_a_resolution_error(self) -> None:
        from mloda.core.api.request import SetupConfigurationError

        assert issubclass(SetupConfigurationError, ValueError)
        assert not issubclass(SetupConfigurationError, FeatureResolutionError)

    def test_bogus_column_ordering_raises_setup_configuration_error(self) -> None:
        from mloda.core.api.request import SetupConfigurationError

        with pytest.raises(SetupConfigurationError):
            mloda.prepare(
                [CONSUMER_FEATURE_836PR],
                compute_frameworks={PandasDataFrame},
                plugin_collector=_PLUGINS_836PR,
                column_ordering="bogus",
            )


# ---------------------------------------------------------------------------
# Deferred-planning hack removal
# ---------------------------------------------------------------------------


class TestDeferredPlanningHackIsRemoved:
    """The _defer_planning / auto_plan / Engine.plan plumbing is gone from the public constructors."""

    def test_mloda_api_init_has_no_defer_planning_parameter(self) -> None:
        assert "_defer_planning" not in inspect.signature(mlodaAPI.__init__).parameters

    def test_engine_init_has_no_auto_plan_parameter(self) -> None:
        assert "auto_plan" not in inspect.signature(Engine.__init__).parameters

    def test_engine_has_no_plan_method(self) -> None:
        assert not hasattr(Engine, "plan")


# ---------------------------------------------------------------------------
# diagnose() sources its failure records from the error payload
# ---------------------------------------------------------------------------


class TestDiagnoseRecordsComeFromTheErrorPayload:
    """The failure projection's records equal the raising path's partial_records for the same request."""

    def test_diagnose_records_equal_the_errors_partial_records(self) -> None:
        diagnosis = mloda.diagnose(
            _failing_request_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_PLUGINS_836PR
        )
        error = _raised_failure_error()

        assert diagnosis.records == list(error.partial_records)


# ---------------------------------------------------------------------------
# Pre-planning setup errors surface as SetupConfigurationError
# ---------------------------------------------------------------------------

DUPLICATE_FEATURE_836PR = "diag_dup_836pr"
UNKNOWN_FRAMEWORK_836PR = "NotAFramework_836pr"


class TestPrePlanningSetupErrorsAreConfigurationErrors:
    """Every ValueError raised during setup BEFORE engine planning surfaces as SetupConfigurationError."""

    def test_duplicate_string_feature_raises_setup_configuration_error(self) -> None:
        with pytest.raises(SetupConfigurationError) as exc_info:
            mloda.prepare([DUPLICATE_FEATURE_836PR, DUPLICATE_FEATURE_836PR], compute_frameworks={PandasDataFrame})

        assert str(exc_info.value) == f"You are adding same feature as string twice: {DUPLICATE_FEATURE_836PR}"

    def test_unknown_framework_name_raises_setup_configuration_error(self) -> None:
        with pytest.raises(SetupConfigurationError):
            mloda.prepare([DUPLICATE_FEATURE_836PR], compute_frameworks=[UNKNOWN_FRAMEWORK_836PR])

    def test_diagnose_projects_the_unknown_framework_error_as_message_only(self) -> None:
        diagnosis = mloda.diagnose([DUPLICATE_FEATURE_836PR], compute_frameworks=[UNKNOWN_FRAMEWORK_836PR])

        with pytest.raises(ValueError) as exc_info:
            mloda.prepare([DUPLICATE_FEATURE_836PR], compute_frameworks=[UNKNOWN_FRAMEWORK_836PR])

        assert diagnosis.complete is False
        assert diagnosis.records == []
        assert diagnosis.feature_name is None
        assert diagnosis.failed_result is None
        assert diagnosis.message == str(exc_info.value)

    def test_pickle_preserves_class_bearing_identified_mappings(self) -> None:
        """Pins current behavior: a partial record whose EvaluationResult maps a FeatureGroup class survives pickle."""
        result = EvaluationResult(identified={PartialRecordsSource_836pr: {PandasDataFrame}})
        record = ResolutionRecord(feature_name="pinned_836pr", requested=True, result=result)
        error = FeatureResolutionError("pin boom 836pr", UNKNOWN_FEATURE_836PR, result, partial_records=[record])

        restored = pickle.loads(pickle.dumps(error))  # nosec B301

        assert PartialRecordsSource_836pr in restored.partial_records[0].result.identified
