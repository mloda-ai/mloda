"""Failing tests for per-feature EvaluationResult capture and ``session.resolution_report()`` (issue #811).

Contract under test:
  * ``mloda.core.prepare.identify_feature_group.ResolutionRecord`` is a frozen dataclass beside
    ``EvaluationResult``, with fields ``feature_name: str``, ``requested: bool``, ``result: EvaluationResult``
    in that order. It is re-exported from both ``mloda.user`` and ``mloda.steward``.
  * ``IdentifyFeatureGroupClass(...).result`` exposes the winning ``EvaluationResult`` on a successful
    resolution (``failure_kind is None``, ``identified`` holding the winner).
  * ``Engine`` gains ``resolution_records: list[ResolutionRecord]``, populated in traversal order as each
    feature is identified: top-level requested features get ``requested=True``, features found via
    ``input_features`` recursion get ``requested=False``. Records are not deduplicated.
  * ``mlodaAPI.resolution_report() -> list[ResolutionRecord]`` returns those records (a fresh list each call)
    without re-matching, available after ``prepare()`` and unchanged by ``run()`` (mirrors ``resolved_plan()``).
  * On a mid-recursion ``FeatureResolutionError`` the records captured before the failing feature remain on
    ``engine.resolution_records``; the failing feature's own record is not appended.

All fixture feature-group names carry an ``_811`` suffix and all root feature names a ``res_report_`` prefix:
test feature groups become global subclasses and a ``DataCreator`` claim is registry-wide, so a shared name
would leak into another module's candidate universe in the parallel suite.
"""

import dataclasses
from typing import Any, Optional
from unittest.mock import patch

import pandas as pd
import pytest

# Aliased: a bare ``import mloda.user`` would bind the name ``mloda`` to the package and collide with the
# ``mloda`` mlodaAPI alias imported below.
import mloda.steward as mloda_steward
import mloda.user as mloda_user
from mloda.core.core.engine import Engine
from mloda.core.prepare.identify_feature_group import (
    EvaluationResult,
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    ResolutionRecord,
)
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import (
    DataAccessCollection,
    Feature,
    FeatureName,
    Features,
    Options,
    PluginCollector,
    mloda,
    mlodaAPI,
)
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


# ---------------------------------------------------------------------------
# Test feature groups
# ---------------------------------------------------------------------------


class ResReportSource_811(FeatureGroup):
    """Pandas root source providing the feature the consumer chains off."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"res_report_sales"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"res_report_sales": [10, 20, 30]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class ResReportConsumer_811(FeatureGroup):
    """Consumes the source feature, so the source becomes a derived input during recursion."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("res_report_sales")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["ResReportConsumer_811"] = data["res_report_sales"] * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


# Counts every match_feature_group_criteria call so a re-match after prepare would be visible.
RES_REPORT_MATCH_CALLS_811: dict[str, int] = {}


class ResReportCountingSource_811(FeatureGroup):
    """Root source whose match hook counts calls, proving resolution_report() does not re-match."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"res_report_counted_811"})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        RES_REPORT_MATCH_CALLS_811[cls.get_class_name()] = RES_REPORT_MATCH_CALLS_811.get(cls.get_class_name(), 0) + 1
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"res_report_counted_811": [1, 2, 3]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


# Feature names for the engine-level and identify-level tests.
GOOD_FEATURE_811 = "res_report_good_811"
BAD_FEATURE_811 = "res_report_bad_811"


class ResReportFw_811(ComputeFramework):
    """Dummy compute framework for the engine-level and identify-level tests."""


class ResReportGoodFG_811(FeatureGroup):
    """Resolves exactly GOOD_FEATURE_811; any other name finds no candidate and raises."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == GOOD_FEATURE_811

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


_RES_REPORT_PLUGINS = PluginCollector.enabled_feature_groups({ResReportSource_811, ResReportConsumer_811})
_COUNTING_PLUGINS = PluginCollector.enabled_feature_groups({ResReportCountingSource_811})


def _prepare_session() -> mlodaAPI:
    """A prepared session whose requested consumer chains off one derived source feature."""
    return mloda.prepare(
        ["ResReportConsumer_811"],
        compute_frameworks={PandasDataFrame},
        plugin_collector=_RES_REPORT_PLUGINS,
    )


# ---------------------------------------------------------------------------
# ResolutionRecord dataclass
# ---------------------------------------------------------------------------


class TestResolutionRecordDataclass:
    """ResolutionRecord is a frozen, value-comparable dataclass with the documented field order."""

    def test_resolution_record_is_a_dataclass(self) -> None:
        assert dataclasses.is_dataclass(ResolutionRecord)

    def test_resolution_record_is_frozen(self) -> None:
        record = ResolutionRecord(feature_name="x", requested=True, result=EvaluationResult(identified={}))

        with pytest.raises(dataclasses.FrozenInstanceError):
            record.feature_name = "y"  # type: ignore[misc]

    def test_resolution_record_field_order(self) -> None:
        field_names = [field.name for field in dataclasses.fields(ResolutionRecord)]

        assert field_names == ["feature_name", "requested", "result"]

    def test_resolution_records_compare_by_value(self) -> None:
        result = EvaluationResult(identified={})
        first = ResolutionRecord(feature_name="x", requested=True, result=result)
        second = ResolutionRecord(feature_name="x", requested=True, result=result)

        assert first == second

    def test_requested_flag_participates_in_equality(self) -> None:
        """A requested record must not compare equal to an otherwise identical derived one."""
        result = EvaluationResult(identified={})
        requested = ResolutionRecord(feature_name="x", requested=True, result=result)
        derived = ResolutionRecord(feature_name="x", requested=False, result=result)

        assert requested != derived


# ---------------------------------------------------------------------------
# resolution_report() over a successful session
# ---------------------------------------------------------------------------


class TestResolutionReportSuccessfulSession:
    """A prepared session reports how each feature resolved, in planning order."""

    def test_resolution_report_returns_non_empty_list_of_records(self) -> None:
        report = _prepare_session().resolution_report()

        assert isinstance(report, list)
        assert report, "a prepared session must report at least one resolution record"
        assert all(isinstance(record, ResolutionRecord) for record in report)

    def test_requested_feature_is_recorded_with_requested_true_and_no_failure(self) -> None:
        report = _prepare_session().resolution_report()

        consumer_records = [record for record in report if record.feature_name == "ResReportConsumer_811"]
        assert len(consumer_records) == 1
        assert consumer_records[0].requested is True
        assert consumer_records[0].result.failure_kind is None

    def test_derived_input_feature_is_recorded_with_requested_false(self) -> None:
        report = _prepare_session().resolution_report()

        source_records = [record for record in report if record.feature_name == "res_report_sales"]
        assert len(source_records) == 1
        assert source_records[0].requested is False
        assert source_records[0].result.failure_kind is None

    def test_requested_feature_is_recorded_before_its_derived_input(self) -> None:
        report = _prepare_session().resolution_report()

        names = [record.feature_name for record in report]
        assert names.index("ResReportConsumer_811") < names.index("res_report_sales")

    def test_each_record_result_is_an_evaluation_result(self) -> None:
        report = _prepare_session().resolution_report()

        assert all(isinstance(record.result, EvaluationResult) for record in report)

    def test_record_result_identifies_the_resolving_feature_group(self) -> None:
        """Each record's ``result.identified`` names the group that actually resolved the feature."""
        report = _prepare_session().resolution_report()

        consumer_record = next(record for record in report if record.feature_name == "ResReportConsumer_811")
        source_record = next(record for record in report if record.feature_name == "res_report_sales")

        assert ResReportConsumer_811 in consumer_record.result.identified
        assert ResReportSource_811 in source_record.result.identified


# ---------------------------------------------------------------------------
# fresh list per call and run() invariance
# ---------------------------------------------------------------------------


class TestResolutionReportFreshListAndRunInvariance:
    """resolution_report() hands back a fresh list and is unchanged by run() (mirrors resolved_plan())."""

    def test_resolution_report_returns_a_distinct_list_object_each_call(self) -> None:
        session = _prepare_session()

        assert session.resolution_report() is not session.resolution_report()

    def test_mutating_the_returned_list_does_not_affect_a_later_call(self) -> None:
        session = _prepare_session()

        first = session.resolution_report()
        count = len(first)
        first.clear()

        assert len(session.resolution_report()) == count

    def test_resolution_report_is_unchanged_by_run(self) -> None:
        session = _prepare_session()

        before_run = session.resolution_report()
        results = session.run()
        after_run = session.resolution_report()

        assert after_run == before_run

        # Sanity check that the session really did execute.
        assert len(results) == 1
        assert "ResReportConsumer_811" in results[0].columns

    def test_mutating_a_returned_records_nested_result_does_not_affect_a_later_call(self) -> None:
        """Clearing a returned record's mutable EvaluationResult must not corrupt a fresh report."""
        session = _prepare_session()

        report = session.resolution_report()
        target_name = report[0].feature_name
        report[0].result.identified.clear()
        report[0].result.criteria_matched.clear()

        fresh_record = next(r for r in session.resolution_report() if r.feature_name == target_name)
        assert fresh_record.result.identified, "a later report must not reflect the caller's mutation"
        assert fresh_record.result.failure_kind is None


# ---------------------------------------------------------------------------
# resolution_report() does not re-match
# ---------------------------------------------------------------------------


class TestResolutionReportDoesNotRematch:
    """resolution_report() reads captured records; it must not run matching again."""

    def test_resolution_report_does_not_increase_the_match_call_count(self) -> None:
        RES_REPORT_MATCH_CALLS_811.clear()

        session = mloda.prepare(
            ["res_report_counted_811"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=_COUNTING_PLUGINS,
        )
        after_prepare = dict(RES_REPORT_MATCH_CALLS_811)
        assert after_prepare, "the counting matcher must have run at least once during prepare()"

        session.resolution_report()
        session.resolution_report()

        assert RES_REPORT_MATCH_CALLS_811 == after_prepare


# ---------------------------------------------------------------------------
# IdentifyFeatureGroupClass exposes the winning result on success
# ---------------------------------------------------------------------------


class TestIdentifyFeatureGroupResultOnSuccess:
    """After a successful single-candidate resolution the identifier keeps its winning EvaluationResult."""

    def test_result_is_an_evaluation_result_with_no_failure(self) -> None:
        identifier = IdentifyFeatureGroupClass(
            feature=Feature(GOOD_FEATURE_811),
            accessible_plugins={ResReportGoodFG_811: {ResReportFw_811}},
            links=None,
            data_access_collection=None,
        )

        assert isinstance(identifier.result, EvaluationResult)
        assert identifier.result.failure_kind is None
        assert ResReportGoodFG_811 in identifier.result.identified


# ---------------------------------------------------------------------------
# Failure reachability at the engine level
# ---------------------------------------------------------------------------


class TestFailureReachabilityAtEngineLevel:
    """Records captured before a mid-recursion FeatureResolutionError survive on the engine."""

    def test_records_before_the_failing_feature_remain_on_the_engine(self) -> None:
        with (
            patch(
                "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
            ) as mocked_accessible_plugins,
            patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
        ):
            mocked_accessible_plugins.return_value = {ResReportGoodFG_811: {ResReportFw_811}}

            features = Features([GOOD_FEATURE_811, BAD_FEATURE_811])
            engine = Engine(features, {ResReportFw_811}, None)

            # The bare call must keep working: setup_features_recursion's new requested parameter defaults True.
            with pytest.raises(FeatureResolutionError):
                engine.setup_features_recursion(features)

            recorded_names = [record.feature_name for record in engine.resolution_records]
            assert GOOD_FEATURE_811 in recorded_names
            assert BAD_FEATURE_811 not in recorded_names

            good_records = [record for record in engine.resolution_records if record.feature_name == GOOD_FEATURE_811]
            assert len(good_records) == 1
            assert good_records[0].requested is True


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestResolutionRecordIsPubliclyExported:
    """ResolutionRecord is re-exported from mloda.user and mloda.steward, mirroring PlanStep."""

    def test_resolution_record_exported_from_user_and_steward(self) -> None:
        assert "ResolutionRecord" in mloda_user.__all__
        assert "ResolutionRecord" in mloda_steward.__all__

    def test_user_and_steward_reexport_the_same_object(self) -> None:
        assert mloda_user.ResolutionRecord is ResolutionRecord
        assert mloda_steward.ResolutionRecord is ResolutionRecord
