"""End-to-end integration test for ``mlodaAPI.diagnose()`` (issue #812).

Exercises the non-raising preflight through the public API with real plugins and the real
PandasDataFrame compute framework: shipped ``PandasAggregatedFeatureGroup`` aggregations chain off raw
columns produced by a ``DataCreator`` source. A fully resolvable request yields a complete diagnosis;
adding an unknown feature name projects a resolution-failure diagnosis instead of raising, and that
projection equals the raising path's ``FeatureResolutionError``.

Everything carries an ``_812`` suffix: the source becomes a global FeatureGroup subclass and its
DataCreator claim is registry-wide, so shared names would leak into another module's candidate universe
in the parallel suite.
"""

from typing import Any

import pytest

from mloda.core.prepare.identify_feature_group import (
    EvaluationResult,
    FeatureResolutionError,
    ResolutionDiagnosis,
    ResolutionRecord,
)
from mloda.provider import DefaultOptionKeys
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator

SALES_COL = "diag_e2e_sales_812"
REVENUE_COL = "diag_e2e_revenue_812"
SUM_FEATURE = "sum_diag_e2e_sales_812"
AVG_FEATURE = "avg_diag_e2e_revenue_812"
UNKNOWN_FEATURE = "diag_e2e_unknown_812"


class DiagE2ESource_812(ATestDataCreator):
    """DataCreator source whose raw columns the shipped aggregations chain off."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        """Return the uniquely named raw columns as a dictionary."""
        return {
            SALES_COL: [100, 200, 300, 400, 500],
            REVENUE_COL: [1000, 2000, 3000, 4000, 5000],
        }


_PLUGINS = PluginCollector.enabled_feature_groups({DiagE2ESource_812, PandasAggregatedFeatureGroup})


def _sum_feature() -> Feature:
    return Feature(
        SUM_FEATURE,
        Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "sum", DefaultOptionKeys.in_features: SALES_COL}),
    )


def _avg_feature() -> Feature:
    return Feature(
        AVG_FEATURE,
        Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "avg", DefaultOptionKeys.in_features: REVENUE_COL}),
    )


def _success_features() -> list[Feature | str]:
    return [_sum_feature(), _avg_feature()]


def _failure_features() -> list[Feature | str]:
    """The two resolvable aggregations first, then an unknown name that no group can resolve."""
    return [_sum_feature(), _avg_feature(), UNKNOWN_FEATURE]


class TestDiagnoseEndToEnd_812:
    """diagnose() projects the real planning pass over a shipped aggregation plan."""

    def test_full_request_yields_a_complete_diagnosis(self) -> None:
        """A fully resolvable request diagnoses complete and mirrors prepare().resolution_report()."""
        diagnosis = mloda.diagnose(_success_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_PLUGINS)

        assert isinstance(diagnosis, ResolutionDiagnosis)
        assert diagnosis.complete is True
        assert diagnosis.feature_name is None
        assert diagnosis.failed_result is None
        assert diagnosis.message is None

        assert isinstance(diagnosis.records, list)
        assert diagnosis.records, "a resolvable request must report at least one resolution record"
        assert all(isinstance(record, ResolutionRecord) for record in diagnosis.records)
        assert all(isinstance(record.result, EvaluationResult) for record in diagnosis.records)

        # Each requested aggregation resolves to the shipped PandasAggregatedFeatureGroup.
        for requested_name in (SUM_FEATURE, AVG_FEATURE):
            matches = [record for record in diagnosis.records if record.feature_name == requested_name]
            assert len(matches) == 1, f"expected exactly one record for {requested_name}"
            assert matches[0].requested is True
            assert matches[0].result.failure_kind is None
            assert PandasAggregatedFeatureGroup in matches[0].result.identified

        # Each derived source column resolves (requested=False) to our DataCreator source.
        for source_name in (SALES_COL, REVENUE_COL):
            matches = [record for record in diagnosis.records if record.feature_name == source_name]
            assert len(matches) == 1, f"expected exactly one record for {source_name}"
            assert matches[0].requested is False
            assert matches[0].result.failure_kind is None
            assert DiagE2ESource_812 in matches[0].result.identified

        # Same eager planning as prepare(): the records equal what prepare(...).resolution_report() reports.
        report = mloda.prepare(
            _success_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_PLUGINS
        ).resolution_report()
        assert diagnosis.records == report

    def test_unknown_feature_yields_a_resolution_failure_diagnosis(self) -> None:
        """Adding an unknown name projects the raising path's FeatureResolutionError, records held so far."""
        diagnosis = mloda.diagnose(_failure_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_PLUGINS)

        assert diagnosis.complete is False
        assert diagnosis.feature_name == UNKNOWN_FEATURE

        names = [record.feature_name for record in diagnosis.records]
        assert SUM_FEATURE in names, "the aggregation resolved before the failure must be recorded"
        assert UNKNOWN_FEATURE not in names

        assert isinstance(diagnosis.failed_result, EvaluationResult)
        assert diagnosis.failed_result.failure_kind is not None

        assert isinstance(diagnosis.message, str)
        assert diagnosis.message

        # The projection equals the raising path's typed error for the same failing request.
        with pytest.raises(FeatureResolutionError) as exc_info:
            mloda.prepare(_failure_features(), compute_frameworks={PandasDataFrame}, plugin_collector=_PLUGINS)
        caught = exc_info.value

        assert diagnosis.message == str(caught)
        assert diagnosis.feature_name == caught.feature_name
        assert diagnosis.failed_result.failure_kind == caught.result.failure_kind
