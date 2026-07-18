"""End-to-end integration test for ``session.resolution_report()`` (issue #811).

This exercises the shipped feature through the public API with real plugins and the real
PandasDataFrame compute framework: shipped ``PandasAggregatedFeatureGroup`` aggregations chain
off raw columns produced by a ``DataCreator`` source, so the report captures both the requested
aggregations (``requested=True``) and their derived source columns (``requested=False``).

Everything carries an ``_811`` suffix: the source becomes a global FeatureGroup subclass and its
DataCreator claim is registry-wide, so shared names would leak into another module's candidate
universe in the parallel suite.
"""

from typing import Any

from mloda.core.prepare.identify_feature_group import EvaluationResult
from mloda.provider import DefaultOptionKeys
from mloda.user import Feature, Options, PluginCollector, ResolutionRecord, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator

SALES_COL = "res_report_e2e_sales_811"
REVENUE_COL = "res_report_e2e_revenue_811"
SUM_FEATURE = "sum_res_report_e2e_sales_811"
AVG_FEATURE = "avg_res_report_e2e_revenue_811"


class ResReportE2ESource_811(ATestDataCreator):
    """DataCreator source whose raw columns the shipped aggregations chain off."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        """Return the uniquely named raw columns as a dictionary."""
        return {
            SALES_COL: [100, 200, 300, 400, 500],
            REVENUE_COL: [1000, 2000, 3000, 4000, 5000],
        }


def _prepared_session() -> mlodaAPI:
    """Prepare a real plan for two aggregations chaining off the DataCreator source."""
    f_sum = Feature(
        SUM_FEATURE,
        Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "sum", DefaultOptionKeys.in_features: SALES_COL}),
    )
    f_avg = Feature(
        AVG_FEATURE,
        Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "avg", DefaultOptionKeys.in_features: REVENUE_COL}),
    )
    plugins = PluginCollector.enabled_feature_groups({ResReportE2ESource_811, PandasAggregatedFeatureGroup})
    return mloda.prepare([f_sum, f_avg], compute_frameworks={PandasDataFrame}, plugin_collector=plugins)


class TestResolutionReportEndToEnd_811:
    """resolution_report() reflects the real planning pass of an executed aggregation plan."""

    def test_resolution_report_end_to_end(self) -> None:
        """Prepare, run real data, then assert the report over the executed plan."""
        session = _prepared_session()

        report_before = session.resolution_report()
        plan_before = session.resolved_plan()

        results = session.run()

        # Real end-to-end execution: the aggregations actually computed over the source columns.
        agg_df = next((df for df in results if SUM_FEATURE in df.columns and AVG_FEATURE in df.columns), None)
        assert agg_df is not None, "DataFrame with both aggregated features not found"
        assert agg_df[SUM_FEATURE].iloc[0] == 1500  # sum of [100, 200, 300, 400, 500]
        assert agg_df[AVG_FEATURE].iloc[0] == 3000  # avg of [1000, 2000, 3000, 4000, 5000]

        report_after = session.resolution_report()

        # A non-empty list of ResolutionRecord, mirroring resolved_plan()'s record shape.
        assert isinstance(report_after, list)
        assert report_after, "a prepared and executed session must report at least one resolution record"
        assert all(isinstance(record, ResolutionRecord) for record in report_after)
        assert all(isinstance(record.result, EvaluationResult) for record in report_after)

        # Each requested aggregation resolves to the shipped PandasAggregatedFeatureGroup.
        for requested_name in (SUM_FEATURE, AVG_FEATURE):
            matches = [record for record in report_after if record.feature_name == requested_name]
            assert len(matches) == 1, f"expected exactly one record for {requested_name}"
            assert matches[0].requested is True
            assert matches[0].result.failure_kind is None
            assert PandasAggregatedFeatureGroup in matches[0].result.identified

        # Each derived source column resolves (requested=False) to our DataCreator source.
        for source_name in (SALES_COL, REVENUE_COL):
            matches = [record for record in report_after if record.feature_name == source_name]
            assert len(matches) == 1, f"expected exactly one record for {source_name}"
            assert matches[0].requested is False
            assert matches[0].result.failure_kind is None
            assert ResReportE2ESource_811 in matches[0].result.identified

        # Traversal order: a requested aggregation is recorded before its own derived source.
        names = [record.feature_name for record in report_after]
        assert names.index(SUM_FEATURE) < names.index(SALES_COL)
        assert names.index(AVG_FEATURE) < names.index(REVENUE_COL)

        # Sibling cross-check: every requested=True record names a user-requested feature.
        requested_names = {record.feature_name for record in report_after if record.requested}
        assert requested_names == {SUM_FEATURE, AVG_FEATURE}

        # Unchanged by run(), mirroring resolved_plan()'s contract on the same session.
        assert report_after == report_before
        assert session.resolved_plan() == plan_before
