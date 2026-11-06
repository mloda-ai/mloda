"""End-to-end empty-result policy tests for polars (eager and lazy) on the shipped RunAllSuite.

Both a zero-row eager ``pl.DataFrame`` and a lazy ``pl.LazyFrame`` still carry their schema (columns), so an empty
schema-bearing result is valid.
"""

import logging
import traceback
from collections.abc import Mapping
from typing import Any

from mloda._testing.compute_framework.run_all_suite import (
    UNDECLARED_COLUMN,
    EmptyResultRunAllSuite,
    MissingColumnRunAllSuite,
    PolicyRaises,
    PolicySuccess,
    RunAllSuite,
    SiblingDictRunAllSuite,
    CauseType,
    UndeclaredRead,
    attribute_read,
)
from mloda._testing.framework_adapter import FrameworkAdapter, PolarsAdapter, PolarsLazyAdapter
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.provider import DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.user import Feature, GlobalFilter, Options, ParallelizationMode, PluginCollector
from tests.helpers.frame_records import records_from_frame
from tests.helpers.lost_input_feature_groups import SiblingDictRoot

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore[assignment]


class TestPolarsAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on eager polars."""

    adapter_class = PolarsAdapter


class TestPolarsLazyAllowEmptyResultRunAll(EmptyResultRunAllSuite):
    """Drives the empty-result policy end-to-end through run_all on lazy polars."""

    adapter_class = PolarsLazyAdapter


class TestPolarsSiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all."""

    adapter_class = PolarsAdapter


class TestPolarsLazySiblingDictRunAll(SiblingDictRunAllSuite):
    """Dict-returning sibling consumers end-to-end through run_all on lazy polars."""

    adapter_class = PolarsLazyAdapter


def _with_columns_read(data: Any) -> Any:
    return data.with_columns(pl.col(UNDECLARED_COLUMN))


def _column_not_found() -> type[BaseException]:
    import polars as pl

    return pl.exceptions.ColumnNotFoundError


class TestPolarsMissingColumnRunAll(MissingColumnRunAllSuite):
    """Undeclared column reads on eager polars raise an explanatory error."""

    adapter_class = PolarsAdapter

    @classmethod
    def undeclared_reads(cls) -> Mapping[str, tuple[UndeclaredRead, CauseType]]:
        return {
            "PolarsWithColumnsReadConsumer": (_with_columns_read, _column_not_found),
            "AttributeReadConsumer": (attribute_read, lambda: AttributeError),
        }


class TestPolarsLazyMissingColumnRunAll(TestPolarsMissingColumnRunAll):
    """Undeclared column reads on lazy polars raise an explanatory error."""

    adapter_class = PolarsLazyAdapter

    def test_undeclared_column_read_with_global_filter_raises_explanatory_error(
        self, adapter: FrameworkAdapter, flight_server: Any
    ) -> None:
        case = "PolarsWithColumnsReadConsumer"
        global_filter = GlobalFilter()
        global_filter.add_filter(case, "min", {"mc_declared": 0})
        error = self.assert_undeclared_read_raises(
            adapter, case, ParallelizationMode.SYNC, flight_server, global_filter
        )
        cause = error.__cause__
        assert cause is not None
        frames = [frame.name for frame in traceback.extract_tb(cause.__traceback__)]
        assert "_validate_filter_columns" in frames


class SingleSeriesOnRoot(FeatureGroup):
    """Non-root consumer of ``sd_a`` returning a single ``pl.Series``."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("sd_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.Series([v * 2 for v in data.collect()["sd_a"].to_list()])


class NativeFrameWithColumn(FeatureGroup):
    """Non-root consumer returning its own native lazy frame, which mloda did not build."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("sd_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.with_columns((pl.col("sd_a") * 2).alias(cls.get_class_name()))


class SingleSeriesOnNativeFrame(FeatureGroup):
    """Non-root consumer returning a single ``pl.Series`` onto a frame a feature group returned."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("NativeFrameWithColumn")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.Series(data.collect()["NativeFrameWithColumn"].to_list())


class DataFrameRoot(DataCreatorFeatureGroup):
    """Root returning an eager ``pl.DataFrame``, which lazy Polars converts through ``transform``."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"df_a"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.DataFrame({"df_a": [1, 2, 3]})


class SingleSeriesOnConvertedRoot(FeatureGroup):
    """Non-root consumer returning a single ``pl.Series`` onto a converted root frame."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("df_a")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.Series([v * 2 for v in data.collect()["df_a"].to_list()])


_SINGLE_COLUMN_COLLECTOR = PluginCollector.enabled_feature_groups(
    {
        SiblingDictRoot,
        SingleSeriesOnRoot,
        NativeFrameWithColumn,
        SingleSeriesOnNativeFrame,
        DataFrameRoot,
        SingleSeriesOnConvertedRoot,
    }
)


class TestPolarsLazySingleColumnRunAll(RunAllSuite):
    """A single returned column needs a frame mloda built; a feature group's own native frame is not one."""

    adapter_class = PolarsLazyAdapter

    def test_series_on_dict_root_frame_succeeds_aligned(self, adapter: FrameworkAdapter, flight_server: Any) -> None:
        def check(result: Any) -> None:
            rows = [row for frame in result for row in records_from_frame(frame)]
            assert [row["SingleSeriesOnRoot"] for row in rows] == [2, 4, 6]

        self.assert_policy_case(
            adapter,
            feature_name="SingleSeriesOnRoot",
            plugin_collector=_SINGLE_COLUMN_COLLECTOR,
            expectation=PolicySuccess(assert_result=check),
            mode=ParallelizationMode.SYNC,
            flight_server=flight_server,
        )

    def test_series_on_converted_dataframe_root_succeeds_aligned(
        self, adapter: FrameworkAdapter, flight_server: Any
    ) -> None:
        def check(result: Any) -> None:
            rows = [row for frame in result for row in records_from_frame(frame)]
            assert [row["SingleSeriesOnConvertedRoot"] for row in rows] == [2, 4, 6]

        self.assert_policy_case(
            adapter,
            feature_name="SingleSeriesOnConvertedRoot",
            plugin_collector=_SINGLE_COLUMN_COLLECTOR,
            expectation=PolicySuccess(assert_result=check),
            mode=ParallelizationMode.SYNC,
            flight_server=flight_server,
        )

    def test_series_on_native_frame_of_a_feature_group_raises(
        self, adapter: FrameworkAdapter, flight_server: Any
    ) -> None:
        self.assert_policy_case(
            adapter,
            feature_name="SingleSeriesOnNativeFrame",
            plugin_collector=_SINGLE_COLUMN_COLLECTOR,
            expectation=PolicyRaises("SingleSeriesOnNativeFrame"),
            mode=ParallelizationMode.SYNC,
            flight_server=flight_server,
        )
