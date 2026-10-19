"""End-to-end ``allow_empty_result`` policy tests for polars.

Two classes: one for the eager PolarsDataFrame, one for the lazy PolarsLazyDataFrame.
Both consume the shared EmptyResultRunAllTestBase.

Both a zero-row eager ``pl.DataFrame`` and a lazy ``pl.LazyFrame`` still carry their schema
(columns), so an empty schema-bearing result is valid (state C -> success). The subclasses
below exercise the end-to-end behavior through ``run_all`` (they do not assert a native
representation).
"""

from typing import Any

import pytest

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import (
    PolicyRaises,
    PolicyRunAllTestBase,
    PolicySuccess,
    records_from_frame,
)
from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import _EmptyResultMatchData
from tests.test_plugins.compute_framework.test_tooling.sibling_dict_run_all_test_base import SiblingDictRoot
from tests.test_plugins.compute_framework.test_tooling.sibling_dict_run_all_test_base import (
    SiblingDictRunAllTestBase,
)
from tests.test_plugins.compute_framework.test_tooling.empty_result_run_all_test_base import (
    EmptyResultRunAllTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore[assignment]


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on eager polars."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsDataFrame"


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyAllowEmptyResultRunAll(EmptyResultRunAllTestBase):
    """Drives the allow_empty_result policy end-to-end through run_all on lazy polars."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsLazyDataFrame"


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsSiblingDictRunAll(SiblingDictRunAllTestBase):
    """Dict-returning sibling consumers end-to-end through run_all."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsDataFrame"


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazySiblingDictRunAll(SiblingDictRunAllTestBase):
    """Dict-returning sibling consumers end-to-end through run_all on lazy polars."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsLazyDataFrame"


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


class DataFrameRoot(FeatureGroup, _EmptyResultMatchData):
    """Root returning an eager ``pl.DataFrame``, which lazy Polars converts through ``transform``."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"df_a"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.DataFrame({"df_a": [1, 2, 3]})

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"df_a"}


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


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazySingleColumnRunAll(PolicyRunAllTestBase):
    """A single returned column needs a frame mloda built; a feature group's own native frame is not one."""

    @classmethod
    def compute_framework_name(cls) -> str:
        return "PolarsLazyDataFrame"

    def test_series_on_dict_root_frame_succeeds_aligned(self, flight_server: Any) -> None:
        def check(result: Any) -> None:
            rows = [row for frame in result for row in records_from_frame(frame)]
            assert [row["SingleSeriesOnRoot"] for row in rows] == [2, 4, 6]

        self.assert_policy_case(
            feature_name="SingleSeriesOnRoot",
            plugin_collector=_SINGLE_COLUMN_COLLECTOR,
            expectation=PolicySuccess(assert_result=check),
            mode=ParallelizationMode.SYNC,
            flight_server=flight_server,
        )

    def test_series_on_converted_dataframe_root_succeeds_aligned(self, flight_server: Any) -> None:
        def check(result: Any) -> None:
            rows = [row for frame in result for row in records_from_frame(frame)]
            assert [row["SingleSeriesOnConvertedRoot"] for row in rows] == [2, 4, 6]

        self.assert_policy_case(
            feature_name="SingleSeriesOnConvertedRoot",
            plugin_collector=_SINGLE_COLUMN_COLLECTOR,
            expectation=PolicySuccess(assert_result=check),
            mode=ParallelizationMode.SYNC,
            flight_server=flight_server,
        )

    def test_series_on_native_frame_of_a_feature_group_raises(self, flight_server: Any) -> None:
        self.assert_policy_case(
            feature_name="SingleSeriesOnNativeFrame",
            plugin_collector=_SINGLE_COLUMN_COLLECTOR,
            expectation=PolicyRaises("SingleSeriesOnNativeFrame"),
            mode=ParallelizationMode.SYNC,
            flight_server=flight_server,
        )
