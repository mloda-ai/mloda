"""
Spark ASOF Merge Engine Tests

This module tests the ASOF (point-in-time) merge implementation of the Spark
merge engine.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 17+ must be installed and JAVA_HOME environment variable must be set

The shared ASOF scenarios run through AsofMergeEngineSuite; the tests below are Spark specific.
They use the shared session-scoped SparkSession fixture to avoid Java gateway conflicts.
"""

from datetime import timedelta
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
import pytest

from mloda._testing.compute_framework.merge_suite import AsofMergeEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, SparkAdapter
from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.spark.spark_merge_engine import (
    SparkMergeEngine,
    spark_name_fold,
)

from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

import logging

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkAsofMergeEngine:
    def test_nearest_raises_value_error(self, spark_session: Any) -> None:
        """direction='nearest' is unsupported and must raise ValueError mentioning nearest."""
        left_data = spark_session.createDataFrame([{"k": 1, "t": 10, "lv": 100}])
        right_data = spark_session.createDataFrame([{"k": 1, "t": 5, "rv": 1}])

        engine = SparkMergeEngine(spark_session)
        with pytest.raises(ValueError, match="nearest"):
            engine.merge_asof(
                left_data,
                right_data,
                Index(("k",)),
                Index(("k",)),
                AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="nearest"),
            )

    @pytest.mark.parametrize(
        "extra",
        [
            "_mloda_lid",
            "_MLODA_LID",
            "_mloda_lid0__",
            "_MLODA_LID0__",
            "_mloda_rn0__",
            "_MLODA_RN0__",
            "_mloda_lıd0__",
            "_mloda_r_rv",
            "_mloda_r0_rv",
        ],
    )
    def test_left_column_named_like_internal_helper_is_kept(self, spark_session: Any, extra: str) -> None:
        left_data = spark_session.createDataFrame(
            [{"k": 1, "t": 10, "lv": 100, extra: "a"}, {"k": 1, "t": 20, "lv": 200, extra: "b"}]
        )
        right_data = spark_session.createDataFrame([{"k": 1, "t": 5, "rv": 1}, {"k": 1, "t": 15, "rv": 2}])

        result = SparkMergeEngine(spark_session).merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward"),
        )

        rows = {r["t"]: (r[extra], r["rv"]) for r in result.collect()}
        assert rows == {10: ("a", 1), 20: ("b", 2)}

    @pytest.mark.parametrize("extra", ["_mloda_r_rv", "_mloda_r0_rv"])
    def test_right_column_named_like_renamed_column_keeps_positional_values(
        self, spark_session: Any, extra: str
    ) -> None:
        left_data = spark_session.createDataFrame([{"k": 1, "t": 10, "lv": 100}])
        right_data = spark_session.createDataFrame([(1, 5, 7, 8)], ["k", "t", "rv", extra])

        result = SparkMergeEngine(spark_session).merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward"),
        )

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["rv"] == 7
        assert rows[0][extra] == 8

    def test_right_column_case_folds_into_rename_prefix_keeps_values(self, spark_session: Any) -> None:
        left_data = spark_session.createDataFrame([(1, 10, 100, "left")], ["k", "t", "lv", "_mloda_r0_é"])
        right_data = spark_session.createDataFrame([(1, 5, "right")], ["k", "t", "É"])

        result = SparkMergeEngine(spark_session).merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward"),
        )

        rows = result.collect()
        assert len(rows) == 1
        assert rows[0]["_mloda_r0_é"] == "left"
        assert rows[0]["É"] == "right"

    @pytest.mark.parametrize("left_name,right_name", [("val", "Val"), ("é", "É")])
    def test_case_only_output_collision_raises(self, spark_session: Any, left_name: str, right_name: str) -> None:
        left_data = spark_session.createDataFrame([{"k": 1, "t": 10, left_name: 1}])
        right_data = spark_session.createDataFrame([{"k": 1, "t": 5, right_name: 2}])

        with pytest.raises(ValueError, match="rename"):
            SparkMergeEngine(spark_session).merge_asof(
                left_data,
                right_data,
                Index(("k",)),
                Index(("k",)),
                AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward"),
            )

    def test_case_only_output_collision_kept_when_case_sensitive(self, spark_case_sensitive: Any) -> None:
        spark = spark_case_sensitive
        left_data = spark.createDataFrame([{"k": 1, "t": 10, "val": 1}])
        right_data = spark.createDataFrame([{"k": 1, "t": 5, "Val": 2}])

        result = SparkMergeEngine(spark).merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward"),
        )

        rows = result.collect()
        assert len(rows) == 1
        assert {"val", "Val"} <= set(result.columns)
        assert rows[0]["val"] == 1
        assert rows[0]["Val"] == 2


class TestSparkAsofTimedeltaGuard:
    """Guard test for a timedelta tolerance on the Spark backend.

    Deliberately NOT decorated with @skipif(not PYSPARK_AVAILABLE): the guard must fire
    BEFORE check_import() and before any DataFrame access, so it is testable without a
    running SparkSession (and without JAVA_HOME). We construct the engine directly and
    pass None DataFrames; the timedelta guard at the top of merge_asof must raise a clear
    ValueError mentioning 'timedelta' before anything touches the (None) data.
    """

    def test_timedelta_tolerance_raises_value_error_without_session(self) -> None:
        engine = SparkMergeEngine()
        cfg = AsOfJoinConfig(
            left_time_column="t",
            right_time_column="t",
            direction="backward",
            tolerance=timedelta(seconds=5),
        )
        with pytest.raises(ValueError, match="timedelta"):
            engine.merge_asof(None, None, Index(("k",)), Index(("k",)), cfg)


class TestSparkNameFold:
    """spark_name_fold reads spark.sql.caseSensitive; testable with a stub, no session needed."""

    @staticmethod
    def _stub(value: Any) -> Any:
        conf = SimpleNamespace(get=lambda key, *a: value)
        return SimpleNamespace(sparkSession=SimpleNamespace(conf=conf))

    def test_case_insensitive_folds_to_lower(self) -> None:
        assert spark_name_fold(self._stub("false"))("ValÉ") == "valé"

    @pytest.mark.parametrize("value", ["true", "TRUE", " true "])
    def test_case_sensitive_is_identity(self, value: str) -> None:
        assert spark_name_fold(self._stub(value))("ValÉ") == "ValÉ"


class TestSparkAsofMergeEngineSuite(AsofMergeEngineSuite):
    adapter_class = SparkAdapter

    @pytest.fixture
    def adapter(self, spark_session: Any) -> Iterator[FrameworkAdapter]:
        """SparkAdapter borrows the active session, so spark_session must exist first."""
        spark_adapter = self.adapter_class()
        with spark_adapter:
            yield spark_adapter
