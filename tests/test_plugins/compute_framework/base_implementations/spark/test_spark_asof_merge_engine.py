"""
Spark ASOF Merge Engine Tests

This module tests the ASOF (point-in-time) merge implementation of the Spark
merge engine.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

The data shapes mirror the shared ASOF scenarios in
tests/test_plugins/compute_framework/test_tooling/asof/asof_scenarios.py so the
expected values are easy to reason about. The tests use the shared session-scoped
SparkSession fixture to avoid Java gateway conflicts.
"""

from typing import Any
import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda_plugins.compute_framework.base_implementations.spark.spark_merge_engine import SparkMergeEngine

from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

import logging

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkAsofMergeEngine:
    def test_backward_single_key(self, spark_session: Any) -> None:
        """Backward, single by-key. Right rows shuffled to prove internal ordering."""
        left_data = spark_session.createDataFrame(
            [{"k": 1, "t": 10, "lv": 100}, {"k": 1, "t": 20, "lv": 200}, {"k": 2, "t": 15, "lv": 300}]
        )
        right_data = spark_session.createDataFrame(
            [
                {"k": 2, "t": 30, "rv": 4},
                {"k": 1, "t": 5, "rv": 1},
                {"k": 1, "t": 18, "rv": 2},
                {"k": 2, "t": 5, "rv": 3},
            ]
        )

        engine = SparkMergeEngine(spark_session)
        result = engine.merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward"),
        )

        rows = {(r["k"], r["t"]): r["rv"] for r in result.collect()}
        assert rows == {(1, 10): 1, (1, 20): 2, (2, 15): 3}

    def test_forward_single_key(self, spark_session: Any) -> None:
        """Forward: row (k=1, t=20) has no right_time >= 20 -> null."""
        left_data = spark_session.createDataFrame(
            [{"k": 1, "t": 10, "lv": 100}, {"k": 1, "t": 20, "lv": 200}, {"k": 2, "t": 15, "lv": 300}]
        )
        right_data = spark_session.createDataFrame(
            [
                {"k": 1, "t": 5, "rv": 1},
                {"k": 1, "t": 18, "rv": 2},
                {"k": 2, "t": 5, "rv": 3},
                {"k": 2, "t": 30, "rv": 4},
            ]
        )

        engine = SparkMergeEngine(spark_session)
        result = engine.merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward"),
        )

        rows = {(r["k"], r["t"]): r["rv"] for r in result.collect()}
        assert rows == {(1, 10): 2, (1, 20): None, (2, 15): 4}

    def test_allow_exact_matches_true(self, spark_session: Any) -> None:
        """Backward + allow_exact_matches=True -> exact-time row matched (rv=99)."""
        left_data = spark_session.createDataFrame([{"k": 1, "t": 10, "lv": 100}])
        right_data = spark_session.createDataFrame([{"k": 1, "t": 10, "rv": 99}, {"k": 1, "t": 5, "rv": 1}])

        engine = SparkMergeEngine(spark_session)
        result = engine.merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=True),
        )

        collected = result.collect()
        assert len(collected) == 1
        assert collected[0]["rv"] == 99

    def test_allow_exact_matches_false(self, spark_session: Any) -> None:
        """Backward + allow_exact_matches=False -> exact excluded, prior row (rv=1)."""
        left_data = spark_session.createDataFrame([{"k": 1, "t": 10, "lv": 100}])
        right_data = spark_session.createDataFrame([{"k": 1, "t": 10, "rv": 99}, {"k": 1, "t": 5, "rv": 1}])

        engine = SparkMergeEngine(spark_session)
        result = engine.merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(
                left_time_column="t", right_time_column="t", direction="backward", allow_exact_matches=False
            ),
        )

        collected = result.collect()
        assert len(collected) == 1
        assert collected[0]["rv"] == 1

    def test_tolerance_numeric(self, spark_session: Any) -> None:
        """Backward, tolerance=5 -> row t=100 gap 92 > 5 -> null right value."""
        left_data = spark_session.createDataFrame([{"k": 1, "t": 10, "lv": 1}, {"k": 1, "t": 100, "lv": 2}])
        right_data = spark_session.createDataFrame([{"k": 1, "t": 8, "rv": 7}])

        engine = SparkMergeEngine(spark_session)
        result = engine.merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=5),
        )

        rows = {r["t"]: r["rv"] for r in result.collect()}
        assert rows == {10: 7, 100: None}

    def test_tolerance_none(self, spark_session: Any) -> None:
        """Backward, tolerance=None -> both rows match (rv=7, rv=7)."""
        left_data = spark_session.createDataFrame([{"k": 1, "t": 10, "lv": 1}, {"k": 1, "t": 100, "lv": 2}])
        right_data = spark_session.createDataFrame([{"k": 1, "t": 8, "rv": 7}])

        engine = SparkMergeEngine(spark_session)
        result = engine.merge_asof(
            left_data,
            right_data,
            Index(("k",)),
            Index(("k",)),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward", tolerance=None),
        )

        rows = {r["t"]: r["rv"] for r in result.collect()}
        assert rows == {10: 7, 100: 7}

    def test_multi_by_key(self, spark_session: Any) -> None:
        """Multi by-key (k1, k2), backward."""
        left_data = spark_session.createDataFrame(
            [{"k1": 1, "k2": "a", "t": 10, "lv": 1}, {"k1": 1, "k2": "b", "t": 10, "lv": 2}]
        )
        right_data = spark_session.createDataFrame(
            [{"k1": 1, "k2": "a", "t": 5, "rv": 10}, {"k1": 1, "k2": "b", "t": 5, "rv": 20}]
        )

        engine = SparkMergeEngine(spark_session)
        result = engine.merge_asof(
            left_data,
            right_data,
            Index(("k1", "k2")),
            Index(("k1", "k2")),
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward"),
        )

        rows = {(r["k1"], r["k2"]): r["rv"] for r in result.collect()}
        assert rows == {(1, "a"): 10, (1, "b"): 20}

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
